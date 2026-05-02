"""Gateway runner for tyagent.

Manages platform adapters, routes messages to the AI agent,
and handles session lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tyagent.agent import AgentError, TyAgent
from tyagent.types import ReplyTarget
from tyagent.config import PlatformConfig, TyAgentConfig
from tyagent.gateway.commands import CommandRegistry
from tyagent.gateway.consumer import StreamConsumer
from tyagent.gateway.lifecycle import GatewaySupervisor
from tyagent.gateway.progress import ProgressSender
from tyagent.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from tyagent.session import SessionStore
from tyagent.tools import memory_tool
from tyagent.tools import search_tool
from tyagent.tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message sanitisation (module-level helper)
# ---------------------------------------------------------------------------


def _sanitize_message_chain(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Iteratively fix message chain so it stays valid for the LLM API.

    Applies fixes in a loop until the chain is fully clean:

    1. Removes assistant messages with neither content nor tool_calls
    2. Inserts synthetic tool responses after orphaned tool_calls

    The original message objects are NOT mutated — the function returns
    a new list with copies of affected messages.
    """
    if not messages:
        return messages

    result = list(messages)  # shallow copy
    while True:
        changed = False
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]

            # Fix 1: empty assistant message → remove
            if (
                msg.get("role") == "assistant"
                and not msg.get("content")
                and not msg.get("tool_calls")
            ):
                logger.info(
                    "Sanitized empty assistant message [%d] "
                    "(no content, no tool_calls) — removing",
                    i,
                )
                result = result[:i] + result[i + 1 :]
                changed = True
                break  # re-scan from end after mutation

            # Fix 2: orphaned tool_calls → insert synthetic tool responses
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                n_expected = len(tool_calls)
                n_actual = 0
                j = i + 1
                while j < len(result) and result[j].get("role") == "tool":
                    n_actual += 1
                    j += 1
                if n_actual < n_expected:
                    logger.info(
                        "Sanitized orphaned tool_calls from assistant "
                        "message [%d] (%d tool_calls, only %d tool "
                        "responses) — inserting %d synthetic",
                        i,
                        n_expected,
                        n_actual,
                        n_expected - n_actual,
                    )
                    synthetic = []
                    for tc in tool_calls[n_actual:]:
                        tc_id = (
                            tc.get("id", "unknown")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", "unknown")
                        )
                        synthetic.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "(tool call interrupted — "
                                "gateway restarted or crashed)",
                            }
                        )
                    result = (
                        result[: i + 1 + n_actual]
                        + synthetic
                        + result[i + 1 + n_actual :]
                    )
                    changed = True
                    break  # re-scan after mutation

        if not changed:
            break

    return result


# ---------------------------------------------------------------------------
# Platform adapter registry (module-level)
# ---------------------------------------------------------------------------

_PLATFORM_REGISTRY: Dict[str, Type[BasePlatformAdapter]] = {}


def register_platform(name: str, adapter_class: Type[BasePlatformAdapter]) -> None:
    """Register a platform adapter class under *name* (e.g. 'feishu')."""
    _PLATFORM_REGISTRY[name] = adapter_class


def _load_builtin_platforms() -> None:
    """Load built-in platform adapters by importing them so they self-register."""
    try:
        from tyagent.platforms.feishu import FeishuAdapter
        _PLATFORM_REGISTRY["feishu"] = FeishuAdapter
    except ImportError as exc:
        logger.debug("Feishu adapter not available: %s", exc)
    except Exception as exc:
        logger.error("Failed to load feishu adapter: %s", exc)


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class Gateway:
    """Main gateway managing platform adapters and AI agent interactions."""

    def __init__(
        self,
        config: TyAgentConfig,
        *,
        session_store: Optional[SessionStore] = None,
        agent: Optional[TyAgent] = None,
    ):
        self.config = config
        self.adapters: Dict[str, BasePlatformAdapter] = {}
        self.session_store = session_store or SessionStore(
            sessions_dir=config.sessions_dir
        )
        self._agent_cache: OrderedDict[str, TyAgent] = OrderedDict()
        self._AGENT_CACHE_MAX_SIZE = 100
        if agent is not None:
            self._agent_cache["_default"] = agent
        # Initialize persistent memory store
        memories_dir = config.home_dir / "memories"
        self.memory_store = memory_tool.MemoryStore(memories_dir)
        memory_tool.set_store(self.memory_store)
        # Wire up session search tool
        search_tool.set_search_db(self.session_store.db)
        self._running = False
        self._shutdown_event = asyncio.Event()
        # Graceful restart / drain support
        self._restart_requested = False
        self._draining = False
        self._active_sessions: set[str] = set()
        self._restart_drain_timeout: float = 60.0
        self._restart_requestor: Optional[Dict[str, str]] = None
        self._restart_notification_pending: Optional[Dict[str, str]] = None
        self._session_to_adapter: Dict[str, str] = {}
        self._active_chat_ids: Dict[str, str] = {}
        # Actor-model session agents
        self._session_agents: Dict[str, TyAgent] = {}
        self._session_adapters: Dict[str, Any] = {}
        self._session_output_tasks: Dict[str, asyncio.Task] = {}
        # Subsystems
        self.commands = CommandRegistry(self)
        self.supervisor = GatewaySupervisor(self)

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def _load_adapters(self) -> None:
        """Load and initialize platform adapters from config."""
        _load_builtin_platforms()
        connected = self.config.get_connected_platforms()
        for name in connected:
            platform_cfg = self.config.get_platform(name)
            if not platform_cfg:
                continue
            adapter_cls = _PLATFORM_REGISTRY.get(name)
            if adapter_cls is None:
                logger.warning("No adapter registered for platform: %s", name)
                continue
            try:
                adapter = adapter_cls(platform_cfg)
                adapter.set_message_handler(self._on_message)
                self.adapters[name] = adapter
                logger.info("Loaded adapter: %s", name)
            except Exception as exc:
                logger.error("Failed to load adapter %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def _get_or_create_agent(self, session_key: str) -> TyAgent:
        """Get cached agent for session, or create new one."""
        # If session_key exists in cache, return it (and move to end for LRU)
        if session_key in self._agent_cache:
            agent = self._agent_cache[session_key]
            self._agent_cache.move_to_end(session_key)
            return agent

        # If a default agent was explicitly provided, use it for all sessions.
        # ⚠️ TyAgent mutates instance state (_prev_msg_count, last_usage) during
        # chat(), creating a race condition with concurrent sessions.
        # For single-session use (e.g. CLI), this is fine.
        if "_default" in self._agent_cache:
            agent = self._agent_cache["_default"]
            self._agent_cache.move_to_end("_default")
            return agent

        # Create a new per-session agent
        agent = TyAgent.from_config(self.config.agent)
        self._agent_cache[session_key] = agent
        self._agent_cache.move_to_end(session_key)
        # Evict oldest if over cap
        while len(self._agent_cache) > self._AGENT_CACHE_MAX_SIZE:
            lru_key = next(iter(self._agent_cache))
            lru_agent = self._agent_cache.pop(lru_key)
            loop = asyncio.get_running_loop()
            loop.create_task(lru_agent.close())
        return agent

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def _on_message(self, event: MessageEvent) -> Optional[str]:
        """Handle an incoming message event."""
        adapter = self._find_adapter_for_event(event)
        if not adapter:
            logger.warning(
                "No adapter found for event from platform: %s", event.platform
            )
            return None

        session_key = adapter.build_session_key(event)

        # If draining (graceful restart), reject incoming messages
        if self._draining:
            logger.info("Rejecting message during drain for %s", session_key)
            await adapter.send_message(
                event.chat_id or "",
                "⏳ Gateway is restarting. Please try again shortly.",
                reply_to_message_id=event.message_id,
            )
            return None

        # If session is suspended (crash recovery), archive it and create fresh
        if self.session_store.is_suspended(session_key):
            logger.info(
                "Session %s was suspended — archiving and creating fresh session",
                session_key,
            )
            self.session_store.archive(session_key)
            self.session_store.clear_resume_pending(session_key)
            session = self.session_store.get_or_create_after_archive(session_key)
            await adapter.send_message(
                event.chat_id or "",
                "🔄 检测到异常中断，已为您恢复新会话。之前的对话已归档保留。",
                reply_to_message_id=event.message_id,
            )
        else:
            session = self.session_store.get(session_key)

        # Dispatch built-in commands via registry
        if event.is_command():
            cmd = event.get_command()
            if cmd:
                result = await self.commands.dispatch(
                    cmd, adapter, event, session_key, session
                )
                if result is not None:
                    return result

        # Build message for LLM
        user_message = event.text
        if event.media_urls:
            media_desc = "\n".join(
                f"[Attached {mt or 'file'}: {url}]"
                for mt, url in zip(event.media_types or [], event.media_urls)
            )
            user_message = (
                f"{user_message}\n\n{media_desc}" if user_message else media_desc
            )

        # Persist user message to DB
        session.add_message("user", user_message)

        # Build tool definitions
        tool_defs = registry.get_definitions()

        # Track this session as actively processing
        self._active_sessions.add(session_key)
        self._session_to_adapter[session_key] = adapter.platform_name
        self._active_chat_ids[session_key] = event.chat_id or ""
        self._session_adapters[session_key] = adapter

        try:
            # Define the persistence callback for tool loop messages
            _persist_sid = getattr(session.metadata, "current_session_id", None) or \
                           getattr(session, "current_session_id", None)

            def persist_message(role: str, content: str, **extras) -> None:
                self.session_store.add_message(
                    session_key, role, content,
                    session_id=_persist_sid, **extras,
                )

            # Create progress sender
            progress_sender = ProgressSender(
                adapter, event.chat_id or "",
                reply_to_message_id=event.message_id,
                enabled=bool(event.chat_id) and not event.is_command(),
            )
            progress_task = asyncio.create_task(progress_sender.run())
            # Suppress "Task exception was never retrieved" warning and log
            # any unhandled exception. ProgressSender.run() handles its own
            # exceptions, so this is belt-and-suspenders.
            def _on_progress_done(t: asyncio.Task) -> None:
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    logger.error("ProgressSender task crashed: %s", exc)
            progress_task.add_done_callback(_on_progress_done)
            _progress_cb = progress_sender.on_tool_started

            # Ensure session agent is running (starts loop if new session)
            agent = await self._ensure_session_agent(
                session_key, session,
                adapter, event.chat_id or "",
                persist_message,
            )

            # Inject memory into user message
            memory_block = self.memory_store.get_all_formatted()
            final_text = user_message
            if memory_block:
                final_text = f"{user_message}\n\n[记忆上下文]\n{memory_block}"

            # Send message to agent loop (fire-and-forget), with per-message
            # progress callbacks so ProgressSender stays alive until the
            # agent's turn completes.
            reply_target = ReplyTarget(
                platform=adapter.platform_name,
                chat_id=event.chat_id or "",
                message_id=event.message_id or "",
            )
            # Capture progress_sender in closure so turn_done finishes it
            _ps = progress_sender
            def _turn_done() -> None:
                _ps.finish()

            await agent.send_message(
                final_text,
                reply_target=reply_target,
                tool_progress_cb=_progress_cb,
                turn_done_cb=_turn_done,
            )

        except AgentError as exc:
            logger.error("Agent error: %s", exc)
            progress_sender.finish()
            try: await progress_task
            except: pass
            await adapter.send_message(
                event.chat_id or "", f"❌ 错误: {exc}",
                reply_to_message_id=event.message_id,
            )
        except Exception:
            logger.exception("Unexpected agent error")
            progress_sender.finish()
            try: await progress_task
            except: pass
            await adapter.send_message(
                event.chat_id or "", "Sorry, something went wrong.",
                reply_to_message_id=event.message_id,
            )

        return None

    # ------------------------------------------------------------------
    # Actor-model session agent management
    # ------------------------------------------------------------------

    async def _ensure_session_agent(
        self, session_key: str, session,
        adapter, chat_id: str,
        persist_message: Callable,
    ) -> TyAgent:
        """Get or create agent with running loop. Starts _consume_output."""
        if session_key in self._session_agents:
            return self._session_agents[session_key]

        agent = self._get_or_create_agent(session_key)

        # Start the permanent agent loop with history and persistence
        await agent.start(
            history=session.messages,
            on_message=persist_message,
        )

        self._session_agents[session_key] = agent
        self._session_adapters[session_key] = adapter

        # Start background output consumer
        self._session_output_tasks[session_key] = asyncio.create_task(
            self._consume_output(session_key)
        )

        return agent

    async def _consume_output(self, session_key: str):
        """Long-running task: consume agent outputs and send to platform."""
        agent = self._session_agents.get(session_key)
        if agent is None:
            return

        while self._running:
            try:
                output = await asyncio.wait_for(agent._output_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            adapter = self._session_adapters.get(session_key)
            chat_id = self._active_chat_ids.get(session_key, "")
            if adapter is None:
                continue

            if output.reply_target:
                # User-message-driven reply
                await adapter.send_message(
                    output.reply_target.chat_id,
                    output.text,
                    reply_to=output.reply_target.message_id,
                )
            else:
                # Auto-reply (child completion triggered)
                await adapter.send_message(chat_id, output.text)

            # Persist to session store
            try:
                self.session_store.add_message(session_key, "assistant", output.text)
            except Exception:
                logger.exception("Failed to persist output for %s", session_key)

    async def _stop_session_agent(self, session_key: str):
        """Stop and clean up a session's agent loop and consumer."""
        task = self._session_output_tasks.pop(session_key, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        agent = self._session_agents.pop(session_key, None)
        if agent is not None:
            await agent.stop()
        self._session_adapters.pop(session_key, None)
        self._active_sessions.discard(session_key)
        self._session_to_adapter.pop(session_key, None)
        self._active_chat_ids.pop(session_key, None)

    def _find_adapter_for_event(
        self, event: MessageEvent
    ) -> Optional[BasePlatformAdapter]:
        """Find the adapter responsible for this event's platform."""
        return self.adapters.get(event.platform)

    def set_restart_requestor(self, platform: str, chat_id: str) -> None:
        """Record restart request info so the new process can send a notification."""
        self._restart_requestor = {"platform": platform, "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all adapters and run the gateway."""
        self._load_adapters()
        self.supervisor.setup_signal_handlers()
        self.supervisor.check_recovery_on_startup()
        if not self.adapters:
            logger.error("No adapters loaded. Check your configuration.")
            return

        self._running = True
        logger.info(
            "Starting tyagent gateway with %d adapter(s)", len(self.adapters)
        )

        # Start all adapters
        tasks = []
        for name, adapter in self.adapters.items():
            task = asyncio.create_task(
                self._run_adapter_with_retry(name, adapter),
                name=f"adapter-{name}",
            )
            tasks.append(task)

        # If there's a pending restart notification, schedule it
        GatewaySupervisor.schedule_restart_notification(self)

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Stop all adapters
        logger.info("Shutting down gateway...")
        for name, adapter in self.adapters.items():
            try:
                await asyncio.wait_for(adapter.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Adapter %s stop timed out", name)

        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Stop all session agents (actor model loops)
        for session_key in list(self._session_agents.keys()):
            await self._stop_session_agent(session_key)

        # Close all cached agents
        for key, cached_agent in list(self._agent_cache.items()):
            try:
                await cached_agent.close()
            except Exception:
                pass
        self._agent_cache.clear()
        self.session_store.close()
        logger.info("Gateway stopped")

    async def _run_adapter_with_retry(
        self, name: str, adapter: BasePlatformAdapter
    ) -> None:
        """Run a single adapter with exponential backoff retry."""
        max_retries = 10
        retry_delay = 5.0
        max_retry_delay = 60.0

        for attempt in range(1, max_retries + 1):
            if not self._running:
                break
            try:
                logger.info(
                    "Starting adapter %s (attempt %d/%d)",
                    name,
                    attempt,
                    max_retries,
                )
                await adapter.start()
                logger.info("Adapter %s stopped gracefully", name)
                break
            except asyncio.CancelledError:
                logger.info("Adapter %s task cancelled", name)
                break
            except Exception as exc:
                logger.error("Adapter %s crashed: %s", name, exc)
                if not self._running or attempt >= max_retries:
                    logger.error(
                        "Adapter %s exceeded max retries, giving up", name
                    )
                    break
                wait = min(retry_delay * (2 ** (attempt - 1)), max_retry_delay)
                logger.info(
                    "Retrying adapter %s in %.1f seconds...", name, wait
                )
                await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_gateway(config_path: Optional[str] = None) -> None:
    """Entry point to start the gateway."""
    from pathlib import Path

    from tyagent.config import load_config

    config = load_config(Path(config_path) if config_path else None)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Initialize profile home directory (isolated shell home for git/ssh/etc)
    profile_home = config.home_dir / "home"
    profile_home.mkdir(parents=True, exist_ok=True)

    # Initialize basic gitconfig if not exists
    gitconfig = profile_home / ".gitconfig"
    if not gitconfig.exists():
        gitconfig.write_text(
            "[user]\n"
            "    name = tyagent\n"
            "    email = agent@tyagent.local\n",
            encoding="utf-8",
        )

    # Initialize .ssh directory
    ssh_dir = profile_home / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    ssh_dir.chmod(0o700)

    # Determine real user home (using passwd entry, not $HOME which may be profile home)
    real_home = os.path.expanduser("~")
    try:
        import pwd

        real_home = pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError):
        pass

    # Set HOME to profile home for consistent isolation
    os.environ["HOME"] = str(profile_home)
    os.environ["TY_AGENT_REAL_HOME"] = str(real_home)
    logger.info("Profile home initialized at: %s", profile_home)
    logger.info("Real home available via TY_AGENT_REAL_HOME: %s", real_home)

    # Change to workspace directory
    workspace = config.workspace_dir
    try:
        os.chdir(workspace)
        logger.info("Working directory set to: %s", workspace)
    except OSError as exc:
        logger.error(
            "Failed to change to workspace directory %s: %s", workspace, exc
        )
        raise

    gateway = Gateway(config)
    await gateway.start()
