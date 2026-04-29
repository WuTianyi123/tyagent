"""Gateway runner for tyagent.

Manages platform adapters, routes messages to the AI agent,
and handles session lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tyagent.agent import AgentError, TyAgent
from tyagent.config import PlatformConfig, TyAgentConfig
from tyagent.gateway.consumer import StreamConsumer
from tyagent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from tyagent.session import SessionStore
from tyagent.tools import memory_tool
from tyagent.tools import search_tool
from tyagent.tools.registry import registry

logger = logging.getLogger(__name__)


def _sanitize_message_chain(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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
            if msg.get("role") == "assistant" and not msg.get("content") and not msg.get("tool_calls"):
                logger.info(
                    "Sanitized empty assistant message [%d] (no content, no tool_calls) — removing", i
                )
                result = result[:i] + result[i + 1:]
                changed = True
                break  # re-scan from end after mutation

            # Fix 2: orphaned tool_calls → insert synthetic tool responses
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Count how many tool responses follow this assistant message.
                # If fewer than tool_calls, insert synthetic responses for the missing ones.
                tool_calls = msg["tool_calls"]
                n_expected = len(tool_calls)
                n_actual = 0
                j = i + 1
                while j < len(result) and result[j].get("role") == "tool":
                    n_actual += 1
                    j += 1
                if n_actual < n_expected:
                    logger.info(
                        "Sanitized orphaned tool_calls from assistant message [%d] "
                        "(%d tool_calls, only %d tool responses) — inserting %d synthetic",
                        i, n_expected, n_actual, n_expected - n_actual,
                    )
                    synthetic = []
                    for tc in tool_calls[n_actual:]:  # only the missing ones
                        tc_id = tc.get("id", "unknown") if isinstance(tc, dict) else getattr(tc, "id", "unknown")
                        synthetic.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": "(tool call interrupted — gateway restarted or crashed)",
                        })
                    result = result[:i + 1 + n_actual] + synthetic + result[i + 1 + n_actual:]
                    changed = True
                    break  # re-scan after mutation

        if not changed:
            break

    return result

# Registry of platform adapter classes
_PLATFORM_REGISTRY: Dict[str, Type[BasePlatformAdapter]] = {}


def register_platform(name: str, adapter_class: Type[BasePlatformAdapter]) -> None:
    """Register a platform adapter class.

    Example::

        from tyagent.gateway import register_platform
        from my_platform import MyAdapter
        register_platform("my_platform", MyAdapter)
    """
    _PLATFORM_REGISTRY[name] = adapter_class
    logger.debug("Registered platform adapter: %s", name)


def _load_builtin_platforms() -> None:
    """Load built-in platform adapters."""
    try:
        from tyagent.platforms.feishu import FeishuAdapter

        register_platform("feishu", FeishuAdapter)
    except ImportError as exc:
        logger.debug("Feishu adapter not available: %s", exc)


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
        self.session_store = session_store or SessionStore(sessions_dir=config.sessions_dir)
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
        self._session_to_adapter: Dict[str, str] = {}
        self._active_chat_ids: Dict[str, str] = {}

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

    @staticmethod
    def _sync_messages_to_session(
        session: Any,
        sanitized: List[Dict[str, Any]],
        original_count: int,
    ) -> None:
        """Sync messages from sanitized list back to session store.

        agent.chat() mutates the sanitized list in-place (appends assistant/tool
        messages). If sanitized is a different list object than session.messages
        (e.g. after _sanitize_message_chain creates a copy), the original session
        would miss these new messages. This method copies only the new messages
        (those beyond original_count) into the session's message list.

        Also syncs via persist_message callback (see _on_message), so this is
        a safety net for any messages the callback might have missed, plus it
        ensures session.messages is kept up-to-date for the next user turn.
        """
        if len(sanitized) <= original_count:
            return
        new_messages = sanitized[original_count:]
        for msg in new_messages:
            kwargs = {}
            for k in ("tool_calls", "tool_call_id", "reasoning"):
                if k in msg:
                    kwargs[k] = msg[k]
            # DeepSeek uses "reasoning_content" instead of "reasoning"
            if "reasoning_content" in msg:
                kwargs["reasoning"] = msg["reasoning_content"]
            session.add_message(
                msg.get("role", "assistant"),
                msg.get("content"),
                **kwargs,
            )

    async def _on_message(self, event: MessageEvent) -> Optional[str]:
        """Handle an incoming message event."""
        adapter = self._find_adapter_for_event(event)
        if not adapter:
            logger.warning("No adapter found for event from platform: %s", event.platform)
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
            logger.info("Session %s was suspended — archiving and creating fresh session", session_key)
            self.session_store.archive(session_key)
            self.session_store.clear_resume_pending(session_key)
            # Get a fresh session with reset metadata (including clearing "suspended")
            session = self.session_store.get_or_create_after_archive(session_key)
            await adapter.send_message(
                event.chat_id or "",
                "🔄 检测到异常中断，已为您恢复新会话。之前的对话已归档保留。",
                reply_to_message_id=event.message_id,
            )
        else:
            session = self.session_store.get(session_key)

        # Handle reset commands (normalize triggers to strip leading /)
        normalized_triggers = {t.lstrip("/").lower() for t in self.config.reset_triggers}
        if event.is_command() and event.get_command() in normalized_triggers:
            self.session_store.archive(session_key)
            self.session_store.freshen_session(session_key)
            session = self.session_store.get(session_key)
            await adapter.send_message(
                event.chat_id or "",
                "✅ 已归档旧会话，开始新的对话。历史记录已保留。",
                reply_to_message_id=event.message_id,
            )
            return "Session archived"

        # Handle /status command
        if event.is_command() and event.get_command() == "status":
            status_text = self._format_status(session_key)
            await adapter.send_message(
                event.chat_id or "",
                status_text,
                reply_to_message_id=event.message_id,
            )
            return "Status sent"

        # Handle /restart command
        if event.is_command() and event.get_command() == "restart":
            await adapter.send_message(
                event.chat_id or "",
                "🔄 正在重启 gateway...",
                reply_to_message_id=event.message_id,
            )
            os.kill(os.getpid(), signal.SIGUSR1)
            return "Gateway restart initiated"

        # Build message for LLM
        user_message = event.text
        if event.media_urls:
            media_desc = "\n".join(
                f"[Attached {mt or 'file'}: {url}]"
                for mt, url in zip(event.media_types or [], event.media_urls)
            )
            user_message = f"{user_message}\n\n{media_desc}" if user_message else media_desc

        # Persist user message to DB
        session.add_message("user", user_message)

        # Build tool definitions
        tool_defs = registry.get_definitions()

        # Track this session as actively processing
        self._active_sessions.add(session_key)
        self._session_to_adapter[session_key] = adapter.platform_name
        self._active_chat_ids[session_key] = event.chat_id or ""
        try:
            # Sanitize message chain (strip orphaned tool_calls at end)
            sanitized = _sanitize_message_chain(session.messages)

            # Inject persistent memory into the last user message content
            # instead of adding a separate system message — this keeps the
            # system prompt prefix stable for prompt caching.
            memory_block = self.memory_store.get_all_formatted()
            if memory_block and sanitized and sanitized[-1].get("role") == "user":
                # Make a shallow copy to avoid mutating the shared dict from session.messages
                sanitized[-1] = dict(sanitized[-1])
                existing = sanitized[-1].get("content") or ""
                sanitized[-1]["content"] = existing + "\n\n[记忆上下文]\n" + memory_block

            # Define the persistence callback for tool loop messages
            def persist_message(role: str, content: str, **extras) -> None:
                self.session_store.add_message(
                    session_key, role, content,
                    session_id=session.metadata.get("current_session_id", ""),
                    **extras
                )

            agent = self._get_or_create_agent(session_key)

            # Streaming path for platform chat messages (non-command)
            if event.chat_id and not event.is_command():
                consumer = StreamConsumer(adapter, event.chat_id, reply_to_message_id=event.message_id)
                consumer_task = asyncio.create_task(consumer.run())
                try:
                    response = await agent.chat(
                        sanitized,
                        tools=tool_defs,
                        stream=True,
                        stream_delta_callback=consumer.on_delta,
                        on_segment_break=consumer.on_segment_break,
                        on_message=persist_message,
                    )
                    streaming_ok = True
                except Exception:
                    streaming_ok = False
                    logger.exception("Agent streaming chat failed, propagating to outer handler")
                    raise
                finally:
                    consumer.finish()
                    try:
                        await consumer_task
                    except asyncio.CancelledError:
                        pass  # Normal cancellation, consumer already handled it
                    except Exception:
                        if streaming_ok:
                            raise  # Consumer failed after agent succeeded — must report
                        logger.exception("StreamConsumer task failed during cleanup (agent also failed)")

                # Response is already sent via StreamConsumer editing
                return response

            # Non-streaming path for commands and fallback
            response = await agent.chat(
                sanitized,
                tools=tool_defs,
                on_message=persist_message,
            )
        except AgentError as exc:
            logger.error("Agent error: %s", exc)
            response = f"❌ 错误: {exc}"
        except Exception:
            logger.exception("Unexpected agent error")
            response = "Sorry, something went wrong."
        finally:
            self._active_sessions.discard(session_key)
            self._active_chat_ids.pop(session_key, None)
            # Clear resume_pending flag after first resumed turn
            try:
                if self.session_store.is_resume_pending(session_key):
                    self.session_store.clear_resume_pending(session_key)
            except Exception as exc:
                logger.warning("Failed to clear resume_pending for %s: %s", session_key, exc)

        # Send response (non-streaming path only)
        result = await adapter.send_message(
            event.chat_id or "",
            response,
            reply_to_message_id=event.message_id,
        )
        if not result.success:
            logger.error("Failed to send response: %s", result.error)

        return response

    def _find_adapter_for_event(self, event: MessageEvent) -> Optional[BasePlatformAdapter]:
        """Find the adapter that should handle this event by platform name."""
        return self.adapters.get(event.platform)

    def _format_status(self, session_key: str) -> str:
        """Format a status message for the /status command."""
        from datetime import datetime

        session = self.session_store.get(session_key)
        connected = list(self.adapters.keys())

        created = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d %H:%M")
        updated = datetime.fromtimestamp(session.updated_at).strftime("%Y-%m-%d %H:%M")

        lines = [
            "📊 **tyagent Status**",
            "",
            f"**Session:** `{session_key}`",
            f"**Messages:** {self.session_store.get_message_count(session_key, session_id=session.metadata.get('current_session_id', ''))}",
            f"**Created:** {created}",
            f"**Last Activity:** {updated}",
            "",
            f"**Model:** `{self._get_or_create_agent(session_key).model}`",
            f"**Connected Platforms:** {', '.join(connected) if connected else 'None'}",
        ]
        return "\n".join(lines)

    async def start(self) -> None:
        """Start all adapters and run the gateway."""
        self._load_adapters()
        self._setup_signal_handlers()
        self._check_recovery_on_startup()
        if not self.adapters:
            logger.error("No adapters loaded. Check your configuration.")
            return

        self._running = True
        logger.info("Starting tyagent gateway with %d adapter(s)", len(self.adapters))

        # Start all adapters
        tasks = []
        for name, adapter in self.adapters.items():
            task = asyncio.create_task(
                self._run_adapter_with_retry(name, adapter),
                name=f"adapter-{name}",
            )
            tasks.append(task)

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

        # Close all cached agents
        for key, cached_agent in list(self._agent_cache.items()):
            try:
                await cached_agent.close()
            except Exception:
                pass
        self._agent_cache.clear()
        self.session_store.close()
        logger.info("Gateway stopped")

    async def _run_adapter_with_retry(self, name: str, adapter: BasePlatformAdapter) -> None:
        """Run a single adapter with exponential backoff retry."""
        max_retries = 10
        retry_delay = 5.0
        max_retry_delay = 60.0

        for attempt in range(1, max_retries + 1):
            if not self._running:
                break
            try:
                logger.info("Starting adapter %s (attempt %d/%d)", name, attempt, max_retries)
                await adapter.start()
                # If start() returns normally, the adapter has stopped
                logger.info("Adapter %s stopped gracefully", name)
                break
            except asyncio.CancelledError:
                logger.info("Adapter %s task cancelled", name)
                break
            except Exception as exc:
                logger.error("Adapter %s crashed: %s", name, exc)
                if not self._running or attempt >= max_retries:
                    logger.error("Adapter %s exceeded max retries, giving up", name)
                    break
                wait = min(retry_delay * (2 ** (attempt - 1)), max_retry_delay)
                logger.info("Retrying adapter %s in %.1f seconds...", name, wait)
                await asyncio.sleep(wait)

    def stop(self) -> None:
        """Signal the gateway to shut down."""
        self._running = False
        self._shutdown_event.set()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM/SIGUSR1."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
        try:
            loop.add_signal_handler(signal.SIGUSR1, self._on_sigusr1)
        except NotImplementedError:
            pass

    def _on_sigusr1(self) -> None:
        """Handle SIGUSR1 — start graceful restart."""
        if self._restart_requested:
            logger.warning("SIGUSR1 already received, ignoring duplicate")
            return
        logger.info("SIGUSR1 received — initiating graceful restart")
        self._restart_requested = True
        self._draining = True
        loop = asyncio.get_running_loop()
        loop.create_task(self._do_graceful_restart())

    async def _do_graceful_restart(self) -> None:
        """Perform graceful restart: notify, drain, mark sessions, exit."""
        try:
            logger.info("Graceful restart: notifying active sessions...")
            await self._notify_active_sessions_of_restart()

            logger.info(
                "Graceful restart: draining up to %.0f seconds (%d active sessions)",
                self._restart_drain_timeout,
                len(self._active_sessions),
            )
            drained = await self._drain_active_agents(self._restart_drain_timeout)

            if not drained:
                logger.warning(
                    "Drain timeout reached — forcing restart with %d active session(s)",
                    len(self._active_sessions),
                )

            # Mark pending sessions for resume, so the new process can pick them up
            for session_key in list(self._active_sessions):
                try:
                    self.session_store.mark_resume_pending(session_key, reason="restart_timeout")
                except Exception:
                    logger.exception("Failed to mark resume_pending for %s", session_key)

            # Write .clean_shutdown marker to indicate intentional restart
            try:
                marker_path = Path.home() / ".tyagent" / ".clean_shutdown"
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text("clean", encoding="utf-8")
                logger.info("Wrote .clean_shutdown marker at %s", marker_path)
            except Exception as exc:
                logger.error("Failed to write .clean_shutdown marker: %s", exc)
        except Exception:
            logger.exception("Graceful restart failed unexpectedly — exiting with code 75 anyway")

        logger.info("Graceful restart complete — exiting with code 75")
        os._exit(75)

    async def _drain_active_agents(self, timeout: float) -> bool:
        """Wait for all active sessions to complete, up to timeout seconds.

        Returns True if all sessions drained, False on timeout.
        """
        if not self._active_sessions:
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._active_sessions:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(0.5)
        return True

    async def _notify_active_sessions_of_restart(self) -> None:
        """Send restart notification to all active sessions."""
        message = "⚠️ Gateway is restarting for an update. Active requests will complete before restart."
        for session_key in list(self._active_sessions):
            try:
                adapter_name = self._session_to_adapter.get(session_key)
                chat_id = self._active_chat_ids.get(session_key, "")
                adapter = self.adapters.get(adapter_name) if adapter_name else None
                if adapter is not None and chat_id:
                    await adapter.send_message(chat_id, message)
            except Exception:
                logger.exception("Failed to notify session %s", session_key)

    def _check_recovery_on_startup(self) -> None:
        """Check for .clean_shutdown marker and handle session recovery.

        If the marker exists, it means the previous shutdown was intentional
        (graceful restart), so we skip suspending sessions. If absent, we
        assume an unclean shutdown and suspend recently-active sessions.
        """
        marker_path = Path.home() / ".tyagent" / ".clean_shutdown"
        if marker_path.exists():
            logger.info("Clean shutdown marker found — previous restart was intentional")
            try:
                marker_path.unlink()
                logger.debug("Removed .clean_shutdown marker")
            except OSError as exc:
                logger.warning("Failed to remove .clean_shutdown marker: %s", exc)
            return

        logger.info("No clean shutdown marker — assuming unclean shutdown, suspending recent sessions")
        try:
            suspended = self.session_store.suspend_recently_active(max_age_seconds=120)
            if suspended:
                logger.warning("Suspended %d recently-active session(s) due to unclean shutdown", suspended)
            else:
                logger.debug("No recently-active sessions to suspend")
        except Exception:
            logger.exception("Failed to suspend recent sessions on startup")


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
            encoding="utf-8"
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

    # Set HOME to profile home for consistent isolation (gateway + subprocesses)
    os.environ["HOME"] = str(profile_home)
    # Preserve real home for tools that need to access user's actual files
    os.environ["TY_AGENT_REAL_HOME"] = str(real_home)
    logger.info("Profile home initialized at: %s", profile_home)
    logger.info("Real home available via TY_AGENT_REAL_HOME: %s", real_home)

    # Change to workspace directory
    workspace = config.workspace_dir
    try:
        os.chdir(workspace)
        logger.info("Working directory set to: %s", workspace)
    except OSError as exc:
        logger.error("Failed to change to workspace directory %s: %s", workspace, exc)
        raise

    gateway = Gateway(config)
    # _setup_signal_handlers is called inside gateway.start()
    await gateway.start()
