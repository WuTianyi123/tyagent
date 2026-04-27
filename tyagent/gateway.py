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
from typing import Any, Dict, List, Optional, Type

from tyagent.agent import AgentError, TyAgent
from tyagent.config import PlatformConfig, TyAgentConfig
from tyagent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from tyagent.session import SessionStore
from tyagent.tools import memory_tool
from tyagent.tools import search_tool
from tyagent.tools.registry import registry

logger = logging.getLogger(__name__)


def _sanitize_message_chain(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Strip orphaned tool_calls from the end of the message chain.

    If the last assistant message has ``tool_calls`` but is not followed
    by a ``tool`` response message (e.g. the process was killed or
    crashed mid-tool-loop), the orphaned tool_calls are removed so the
    chain remains valid for the LLM API.

    The original message objects are NOT mutated — the function returns
    a new list with copies of affected messages.
    """
    if not messages:
        return messages

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Check if the *next* message (if any) is a tool response
            if i + 1 >= len(messages) or messages[i + 1].get("role") != "tool":
                # Orphaned tool_calls — strip them
                logger.info(
                    "Sanitized orphaned tool_calls from assistant message [%d] "
                    "(next message role=%s)",
                    i,
                    messages[i + 1].get("role", "(end of chain)") if i + 1 < len(messages) else "(end of chain)",
                )
                clean = dict(msg)
                clean.pop("tool_calls", None)
                return messages[:i] + [clean] + messages[i + 1 :]

    return messages

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
        self.agent = agent or TyAgent.from_config(config.agent)
        # Initialize persistent memory store
        memories_dir = config.home_dir / "memories"
        self.memory_store = memory_tool.MemoryStore(memories_dir)
        memory_tool.set_store(self.memory_store)
        # Wire up session search tool
        search_tool.set_search_db(self.session_store.db)
        self._running = False
        self._shutdown_event = asyncio.Event()

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

    async def _on_message(self, event: MessageEvent) -> Optional[str]:
        """Handle an incoming message event."""
        adapter = self._find_adapter_for_event(event)
        if not adapter:
            logger.warning("No adapter found for event from platform: %s", event.platform)
            return None

        session_key = adapter.build_session_key(event)
        session = self.session_store.get(session_key)

        # Handle reset commands (normalize triggers to strip leading /)
        normalized_triggers = {t.lstrip("/").lower() for t in self.config.reset_triggers}
        if event.is_command() and event.get_command() in normalized_triggers:
            self.session_store.archive(session_key)
            await adapter.send_message(
                event.chat_id or "",
                "✅ 已归档旧会话，开始新的对话。历史记录已保留。",
                reply_to=event.reply_to_text,
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
                    session_key, role, content, **extras
                )

            response = await self.agent.chat(
                sanitized,
                tools=tool_defs,
                on_message=persist_message,
            )
        except AgentError as exc:
            logger.error("Agent error: %s", exc)
            response = "Sorry, I encountered an error processing your request."
        except Exception:
            logger.exception("Unexpected agent error")
            response = "Sorry, something went wrong."

        # Send response
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
            f"**Messages:** {self.session_store.get_message_count(session_key)}",
            f"**Created:** {created}",
            f"**Last Activity:** {updated}",
            "",
            f"**Model:** `{self.agent.model}`",
            f"**Connected Platforms:** {', '.join(connected) if connected else 'None'}",
        ]
        return "\n".join(lines)

    async def start(self) -> None:
        """Start all adapters and run the gateway."""
        self._load_adapters()
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
        await self.agent.close()
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
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass


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
    gateway._setup_signal_handlers()
    await gateway.start()
