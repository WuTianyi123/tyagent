"""Gateway runner for tyagent.

Manages platform adapters, routes messages to the AI agent,
and handles session lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import signal
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Type

from tyagent.agent import AgentError, TyAgent
from tyagent.config import PlatformConfig, TyAgentConfig
from tyagent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from tyagent.session import SessionStore
from tyagent.tools import memory_tool
from tyagent.tools import search_tool
from tyagent.tools.registry import registry

logger = logging.getLogger(__name__)

# Sentinels for StreamConsumer
_DONE = object()
_NEW_SEGMENT = object()


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


class StreamConsumer:
    """Bridges sync stream_delta from agent thread to async platform message edit.

    Usage:
        consumer = StreamConsumer(adapter, chat_id)
        consumer_task = asyncio.create_task(consumer.run())
        await agent.chat(..., stream=True,
                        stream_delta_callback=consumer.on_delta,
                        on_segment_break=consumer.on_segment_break)
        consumer.finish()
        await consumer_task
        final_content = consumer.final_content
    """

    def __init__(self, adapter: BasePlatformAdapter, chat_id: str):
        self.adapter = adapter
        self.chat_id = chat_id
        self._queue: "queue.Queue" = queue.Queue()
        self._message_id: Optional[str] = None
        self._msg_type: Optional[str] = None
        self._accumulated: str = ""
        self._last_edit_time: float = 0.0
        self._edit_interval: float = 1.0
        self._buffer_threshold: int = 30
        self._last_sent_text: str = ""
        self._edit_supported: bool = True
        self._flood_strikes: int = 0
        self._MAX_FLOOD_STRIKES: int = 3
        self._current_edit_interval: float = 1.0
        self.final_content: str = ""
        self._already_sent: bool = False

    def on_delta(self, text: str) -> None:
        """Called from agent's thread (sync) for each text delta."""
        self._queue.put(text)

    def on_segment_break(self) -> None:
        """Called on tool boundary — finalize current message, prepare for new one."""
        self._queue.put(_NEW_SEGMENT)

    def finish(self) -> None:
        """Signal that the stream is complete."""
        self._queue.put(_DONE)

    async def run(self) -> str:
        """Drain queue, progressively edit platform message, return final content."""
        try:
            _raw_limit = self.adapter.MAX_MESSAGE_LENGTH
            _safe_limit = max(500, _raw_limit - 100)
        except (AttributeError, TypeError):
            _raw_limit = 4096
            _safe_limit = 3500

        try:
            while True:
                # Drain available items
                got_done = False
                got_segment_break = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is _DONE:
                            got_done = True
                            break
                        if item is _NEW_SEGMENT:
                            got_segment_break = True
                            break
                        self._accumulated += item
                    except queue.Empty:
                        break

                # Decide whether to flush
                now = time.monotonic()
                elapsed = now - self._last_edit_time
                should_edit = (
                    got_done or got_segment_break
                    or (elapsed >= self._current_edit_interval and self._accumulated)
                    or len(self._accumulated) >= self._buffer_threshold
                )

                if should_edit and self._accumulated:
                    # Don't send tiny fragments on first edit (avoid "<cursor>")
                    if not self._already_sent and len(self._accumulated) < 4 and not got_done:
                        pass  # Wait for more content
                    elif self._message_id is None:
                        # First send: create the message
                        display = self._accumulated
                        if not got_done:
                            display += " ▉"
                        result = await self.adapter.send_message(self.chat_id, display)
                        if result.success:
                            self._message_id = result.message_id
                            self._already_sent = True
                            self._last_sent_text = display
                            # Preserve msg_type from send_message for edit consistency.
                            # The adapter should communicate the type used; if not,
                            # leave None so edit_message auto-detects.
                            if hasattr(result, "msg_type") and result.msg_type:
                                self._msg_type = result.msg_type
                    else:
                        # Edit existing message
                        display = self._accumulated
                        await self._try_edit(display, add_cursor=not got_done)
                    self._last_edit_time = time.monotonic()

                # --- 截断检查：在发送前确保累积文本不超 safe_limit ---
                if len(self._accumulated) > _safe_limit:
                    logger.warning(
                        "StreamConsumer accumulated text exceeds safe limit (%d > %d), truncating from start",
                        len(self._accumulated), _safe_limit,
                    )
                    self._accumulated = self._accumulated[:_safe_limit]

                # Segment break: finalize current message
                if got_segment_break:
                    self._message_id = None
                    self._accumulated = ""

                if got_done:
                    # Final send without cursor
                    if self._accumulated:
                        if self._message_id:
                            await self._try_edit(self._accumulated)
                        elif not self._already_sent:
                            result = await self.adapter.send_message(self.chat_id, self._accumulated)
                            if result.success:
                                self._message_id = result.message_id
                                self._already_sent = True
                    self.final_content = self._accumulated
                    return self.final_content

                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            if self._accumulated and self._message_id:
                try:
                    await self._try_edit(self._accumulated)
                except Exception:
                    pass
            self.final_content = self._accumulated
            return self.final_content
        except Exception as e:
            logger.error("StreamConsumer error: %s", e)
            self.final_content = self._accumulated
            return self.final_content

    async def _try_edit(self, text: str, *, add_cursor: bool = False) -> None:
        """Try to edit platform message with flood control protection."""
        if add_cursor:
            text = text + " ▉"
        if self._message_id and text == self._last_sent_text:
            return
        self._last_sent_text = text

        if self._message_id and self._edit_supported:
            result = await self.adapter.edit_message(
                self.chat_id, self._message_id, text,
                msg_type=self._msg_type,
            )
            if result.success:
                self._flood_strikes = 0
                self._current_edit_interval = self._edit_interval
                self._already_sent = True
                return
            if self._is_flood_error(result.error):
                self._flood_strikes += 1
                self._current_edit_interval = min(self._current_edit_interval * 2, 10.0)
                self._last_edit_time = time.monotonic()
                if self._flood_strikes >= self._MAX_FLOOD_STRIKES:
                    self._edit_supported = False
                    logger.warning("Flood control: progressive edit disabled after %d strikes", self._flood_strikes)
                return
            self._edit_supported = False

        # Fallback: send new message
        result = await self.adapter.send_message(self.chat_id, text)
        if result.success:
            self._message_id = result.message_id
            self._already_sent = True

    @staticmethod
    def _is_flood_error(error: Optional[str]) -> bool:
        if not error:
            return False
        return any(code in error for code in ["99991400", "99991401", "99991402", "429 ", "rate limit"])


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

        # If a default agent was explicitly provided, return it for all sessions
        # (do NOT create per-session agents in that case)
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
            session.add_message(
                msg.get("role", "assistant"),
                msg.get("content"),
                **{k: msg[k] for k in ("tool_calls", "tool_call_id", "reasoning", "reasoning_content") if k in msg},
            )

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

            agent = self._get_or_create_agent(session_key)

            # Track original message count to sync back sanitized -> session
            original_msg_count = len(session.messages)

            # Streaming path for platform chat messages (non-command)
            if event.chat_id and not event.is_command():
                consumer = StreamConsumer(adapter, event.chat_id)
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
                except Exception:
                    logger.exception("Agent streaming chat failed, propagating to outer handler")
                    raise
                finally:
                    consumer.finish()
                    try:
                        await consumer_task
                    except Exception:
                        logger.exception("StreamConsumer task failed during cleanup")

                # Sync sanitized messages back to session
                self._sync_messages_to_session(session, sanitized, original_msg_count)
                # Response is already sent via StreamConsumer editing
                return response

            # Non-streaming path for commands and fallback
            response = await agent.chat(
                sanitized,
                tools=tool_defs,
                on_message=persist_message,
            )

            # Sync sanitized messages back to session
            self._sync_messages_to_session(session, sanitized, original_msg_count)
        except AgentError as exc:
            logger.error("Agent error: %s", exc)
            response = "Sorry, I encountered an error processing your request."
        except Exception:
            logger.exception("Unexpected agent error")
            response = "Sorry, something went wrong."

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
            f"**Messages:** {self.session_store.get_message_count(session_key)}",
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
