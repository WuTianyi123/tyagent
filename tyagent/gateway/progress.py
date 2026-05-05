"""Tool progress display utilities for tyagent gateway.

Provides:
- Tool preview generation (one-line summary of primary argument)
- Emoji lookup per tool
- Async progress message sender (queue-based, single message editing)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from tyagent.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool emoji registry
# ---------------------------------------------------------------------------
_TOOL_EMOJIS: Dict[str, str] = {
    "terminal": "💻",
    "read_file": "📖",
    "write_file": "✏️",
    "patch": "🔧",
    "search_files": "🔍",
    "web_search": "🌐",
    "web_extract": "📄",
    "browser_navigate": "🌍",
    "browser_click": "👆",
    "browser_type": "⌨️",
    "browser_snapshot": "📸",
    "browser_vision": "👁️",
    "browser_scroll": "📜",
    "browser_console": "🖥️",
    "browser_back": "🔙",
    "image_generate": "🎨",
    "text_to_speech": "🔊",
    "vision_analyze": "👁️",
    "execute_code": "⚙️",
    "spawn_task": "📤",
    "wait_task": "⏳",
    "close_task": "🛑",
    "list_tasks": "📋",
    "send_input": "📨",
    "clarify": "❓",
    "memory": "🧠",
    "send_message": "📨",
    "skill_view": "📋",
    "skill_manage": "⚒️",
    "skills_list": "📋",
    "session_search": "🔎",
    "todo": "📝",
    "cronjob": "⏰",
}


def get_tool_emoji(tool_name: str, default: str = "⚡") -> str:
    """Return the display emoji for a tool name."""
    return _TOOL_EMOJIS.get(tool_name, default)


# ---------------------------------------------------------------------------
# Tool preview (one-line summary of primary argument)
# ---------------------------------------------------------------------------
def _oneline(text: str) -> str:
    """Collapse whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def build_tool_preview(tool_name: str, args: dict, max_len: int = 40) -> str | None:
    """Build a short preview of a tool call's primary argument.

    *max_len* controls truncation.  Returns None if no preview can be built.
    """
    if not args:
        return None

    # Primary argument key per tool
    primary_args: Dict[str, str] = {
        "terminal": "command",
        "web_search": "query",
        "web_extract": "urls",
        "read_file": "path",
        "write_file": "path",
        "patch": "path",
        "search_files": "pattern",
        "browser_navigate": "url",
        "browser_click": "ref",
        "browser_type": "text",
        "image_generate": "prompt",
        "text_to_speech": "text",
        "vision_analyze": "question",
        "skill_view": "name",
        "skills_list": "category",
        "cronjob": "action",
        "execute_code": "code",
        "spawn_task": "goal",
        "wait_task": "task_ids",
        "close_task": "task_id",
        "list_tasks": "name",
        "send_input": "message",
        "clarify": "question",
        "skill_manage": "name",
        "todo": "todos",
        "session_search": "query",
        "memory": "action",
        "send_message": "message",
        "process": "action",
        "browser_scroll": "direction",
        "browser_vision": "question",
        "browser_console": "clear",
    }

    key = primary_args.get(tool_name)
    if not key:
        for fallback_key in ("query", "text", "command", "path", "name", "prompt", "code", "goal"):
            if fallback_key in args:
                key = fallback_key
                break

    if not key or key not in args:
        return None

    value = args[key]
    if isinstance(value, list):
        value = value[0] if value else ""

    preview = _oneline(str(value))
    if not preview:
        return None
    if max_len > 0 and len(preview) > max_len:
        preview = preview[: max_len - 3] + "..."
    return preview


# ---------------------------------------------------------------------------
# Progress sender: queue-based async tool progress message editing
# ---------------------------------------------------------------------------
class ProgressSender:
    """Async task that accumulates tool progress lines into a single
    platform message (created once, edited progressively).

    One instance per message turn.  Usage::

        sender = ProgressSender(adapter, chat_id)
        task = asyncio.create_task(sender.run())
        # ... agent runs, fires progress_callback ...
        sender.finish()
        await task
    """

    # Minimum seconds between edits (throttle)
    _EDIT_INTERVAL: float = 1.5

    def __init__(
        self,
        adapter: BasePlatformAdapter,
        chat_id: str,
        *,
        reply_to_message_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.adapter = adapter
        self.chat_id = chat_id
        self.reply_to_message_id = reply_to_message_id
        self.enabled = enabled
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._done = False
        self._progress_msg_type: str | None = None

    def on_tool_started(self, tool_name: str, args: dict | None = None, prefix: str = "") -> None:
        """Called by the agent when a tool starts executing.

        Fires from the event loop thread (before ``await run_in_executor``),
        so ``put_nowait`` is safe without thread-safety wrappers.

        *prefix* is an optional string prepended to the progress line
        (used by subagent progress relay, e.g. ``"📤 "``).
        """
        if not self.enabled:
            return
        emoji = get_tool_emoji(tool_name)
        preview = build_tool_preview(tool_name, args or {})
        if preview:
            msg = f"{prefix}{emoji} {tool_name}: \"{preview}\""
        else:
            msg = f"{prefix}{emoji} {tool_name}..."

        # Queue from the event loop thread (agent.chat() fires this callback
        # on the event loop before awaiting run_in_executor).
        self._queue.put_nowait(msg)

    def finish(self) -> None:
        """Signal that no more progress updates will come."""
        self._done = True

    async def run(self) -> None:
        """Async task: drain queue and edit a single progress message."""
        if not self.enabled:
            drained = 0
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            if drained:
                logger.debug("Progress disabled — discarded %d queued items", drained)
            return

        progress_lines: list[str] = []
        progress_msg_id: str | None = None
        can_edit = True
        last_edit_ts = 0.0

        try:
            while True:
                # Drain all available items
                items: list[str] = []
                while not self._queue.empty():
                    try:
                        items.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                for item in items:
                    progress_lines.append(str(item))

                # Skip edit if no new items arrived and we already have a message.
                if not items and progress_msg_id is not None:
                    if self._done and self._queue.empty():
                        return
                    await asyncio.sleep(0.05)
                    continue

                if not progress_lines:
                    if self._done and self._queue.empty():
                        return
                    await asyncio.sleep(0.05)
                    continue

                # Throttle edits
                now = time.monotonic()
                elapsed = now - last_edit_ts
                if elapsed < self._EDIT_INTERVAL and not self._done:
                    await asyncio.sleep(self._EDIT_INTERVAL - elapsed)
                    continue

                text = "\n".join(progress_lines)

                if progress_msg_id is not None and can_edit:
                    result = await self.adapter.edit_message(
                        self.chat_id, progress_msg_id, text,
                        msg_type=self._progress_msg_type,
                    )
                    if result.success:
                        last_edit_ts = time.monotonic()
                    else:
                        can_edit = False
                        logger.warning(
                            "ProgressSender: edit failed for %s (adapter=%s) — "
                            "falling back to new message",
                            progress_msg_id, type(self.adapter).__name__,
                        )
                        fallback = await self.adapter.send_message(
                            self.chat_id, text,
                            reply_to_message_id=self.reply_to_message_id,
                        )
                        if fallback.success and fallback.message_id:
                            progress_msg_id = fallback.message_id
                            can_edit = True
                            self._progress_msg_type = getattr(fallback, "msg_type", None)
                            last_edit_ts = time.monotonic()
                elif progress_msg_id is None:
                    result = await self.adapter.send_message(
                        self.chat_id, text,
                        reply_to_message_id=self.reply_to_message_id,
                    )
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id
                        self._progress_msg_type = getattr(result, "msg_type", None)
                        last_edit_ts = time.monotonic()
                    elif result.success and not result.message_id:
                        logger.warning(
                            "ProgressSender: send_message succeeded but no "
                            "message_id returned (adapter=%s)",
                            type(self.adapter).__name__,
                        )
                        can_edit = False
                        progress_msg_id = ""
                else:
                    # Stale progress_msg_id with can_edit=False — retry send
                    retry = await self.adapter.send_message(
                        self.chat_id, text,
                        reply_to_message_id=self.reply_to_message_id,
                    )
                    if retry.success and retry.message_id:
                        progress_msg_id = retry.message_id
                        can_edit = True
                        self._progress_msg_type = getattr(retry, "msg_type", None)
                        last_edit_ts = time.monotonic()

                if self._done:
                    return

                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # Task cancelled during shutdown — exit cleanly
            pass
        except Exception:
            logger.exception("ProgressSender.run() crashed")
