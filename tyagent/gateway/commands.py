"""Command registry for tyagent gateway.

Handles built-in commands (/help, /status, /restart, /new) with a
pluggable registry pattern instead of if/elif chains.
"""

from __future__ import annotations

import logging
import os
import signal
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict

if TYPE_CHECKING:
    from tyagent.gateway.gateway import Gateway
    from tyagent.platforms.base import BasePlatformAdapter, MessageEvent

logger = logging.getLogger(__name__)


class CommandRegistry:
    """Pluggable command handler registry.

    Each command maps a name (e.g. ``"help"``) to a tuple of
    ``(description, async_handler)``.  The handler signature is::

        async def handler(
            adapter: BasePlatformAdapter,
            event: MessageEvent,
            session_key: str,
            session: Any,
        ) -> str:
    """

    def __init__(self, gateway: Gateway) -> None:
        self._gateway = gateway
        self._commands: Dict[
            str, tuple[str, Callable[..., Awaitable[str]]]
        ] = OrderedDict()
        self._init_commands()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def commands(self) -> Dict[str, tuple[str, Callable[..., Awaitable[str]]]]:
        return self._commands

    async def dispatch(
        self,
        cmd_name: str,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
        session: Any,
    ) -> str | None:
        """Dispatch a command by name.  Returns None if unknown."""
        if cmd_name not in self._commands:
            return None
        _, handler = self._commands[cmd_name]
        return await handler(adapter, event, session_key, session)

    # ------------------------------------------------------------------
    # Command initialisation
    # ------------------------------------------------------------------

    def _init_commands(self) -> None:
        """Register all built-in and configurable commands."""
        gw = self._gateway
        self._commands["help"] = ("显示此帮助信息", self._cmd_help)
        self._commands["status"] = (
            "查看当前会话状态（模型、消息数、会话时长）",
            self._cmd_status,
        )
        self._commands["restart"] = (
            "重启 gateway（约 2~3 秒，重启完成后自动通知）",
            self._cmd_restart,
        )
        for trigger in gw.config.reset_triggers:
            trigger = trigger.strip().lower().lstrip("/")
            if trigger not in self._commands:
                self._commands[trigger] = (
                    "归档当前会话并开始新对话（历史记录保留）",
                    self._cmd_reset,
                )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_help(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
        _session: Any,
    ) -> str:
        """Handle /help: auto-generate help from command registry."""
        lines = ["📖 **tyagent 可用命令**", ""]
        for name, (desc, _) in self._commands.items():
            lines.append(f"**`/{name}`** — {desc}")
        await adapter.send_message(
            event.chat_id or "",
            "\n".join(lines),
            reply_to_message_id=event.message_id,
        )
        return "Help sent"

    async def _cmd_status(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
        _session: Any,
    ) -> str:
        """Handle /status: show session info."""
        gw = self._gateway
        status_text = _format_status(gw, session_key)
        await adapter.send_message(
            event.chat_id or "",
            status_text,
            reply_to_message_id=event.message_id,
        )
        return "Status sent"

    async def _cmd_restart(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        _session_key: str,
        _session: Any,
    ) -> str:
        """Handle /restart: trigger graceful gateway restart."""
        await adapter.send_message(
            event.chat_id or "",
            "🔄 正在重启 gateway...",
            reply_to_message_id=event.message_id,
        )
        self._gateway.set_restart_requestor(event.platform, event.chat_id or "")
        os.kill(os.getpid(), signal.SIGUSR1)
        return "Gateway restart initiated"

    async def _cmd_reset(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
        _session: Any,
    ) -> str:
        """Handle /new and /reset: archive current session, start fresh."""
        gw = self._gateway
        gw.session_store.archive(session_key)
        gw.session_store.freshen_session(session_key)
        await adapter.send_message(
            event.chat_id or "",
            "✅ 已归档旧会话，开始新的对话。历史记录已保留。",
            reply_to_message_id=event.message_id,
        )
        return "Session archived"


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _format_status(gateway: Gateway, session_key: str) -> str:
    """Format a status message for the /status command.

    Lives here (not inside CommandRegistry) because it is a
    pure formatting function that does not need the command registry.
    """
    from datetime import datetime

    session = gateway.session_store.get(session_key)
    connected = list(gateway.adapters.keys())

    created = datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d %H:%M")
    updated = datetime.fromtimestamp(session.updated_at).strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [
        "📊 **tyagent Status**",
        "",
        f"**Session:** `{session_key}`",
        f"**Messages:** {gateway.session_store.get_message_count(session_key, session_id=session.metadata.get('current_session_id', ''))}",
        f"**Created:** {created}",
        f"**Last Activity:** {updated}",
        "",
        f"**Model:** `{gateway._get_or_create_agent(session_key).model}`",
        f"**Connected Platforms:** {', '.join(connected) if connected else 'None'}",
    ]
    return "\n".join(lines)
