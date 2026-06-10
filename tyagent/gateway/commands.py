"""Command registry for tyagent gateway.

Handles built-in commands (/help, /status, /restart, /new) with a
pluggable registry pattern instead of if/elif chains.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime
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
        self._commands["compact"] = (
            "压缩当前会话上下文（生成摘要并持久化）",
            self._cmd_compact,
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
        self._gateway.supervisor._on_sigusr1()
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
        # Stop and remove cached agent so the next message creates
        # a fresh agent with the new session_id (empty history).
        await gw._stop_session_agent(session_key)
        await adapter.send_message(
            event.chat_id or "",
            "✅ 已归档旧会话，开始新的对话。历史记录已保留。",
            reply_to_message_id=event.message_id,
        )
        return "Session archived"

    async def _cmd_compact(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
        _session: Any,
    ) -> str:
        """Handle /compact: manually trigger context compaction."""
        from tyagent.compaction import run_compact

        gw = self._gateway
        ctx = gw._sessions.get(session_key)
        if ctx is None or ctx.agent is None:
            await adapter.send_message(
                event.chat_id or "",
                "❌ 当前没有活跃的 agent 会话。",
                reply_to_message_id=event.message_id,
            )
            return "No active session"

        agent = ctx.agent
        messages = agent._messages
        if not messages:
            await adapter.send_message(
                event.chat_id or "",
                "ℹ️ 当前没有消息可压缩。",
                reply_to_message_id=event.message_id,
            )
            return "No messages"

        await adapter.send_message(
            event.chat_id or "",
            "🔄 正在压缩会话上下文...",
            reply_to_message_id=event.message_id,
        )

        try:
            compacted = await run_compact(
                messages,
                model=agent.compact_model,
                api_key=agent.compact_api_key,
                base_url=agent.compact_base_url,
                http_client=agent._client,
            )
        except Exception as exc:
            logger.exception("Manual compaction failed")
            await adapter.send_message(
                event.chat_id or "",
                f"❌ 压缩失败: {exc}",
                reply_to_message_id=event.message_id,
            )
            return "Compaction failed"

        if compacted is None:
            await adapter.send_message(
                event.chat_id or "",
                "⚠️ 压缩未能完成（可能没有足够的用户消息）。",
                reply_to_message_id=event.message_id,
            )
            return "Compaction returned None"

        # Persist via the same on_compacted path as automatic compaction
        if agent._on_compacted is not None:
            try:
                await agent._on_compacted(compacted)
            except Exception:
                logger.exception("on_compacted callback failed during manual compaction")

        # Replace in-memory messages
        agent._messages[:] = compacted
        agent._refresh_memory_and_prompt()
        agent.last_usage = None

        # Count the compaction result
        old_count = len(messages)
        new_count = len(compacted)
        from tyagent.compaction import is_summary_message
        has_summary = any(is_summary_message(m.get("content", "")) for m in compacted)

        await adapter.send_message(
            event.chat_id or "",
            f"✅ 压缩完成：{old_count} 条消息 → {new_count} 条"
            + ("（含摘要）" if has_summary else "")
            + "。重启后亦不丢失。",
            reply_to_message_id=event.message_id,
        )
        return "Compaction complete"


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _format_status(gateway: Gateway, session_key: str) -> str:
    """Format a status message for the /status command.

    Lives here (not inside CommandRegistry) because it is a
    pure formatting function that does not need the command registry.
    """
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
