"""Per-agent mailbox for inter-agent communication (Codex v2 model).

每个 agent thread（主 agent 和每个子 agent）各有一个 Mailbox。
其他 agent 通过 ``send()`` 投递消息；owner 通过 ``drain()`` 取走。

两种消息类型：

- **InterAgentMessage**：中途进度报告或协作消息（不表示 agent 完成）
- **FinalNotification**：子 agent 完成时的最终通知（含 summary）

``_agent_loop`` drains the mailbox on each iteration and injects
pending messages into ``_messages`` before running the next turn.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Message types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class InterAgentMessage:
    """A mid-task message from one agent to another.

    Does NOT mean the sender has finished — use :class:`FinalNotification`
    for completion signals.
    """

    author: str  # canonical task_path of sender
    recipient: str  # canonical task_path of receiver
    content: str  # message body (plain text)
    trigger_turn: bool = False  # inbox→LLM 自动触发新 turn
    timestamp: float = field(default_factory=time.time)


@dataclass
class FinalNotification:
    """Final result from a completed (or failed/timed-out) child agent."""

    task_path: str
    success: bool
    summary: Optional[str]
    error: Optional[str]
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)


# Union type for drained mailbox items
MailboxItem = InterAgentMessage | FinalNotification


# ═══════════════════════════════════════════════════════════════════
# Mailbox
# ═══════════════════════════════════════════════════════════════════


class Mailbox:
    """Per-agent mailbox — lightweight async-safe message channel.

    Uses an :class:`asyncio.Event` for efficient ``wait_next()`` polling
    in the agent's event loop.

    Thread-safe for send() (called from sibling agent tasks), but
    drain()/peek()/wait_next() should only be called from the owning
    agent's event loop.
    """

    def __init__(self, owner_path: str = "/root"):
        self.owner_path = owner_path
        self._queue: List[MailboxItem] = []
        self._event = asyncio.Event()

    # ── send ────────────────────────────────────────────────────

    def send(self, item: MailboxItem) -> None:
        """Enqueue a message.  Fire-and-forget — thread-safe."""
        self._queue.append(item)
        self._event.set()

    # ── drain / peek ────────────────────────────────────────────

    def drain(self) -> List[MailboxItem]:
        """Dequeue ALL pending items and clear the awaitable event."""
        items = self._queue[:]
        self._queue.clear()
        self._event.clear()
        return items

    def peek(self) -> bool:
        """True when there are pending items."""
        return bool(self._queue)

    # ── wait ────────────────────────────────────────────────────

    async def wait_next(self, timeout: Optional[float] = None) -> bool:
        """Block until at least one message arrives (or *timeout*).

        Returns True when a message is available, False on timeout.
        """
        if self._queue:
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── helpers for _agent_loop injection ───────────────────────

    def drain_as_conversation_messages(
        self,
    ) -> List[Dict[str, Any]]:
        """Drain the mailbox and convert each item to an LLM conversation message.

        InterAgentMessage → ``role: user`` with ``[author]: content`` prefix.
        FinalNotification   → ``role: user`` with completion summary.

        Messages with ``trigger_turn=False`` are still returned — the caller
        (``_agent_loop``) decides whether to trigger a turn.
        """
        items = self.drain()
        messages: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, FinalNotification):
                if item.success and item.summary:
                    text = (
                        f"（子代理完成）{item.task_path} "
                        f"({item.duration_seconds:.1f}s):\n\n{item.summary}"
                    )
                elif item.success:
                    text = (
                        f"（子代理完成）{item.task_path} "
                        f"已成功结束（{item.duration_seconds:.1f}s）"
                    )
                else:
                    text = (
                        f"（子代理失败）{item.task_path} "
                        f"({item.duration_seconds:.1f}s): "
                        f"{item.error or '未知错误'}"
                    )
                messages.append({"role": "user", "content": text})
            else:  # InterAgentMessage
                prefix = f"[{item.author}]"
                text = f"{prefix}\n{item.content}"
                messages.append({"role": "user", "content": text})
        return messages

    def drain_with_trigger_info(
        self,
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Drain and return (messages, should_trigger_turn).

        ``should_trigger_turn`` is True when at least one InterAgentMessage
        has ``trigger_turn=True``, OR there is at least one FinalNotification.
        """
        items = self.drain()
        messages: List[Dict[str, Any]] = []
        should_trigger = False

        for item in items:
            if isinstance(item, FinalNotification):
                should_trigger = True
                if item.success and item.summary:
                    text = (
                        f"（子代理完成）{item.task_path} "
                        f"({item.duration_seconds:.1f}s):\n\n{item.summary}"
                    )
                elif item.success:
                    text = (
                        f"（子代理完成）{item.task_path} "
                        f"已成功结束（{item.duration_seconds:.1f}s）"
                    )
                else:
                    text = (
                        f"（子代理失败）{item.task_path} "
                        f"({item.duration_seconds:.1f}s): "
                        f"{item.error or '未知错误'}"
                    )
                messages.append({"role": "user", "content": text})
            else:  # InterAgentMessage
                prefix = f"[{item.author}]"
                text = f"{prefix}\n{item.content}"
                messages.append({"role": "user", "content": text})
                if item.trigger_turn:
                    should_trigger = True

        return messages, should_trigger
