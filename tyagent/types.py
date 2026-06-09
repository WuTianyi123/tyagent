"""Shared type definitions for tyagent actor model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ReplyTarget:
    """Route information for agent responses."""
    platform: str = ""
    chat_id: str = ""
    message_id: str = ""


@dataclass
class AgentOutput:
    """Single output from the agent loop."""
    kind: str = "text"  # "text" | "progress"
    text: str = ""
    reply_target: Optional[ReplyTarget] = None
    finish: bool = False  # for progress: True means finalize this batch


@dataclass
class InboxMessage:
    """A user message going into the agent loop."""
    text: str
    reply_target: Optional[ReplyTarget] = None
    tool_progress_cb: Optional[Any] = None
    segment_break_cb: Optional[Any] = None
    turn_done_cb: Optional[Any] = None
