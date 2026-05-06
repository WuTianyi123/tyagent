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
    text: str
    reply_target: Optional[ReplyTarget] = None


@dataclass
class InboxMessage:
    """A user message going into the agent loop."""
    text: str
    reply_target: Optional[ReplyTarget] = None
    tool_progress_cb: Optional[Any] = None
    turn_done_cb: Optional[Any] = None


@dataclass
class InterAgentNotification:
    """A notification from a sub-agent injected into the parent's conversation.

    ``role: user`` because sub-agent messages come from outside the model.
    ``[author]:`` prefix so the LLM can distinguish sub-agent reports
    from user instructions.
    """
    author: str       # canonical task_path of the sender
    content: str      # notification body

    def to_message(self) -> dict:
        return {
            "role": "user",
            "content": f"[{self.author}]\n{self.content}",
        }
