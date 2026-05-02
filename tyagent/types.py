"""Shared type definitions for tyagent actor model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


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
