"""Context compression for ty-agent.

When a conversation grows too long for the LLM's context window,
this module compresses the middle portion into a structured summary
while preserving the head (system prompt + initial exchange) and
tail (recent messages).

The original session.messages is NEVER modified — compression produces
a temporary message list that is sent to the LLM API.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough token estimation: ~4 chars per token for English/mixed content.
# Conservative (overestimates) to avoid hitting actual context limits.
_CHARS_PER_TOKEN = 3.5

# Default context budget
_DEFAULT_MAX_CHARS = 80_000  # ~23k tokens, safe for most 32k+ models


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token count estimation from message list."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        # Tool call arguments also consume tokens
        for tc in msg.get("tool_calls", []) or []:
            args = tc.get("function", {}).get("arguments", "")
            total_chars += len(args)
    return int(total_chars / _CHARS_PER_TOKEN)


def should_compress(
    messages: List[Dict[str, Any]],
    max_tokens: int = 0,
    max_chars: int = 0,
) -> bool:
    """Return True if the message list exceeds the budget."""
    if max_tokens > 0:
        return estimate_tokens(messages) > max_tokens
    if max_chars > 0:
        total = sum(
            len(m.get("content", "") or "") for m in messages
        )
        return total > max_chars
    return False


def compress_messages(
    messages: List[Dict[str, Any]],
    max_chars: int = _DEFAULT_MAX_CHARS,
    head_size: int = 3,
    tail_ratio: float = 0.25,
) -> List[Dict[str, Any]]:
    """Compress a message list by replacing the middle with a summary.

    Strategy:
    - Keep head_size messages from the start (system prompt + initial exchange)
    - Keep tail_ratio of total messages from the end (recent context)
    - Replace everything in between with a single summary message

    The original messages list is NOT modified. A new list is returned.

    Args:
        messages: Original message list (will not be modified).
        max_chars: Character budget for the final message list.
        head_size: Number of messages to preserve at the start.
        tail_ratio: Fraction of messages to preserve at the end (0.0-1.0).

    Returns:
        A new compressed message list, or the original if already within budget.
    """
    n = len(messages)
    if n <= head_size + 2:
        return messages

    # Check if already within budget
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    if total_chars <= max_chars:
        return messages

    # Calculate tail size
    tail_size = max(2, int(n * tail_ratio))
    tail_size = min(tail_size, n - head_size - 1)

    head = messages[:head_size]
    middle = messages[head_size:n - tail_size]
    tail = messages[n - tail_size:]

    if not middle:
        return messages

    # Build summary of middle section
    summary_text = _summarize_middle(middle)

    summary_msg = {
        "role": "user",
        "content": (
            f"[Context Summary — {len(middle)} earlier messages compressed]\n\n"
            f"{summary_text}"
        ),
        "_compressed": True,  # marker so we can identify it
    }

    compressed = head + [summary_msg] + tail
    logger.info(
        "Context compressed: %d messages → %d (head=%d, summary=1, tail=%d), "
        "%.0fk → %.0fk chars",
        n, len(compressed), head_size, tail_size,
        total_chars / 1000,
        sum(len(m.get("content", "") or "") for m in compressed) / 1000,
    )

    return compressed


def _summarize_middle(messages: List[Dict[str, Any]]) -> str:
    """Generate a text summary of the middle messages.

    This is a RULE-BASED summarizer (no LLM call) to keep things lightweight.
    It extracts key information: user intents, tool usage, and outcomes.
    """
    parts: List[str] = []
    user_topics: List[str] = []
    tool_actions: List[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "user":
            # Capture user intents (first line, truncated)
            first_line = content.split("\n")[0].strip()
            if first_line and not first_line.startswith("[Context Summary"):
                user_topics.append(first_line[:120])
        elif role == "tool":
            # Summarize tool outcomes
            text = content[:200]
            if text:
                tool_actions.append(text)
        elif role == "assistant":
            # Note assistant responses (brief)
            first_line = content.split("\n")[0].strip() if content else ""
            if first_line:
                user_topics.append(f"→ {first_line[:100]}")

    # Assemble summary
    if user_topics:
        parts.append("Topics discussed:")
        for i, topic in enumerate(user_topics[:15], 1):
            parts.append(f"  {i}. {topic}")

    if tool_actions:
        parts.append("\nTool operations performed:")
        for action in tool_actions[:10]:
            # Truncate and clean
            clean = action.replace("\n", " ")[:100]
            parts.append(f"  - {clean}")

    if not parts:
        parts.append(f"({len(messages)} messages of prior conversation)")

    return "\n".join(parts)
