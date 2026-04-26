"""Context compression for tyagent.

When a conversation grows too long for the LLM's context window,
this module builds a compressed API message list by deterministically
dropping old tool messages while preserving conversation structure.

Strategy:
- Always keep the system prompt and all non-tool messages (user, assistant text)
- Keep the LAST user message and ALL tool messages after it (active tool chain)
- Drop tool messages that are BEFORE the last user message
- Drop tool_calls from assistant messages that are BEFORE the last user message

This is deterministic — no LLM call needed, zero extra API cost.
The original messages in the database are NEVER modified.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Rough token estimation: ~3.5 chars per token for mixed content.
# Using 3.5 (slightly overestimates token count) as a conservative choice
# to avoid hitting actual context window limits.
_CHARS_PER_TOKEN = 3.5

# Default context budget: ~280k chars ≈ 70k tokens
# Safe for most 128k+ context models (GPT-4o, Claude 3.5, DeepSeek, etc.)
# Leaves headroom for system prompt, formatting overhead, and tool definitions.
_DEFAULT_MAX_CHARS = 280_000


def _content_chars(messages: List[Dict[str, Any]]) -> int:
    """Count total characters in message content and tool_calls."""
    total = 0
    for msg in messages:
        total += len(msg.get("content", "") or "")
        for tc in msg.get("tool_calls", []) or []:
            func = tc.get("function", {}) or {}
            total += len(tc.get("id", ""))
            total += len(tc.get("type", ""))
            total += len(func.get("name", ""))
            total += len(func.get("arguments", ""))
    return total


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token count estimation from message list.

    Counts: content, reasoning_content, tool_calls arguments/name/id/type.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        # Reasoning content also consumes tokens
        rc = msg.get("reasoning_content")
        if isinstance(rc, str):
            total_chars += len(rc)
        # Tool call metadata all consumes tokens
        for tc in msg.get("tool_calls", []) or []:
            func = tc.get("function", {}) or {}
            total_chars += len(tc.get("id", ""))
            total_chars += len(tc.get("type", ""))
            total_chars += len(func.get("name", ""))
            total_chars += len(func.get("arguments", ""))
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
        return _content_chars(messages) > max_chars
    return False


def build_api_messages(
    messages: List[Dict[str, Any]],
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Build a compressed message list for the LLM API.

    Strategy: drop tool messages before the last user message, but
    keep the active tool chain (last user + everything after it).

    The original message list is NOT modified.

    Args:
        messages: All messages from the session, in chronological order.
        max_chars: Character budget for the final message list.

    Returns:
        A compressed message list, or the original if already within budget.
    """
    n = len(messages)
    if n == 0:
        return messages

    # Check if already within budget (fast path)
    total_chars = _content_chars(messages)
    if total_chars <= max_chars:
        return messages

    # Find the last user message
    last_user_idx = -1
    for i in range(n - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    # If no user message, return as-is (shouldn't happen)
    if last_user_idx < 0:
        return messages

    # Build filtered list
    compressed: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if i > last_user_idx:
            # After last user message — keep everything as-is
            compressed.append(msg)
        elif role == "tool":
            # Before last user message — drop tool messages
            continue
        elif role == "assistant":
            # Before last user message — keep content+reasoning but drop tool_calls
            new_msg: Dict[str, Any] = {"role": "assistant"}
            content = msg.get("content")
            if content is not None:
                new_msg["content"] = content
            reasoning = msg.get("reasoning_content")
            if reasoning is not None:
                new_msg["reasoning_content"] = reasoning
            # tool_calls intentionally omitted — this assistant msg's
            # tools were responded to before the last user question
            compressed.append(new_msg)
        else:
            # system, user — keep as-is
            compressed.append(msg)

    # Log compression stats
    original_chars = total_chars
    compressed_chars = _content_chars(compressed)
    logger.info(
        "Context compressed: %d → %d messages, %dk → %dk chars "
        "(last user at idx %d, dropped %d tool msgs before it)",
        n, len(compressed),
        original_chars // 1000, compressed_chars // 1000,
        last_user_idx,
        n - len(compressed),
    )

    # If somehow still over budget (very unlikely), log a warning
    still_over = _content_chars(compressed)
    if still_over > max_chars:
        logger.warning(
            "After compression, context still exceeds budget "
            "(%dk > %dk). Consider splitting or increasing max_chars.",
            still_over // 1000, max_chars // 1000,
        )

    return compressed


# Backward-compat alias for agent.py
compress_messages = build_api_messages

# Public export of default budget
DEFAULT_MAX_CHARS = _DEFAULT_MAX_CHARS
