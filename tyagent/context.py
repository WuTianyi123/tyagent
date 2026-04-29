"""Context compression for tyagent.

Level 1 — Deterministic tool-message dropping (free, zero LLM calls):
  - Drop old tool messages before the last user message
  - Strip tool_calls from old assistant messages before the last user message
  - Preserves system prompt, all user inputs, and all active tool chains

Compression is triggered by API 400 "context too long" errors, not proactively.
Level 2 (LLM summarization) is unimplemented — if Level 1 can't fit the context
into the model's window, the error is propagated as-is.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_api_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a compressed message list for the LLM API.

    **Level 1 compression** — deterministic tool-message dropping.
    This is always safe: it only removes tool results and tool_calls
    metadata that are no longer needed because the user has moved on
    to a new topic/question.

    Strategy:
    - Always keep the system prompt and all non-tool messages
      (user, assistant text)
    - Keep the LAST user message and ALL tool messages after it
      (active tool chain)
    - Drop tool messages that are BEFORE the last user message
    - Drop tool_calls from assistant messages that are BEFORE
      the last user message
    - Preserve reasoning_content

    The original message list is NEVER modified.

    Args:
        messages: All messages from the session, in chronological order.

    Returns:
        A compressed message list, or the original if no compression needed.
    """
    n = len(messages)
    if n == 0:
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
    old_tool_dropped = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if i > last_user_idx:
            # After last user message — keep everything as-is
            compressed.append(msg)
        elif role == "tool":
            # Before last user message — drop tool messages
            old_tool_dropped += 1
            continue
        elif role == "assistant":
            # Before last user message — keep content+reasoning
            # but drop tool_calls
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

    if old_tool_dropped > 0:
        logger.info(
            "Context compressed (level 1): dropped %d old tool messages "
            "before last user message at idx %d (%d → %d messages)",
            old_tool_dropped, last_user_idx, n, len(compressed),
        )

    return compressed
