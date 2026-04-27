"""Context compression for tyagent.

Two compression strategies, applied in order when the context exceeds budget:

**Level 1 — Deterministic tool-dropping** (free, zero LLM calls):
  - Drop old tool messages before the last user message
  - Strip tool_calls from old assistant messages before the last user message
  - Preserves system prompt, all user inputs, and all active tool chains

**Level 2 — LLM summarization** (planned, not yet implemented):
  - Summarize middle turns with an auxiliary model
  - Protected head and tail regions

Compression is triggered by real token counts from the LLM API response,
with a character-based estimate as fallback for the first turn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough chars-per-token estimate for pre-flight (no API call yet).
# Used ONLY on the very first turn before we have real token counts.
# Hermes uses 4.0; tyagent targets DL/CLI agents with mixed CJK content.
_CHARS_PER_TOKEN = 4.0

# Default context window budget for unknown models (in tokens)
_DEFAULT_CONTEXT_TOKENS = 128_000

# Default trigger threshold (percentage of context window)
# Compression fires when used tokens exceed this fraction.
_DEFAULT_THRESHOLD_PCT = 0.60

# Minimum threshold (in tokens): never compress below this
_MINIMUM_THRESHOLD = 20_000


def estimate_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list (pre-flight only).

    Uses 4 chars/token — intentionally conservative to avoid premature
    compression. Used only before the first API call when no real
    token count is available.
    """
    total_chars = sum(len(str(msg)) for msg in messages)
    return int(total_chars // _CHARS_PER_TOKEN) + 1


def _content_chars(messages: List[Dict[str, Any]]) -> int:
    """Count total characters in message content and tool_calls."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", "") or "")
        for tc in msg.get("tool_calls", []) or []:
            func = tc.get("function", {}) or {}
            total += len(tc.get("id", ""))
            total += len(tc.get("type", ""))
            total += len(func.get("name", ""))
            total += len(func.get("arguments", ""))
        rc = msg.get("reasoning_content")
        if isinstance(rc, str):
            total += len(rc)
    return total


def should_compress(
    messages: List[Dict[str, Any]],
    prompt_tokens: Optional[int] = None,
    *,
    threshold_tokens: int = 0,
) -> bool:
    """Return True if the message list exceeds the compression threshold.

    Args:
        messages: The full message list to check.
        prompt_tokens: Real token count from the LLM API response.
            When provided (after first API call), this is the authoritative
            value — the character estimate is only a fallback.
        threshold_tokens: Token budget at which compression triggers.
            Defaults to 60% of a 128K window if not set.
    """
    if threshold_tokens <= 0:
        threshold_tokens = int(_DEFAULT_CONTEXT_TOKENS * _DEFAULT_THRESHOLD_PCT)
    if threshold_tokens < _MINIMUM_THRESHOLD:
        threshold_tokens = _MINIMUM_THRESHOLD

    if prompt_tokens is not None:
        # Authoritative: use real token count from API
        return prompt_tokens > threshold_tokens

    # Fallback: character estimate (first turn)
    estimated = estimate_tokens_rough(messages)
    return estimated > threshold_tokens


# Backward compat alias
should_compress_old = should_compress


def build_api_messages(
    messages: List[Dict[str, Any]],
    max_chars: int = 0,
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
        max_chars: Character budget (unused with token-based triggering;
            kept for backward compat).

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


def resolve_threshold(
    context_tokens: Optional[int] = None,
    threshold_pct: Optional[float] = None,
) -> int:
    """Resolve a compression threshold from optional model config.

    Args:
        context_tokens: Model's context window size.
            Default: 128_000.
        threshold_pct: Fraction of context window to trigger compression.
            Default: 0.60.

    Returns:
        Threshold in tokens.
    """
    ctx = context_tokens or _DEFAULT_CONTEXT_TOKENS
    pct = threshold_pct if threshold_pct is not None else _DEFAULT_THRESHOLD_PCT
    threshold = int(ctx * pct)
    if threshold < _MINIMUM_THRESHOLD:
        threshold = _MINIMUM_THRESHOLD
    return threshold


# Backward-compat aliases
compress_messages = build_api_messages
DEFAULT_MAX_CHARS = int(_DEFAULT_CONTEXT_TOKENS * _DEFAULT_THRESHOLD_PCT * _CHARS_PER_TOKEN)
