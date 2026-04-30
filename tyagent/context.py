"""Context compression for tyagent.

Level 1 — Deterministic tool-message dropping (free, zero LLM calls):
  - Drop old tool messages before the last user message
  - Strip tool_calls from old assistant messages before the last user message
  - Preserves system prompt, all user inputs, and all active tool chains

Level 2 — LLM-based summarization (aggressive compression):
  - When Level 1 is insufficient, summarizes the entire pre-user
    conversation with a compression LLM
  - Keeps original system prompts separate, inserts the summary
    as an additional system message
  - Replaces all pre-user non-system messages with a single summary
  - Supports a separately configured compression model (cheaper/faster)

Both levels are triggered by API 400 "context too long" errors, not
proactively. Level 1 is tried first (free), then Level 2 (costs tokens).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Level 2: summarization prompt for the compression LLM
_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a compression assistant. Given an excerpt of a conversation "
    "between a user, an AI assistant, and tool execution results, produce "
    "a brief factual summary (under 200 words). Retain any data, insights, "
    "or decisions that remain relevant. Omit technical details that are no "
    "longer actionable."
)


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


async def summarize_middle(
    messages: List[Dict[str, Any]],
    http_client: httpx.AsyncClient,
    model: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
) -> Optional[List[Dict[str, Any]]]:
    """Level 2: LLM-based context summarization.

    When Level 1 (deterministic tool-message dropping) is insufficient,
    Level 2 summarizes the entire pre-user conversation with an LLM and
    inserts the summary as a system message.

    Original system prompts are preserved separately (not summarized).
    The summary is injected as an additional system message after them.

    Args:
        messages: Full message list (never modified).
        http_client: HTTP client for API calls.
        model: The compression model to use.
        api_key: API key for the compression model.
        base_url: Base URL for the compression model API.

    Returns:
        Compressed message list, or None if compression wasn't possible.
    """
    n = len(messages)
    if n < 3:
        return None

    # Find the last user message
    last_user_idx = -1
    for i in range(n - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx <= 0:
        return None  # nothing meaningful to summarize

    # Split pre-user messages: keep system prompts, summarize everything else
    pre_user_system: List[Dict[str, Any]] = []
    pre_user_content: List[Dict[str, Any]] = []
    for msg in messages[:last_user_idx]:
        if msg.get("role") == "system":
            pre_user_system.append(msg)
        else:
            pre_user_content.append(msg)

    if len(pre_user_content) < 1:
        return None  # nothing to summarize

    tail = messages[last_user_idx:]

    # Serialize pre_user_content for the LLM — format as a conversation excerpt
    serialized = json.dumps([
        {
            "role": m.get("role", ""),
            "content": (m.get("content") or "")[:2000],
            "tool_calls": (
                [tc.get("function", {}).get("name", "?")
                 for tc in (m.get("tool_calls") or [])]
                if m.get("tool_calls") else None
            ),
            "tool_call_id": m.get("tool_call_id"),
        }
        for m in pre_user_content
    ], ensure_ascii=False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation excerpt, "
                    "retaining all factual information still relevant:\n\n"
                    + serialized
                ),
            },
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    try:
        resp = await http_client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        summary: Optional[str] = data["choices"][0]["message"].get("content")
        if not summary or not summary.strip():
            logger.warning("Level 2 summarization returned empty response")
            return None
    except Exception:
        logger.exception("Level 2 summarization failed")
        return None

    # Build compressed message list:
    # [original system prompts] + [summary system message] + [tail]
    result: List[Dict[str, Any]] = list(pre_user_system)
    result.append({
        "role": "system",
        "content": f"[Summary of previous conversation] {summary.strip()}",
    })
    result.extend(tail)

    logger.info(
        "Context compressed (level 2): summarized %d pre-user messages via %s "
        "(%d → %d messages, ratio %.1f%%)",
        len(pre_user_content), model, n, len(result),
        100.0 * (1 - len(result) / n),
    )

    return result


# ---------------------------------------------------------------------------
# Single-pass compression (replaces old Level 1 + Level 2 two-step)
# ---------------------------------------------------------------------------


async def compress_context(
    messages: List[Dict[str, Any]],
    http_client: httpx.AsyncClient,
    model: str,
    api_key: str,
    base_url: str,
    token_history: List[tuple],
    context_window: int,
    cut_ratio: float,
) -> Optional[List[Dict[str, Any]]]:
    """Single-pass context compression triggered by API 400 overflow.

    Uses precise token counts from ``usage.prompt_tokens`` (stored in
    *token_history*) to find a cut point at roughly ``context_window *
    cut_ratio`` tokens.  The cut is then aligned backward to a clean
    conversation boundary — either a user message or the end of a complete
    assistant reply (content present, no tool_calls).

    Everything before the cut point is summarized by an LLM; everything
    from the cut point onwards is kept verbatim.

    Returns the compressed message list, or *None* if compression is
    impossible (too few messages, no token history, etc.).
    """
    n = len(messages)
    if n < 3 or not token_history:
        return None

    # — Step 1: find a cut index from the token history ——————
    target_tokens = int(context_window * cut_ratio)
    cut_idx = 0

    for msg_count, prompt_tokens in reversed(token_history):
        if msg_count <= 0:
            continue
        if prompt_tokens <= target_tokens:
            cut_idx = msg_count
            break
        cut_idx = msg_count  # fallback: earliest known point

    if cut_idx <= 0 or cut_idx >= n:
        return None

    # — Step 2: align to a clean conversation boundary ——————
    # Walk backward from cut_idx looking for a natural turn boundary:
    #  • user message → cut just before it (tail starts with user turn)
    #  • assistant message with content and no tool_calls → cut right
    #    after it (tail starts fresh after a complete reply)
    aligned = cut_idx
    for i in range(cut_idx - 1, 0, -1):
        role = messages[i].get("role", "")
        if role == "user":
            aligned = i
            break
        if role == "assistant":
            if messages[i].get("content") and not messages[i].get("tool_calls"):
                aligned = i + 1
                break

    if aligned <= 0 or aligned >= n:
        return None

    pre_cut = messages[:aligned]
    tail = messages[aligned:]

    # — Step 3: separate system prompts from content to summarize —
    pre_system: List[Dict[str, Any]] = []
    pre_content: List[Dict[str, Any]] = []
    for msg in pre_cut:
        if msg.get("role") == "system":
            pre_system.append(msg)
        else:
            pre_content.append(msg)

    if not pre_content:
        return None

    # — Step 4: ask a lightweight LLM to summarise pre_content —
    serialized = json.dumps([
        {
            "role": m.get("role", ""),
            "content": (m.get("content") or "")[:2000],
            "tool_calls": (
                [tc.get("function", {}).get("name", "?")
                 for tc in (m.get("tool_calls") or [])]
                if m.get("tool_calls") else None
            ),
            "tool_call_id": m.get("tool_call_id"),
        }
        for m in pre_content
    ], ensure_ascii=False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation excerpt, "
                    "retaining all factual information still relevant:\n\n"
                    + serialized
                ),
            },
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    try:
        resp = await http_client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        summary: Optional[str] = data["choices"][0]["message"].get("content")
        if not summary or not summary.strip():
            logger.warning("compress_context: LLM returned empty summary")
            return None
    except Exception:
        logger.exception("compress_context: summarization API call failed")
        return None

    # — Step 5: assemble the compressed message list ———————
    result: List[Dict[str, Any]] = list(pre_system)
    result.append({
        "role": "system",
        "content": f"[Summary of previous conversation] {summary.strip()}",
    })
    result.extend(tail)

    logger.info(
        "Context compressed (single-pass): summarized %d pre-cut messages "
        "via %s (%d → %d messages, target ~%d token cut at idx %d→%d)",
        len(pre_content), model, n, len(result),
        target_tokens, cut_idx, aligned,
    )

    return result
