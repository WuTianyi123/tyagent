"""Context compression for tyagent.

Single-pass compression triggered by API 400 "context too long":
  - Uses precise token counts from usage.prompt_tokens to find a cut point
    at roughly cut_ratio of the context window
  - Aligns the cut to a clean conversation boundary (user message or
    complete assistant reply)
  - Summarizes pre-cut messages via LLM, keeps the tail verbatim
  - KV-cache friendly: stable [summary] + [tail] prefix structure
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Summarization prompt for the compression LLM
_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a compression assistant. Given an excerpt of a conversation "
    "between a user, an AI assistant, and tool execution results, produce "
    "a brief factual summary (under 200 words). Retain any data, insights, "
    "or decisions that remain relevant. Omit technical details that are no "
    "longer actionable."
)





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
