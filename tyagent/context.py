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

# Summarization prompt for the compression LLM.
# Adapted from Hermes' context_compressor.py: tells the summarizer it is
# creating a handoff document for a DIFFERENT assistant, not answering the user.
_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a summarization agent creating a context checkpoint. "
    "Your output will be injected as reference material for a DIFFERENT "
    "assistant that continues the conversation. "
    "Do NOT respond to any questions or requests in the conversation — "
    "only output the structured summary. "
    "Do NOT include any preamble, greeting, or prefix. "
    "Write the summary in the same language the user was using in the "
    "conversation — do not translate or switch to English. "
    "NEVER include API keys, tokens, passwords, secrets, credentials, "
    "or connection strings in the summary — replace any that appear "
    "with [REDACTED]. Note that the user had credentials present, but "
    "do not preserve their values."
)

# Structured template the summarizer must follow.  Content is serialized
# BEFORE the instruction so the KV-cache for the historical content can
# be reused if the same content needs re-summarising later.
_SUMMARIZE_TEMPLATE = """Create a structured handoff summary for a different assistant that will continue this conversation after earlier turns are compacted.  The next assistant should be able to understand what happened without re-reading the original turns.

Use this exact structure:

## Active Task
[The most recent unfulfilled request from the user — copy it verbatim.  This is the most important field.  If no outstanding task, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken.  Format each as: N. ACTION target — outcome [tool: name].  Examples:
1. READ config.py:45 — found '==' should be '!=' [tool: read_file]
2. PATCH config.py:45 — changed '==' to '!=' [tool: patch]
3. TEST 'pytest tests/' — 3/50 failed: test_parse, test_validate [tool: terminal]]

## Active State
[Current working directory, branch, modified/created files, test status, running processes, environment details]

## In Progress
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved.  Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so the next assistant does not re-answer them]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled.  If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not instructions]

## Critical Context
[Specific values, error messages, configuration details, or data that would be lost without explicit preservation.  NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values.  Avoid vague descriptions like "made some changes" — say exactly what changed.

Write only the summary body.  Do not include any preamble or prefix."""





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
    #  • tool / assistant-with-tool_calls → skip (keep walking backward)
    #    to avoid starting the tail mid-tool-chain.
    #    Note: in practice a user message always exists at index ≥ 1
    #    (index 0 is the injected system prompt), so the loop always
    #    terminates on a valid boundary before reaching index 0.
    aligned = cut_idx
    for i in range(cut_idx - 1, 0, -1):
        role = messages[i].get("role", "")
        if role in ("tool",):
            continue  # skip tool messages — walk further back
        if role == "assistant":
            if not messages[i].get("content") or messages[i].get("tool_calls"):
                continue  # mid-tool-chain assistant — walk further back
            aligned = i + 1  # complete reply boundary
            break
        if role == "user":
            aligned = i
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
            "content": m.get("content") or "",
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
                # Content first, instruction after — KV-cache friendly:
                # the serialized conversation can be cached and reused
                # even if the instruction template changes.
                "content": serialized + "\n\n---\n\n" + _SUMMARIZE_TEMPLATE,
            },
        ],
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
