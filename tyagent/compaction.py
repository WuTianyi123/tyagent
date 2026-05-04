"""Codex-style context compaction for tyagent.

Architecture mirrors Codex CLI's compact.rs:

Conversation messages and system configuration are ALWAYS separate:
  - ``self._messages`` = conversation only (user/assistant/tool), NEVER contains system
  - System prompt is built fresh per turn, injected at API call time
  - Tools are sent via the ``tools`` API parameter, never embedded in messages

Compaction only touches conversation messages:
  1. ``collect_user_messages()`` extracts user-role text, skipping previous summaries
  2. ``select_tail_messages()`` keeps ≤20K tokens of the most recent user messages
  3. A standalone LLM call (``run_compact()``) generates the summary
  4. ``build_compacted_history()`` assembles: [tail_user_msgs..., summary_assistant_msg]

After compaction, nothing needs re-injection — system prompt and tools were
never in the compaction path to begin with.

Pre-turn compaction runs before the first API call of a turn, proactively.
Mid-turn compaction runs after tool call output pushes usage over the limit.
There is NO reactive "catch 400 then compact" fallback — the proactive estimate
is authoritative.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

# Maximum tokens of recent user messages to preserve verbatim.
# Exact value from Codex CLI compact.rs L44.
COMPACT_USER_MESSAGE_MAX_TOKENS: int = 20_000

# Prefixed to the summary so the receiving model treats it as background
# reference, not an active instruction.  Exact text from Codex CLI
# templates/compact/summary_prefix.md.
SUMMARY_PREFIX: str = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process. You also have access to the state of the tools that "
    "were used by that language model. Use this to build on the work that has "
    "already been done and avoid duplicating work. Here is the summary produced "
    "by the other language model, use the information in this summary to assist "
    "with your own analysis:"
)

# Compaction prompt sent as a standalone turn.  Exact text from Codex CLI
# templates/compact/prompt.md.
COMPACTION_PROMPT: str = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. "
    "Create a handoff summary for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM "
    "seamlessly continue the work."
)

# Approximate chars-per-token for budget calculations.
_CHARS_PER_TOKEN: int = 4


# ── Public API ──────────────────────────────────────────────────────────────


def is_summary_message(message: str) -> bool:
    """Return True if *message* is a previously injected compaction summary.

    Matches by ``SUMMARY_PREFIX`` prefix, same strategy as Codex CLI's
    ``is_summary_message()`` (compact.rs L386-388).
    """
    return message.startswith(SUMMARY_PREFIX)


def collect_user_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract user message text from conversation history.

    Rules (mirroring Codex CLI ``collect_user_messages()``):
      - Only ``role == "user"`` messages are collected
      - Previous compaction summaries are filtered out via ``is_summary_message()``
      - Multimodal content arrays are flattened to text parts only
    """
    result: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            content = " ".join(parts)
        if not isinstance(content, str):
            continue
        if is_summary_message(content):
            continue
        result.append(content)
    return result


def select_tail_messages(
    user_messages: List[str],
    max_tokens: int = COMPACT_USER_MESSAGE_MAX_TOKENS,
) -> List[str]:
    """Select recent user messages up to *max_tokens*, walking backward.

    Algorithm mirrors Codex CLI ``build_compacted_history_with_limit()``
    (compact.rs L460-478):

      1. Iterate user messages in reverse (most recent first)
      2. If message fits in remaining budget → keep entirely
      3. If doesn't fit → truncate to remaining budget, stop
      4. Reverse back to original order

    This is a token budget, not a message-count budget.
    """
    if max_tokens <= 0:
        return []
    selected: List[str] = []
    remaining = max_tokens
    for msg in reversed(user_messages):
        tokens = _approx_token_count(msg)
        if tokens <= remaining:
            selected.append(msg)
            remaining -= tokens
        else:
            char_budget = remaining * _CHARS_PER_TOKEN
            selected.append(msg[:char_budget])
            break
    selected.reverse()
    return selected


def build_compacted_history(
    selected_messages: List[str],
    summary_text: str,
) -> List[Dict[str, Any]]:
    """Build the replacement messages list after compaction.

    Structure::

        [
          {"role": "user", "content": msg1},
          {"role": "user", "content": msg2},
          ...
          {"role": "assistant", "content": "{SUMMARY_PREFIX}\\n{summary}"},
        ]

    The summary is injected as a ``user``-role message (matching Codex CLI's
    ``build_compacted_history()`` which uses ``role: "user"`` for the summary).
    This is fine for OpenAI-compatible APIs which accept consecutive ``user``
    messages; the next real assistant response from the tool loop follows
    naturally.
    """
    history: List[Dict[str, Any]] = []
    for msg_text in selected_messages:
        history.append({"role": "user", "content": msg_text})
    history.append({
        "role": "user",
        "content": f"{SUMMARY_PREFIX}\n{summary_text}",
    })
    return history


def total_token_estimate(messages: List[Dict[str, Any]], *, system_prompt: str = "") -> int:
    """Rough total-token estimate for the conversation + optional system prompt.

    Used for pre-turn and mid-turn threshold checks.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_len = sum(
                len(b.get("text", "")) for b in content
                if isinstance(b, dict)
            )
            total += text_len // _CHARS_PER_TOKEN + 10
        elif isinstance(content, str):
            total += len(content.encode("utf-8")) // _CHARS_PER_TOKEN + 10
        else:
            total += len(str(content)) // _CHARS_PER_TOKEN + 10
        # Account for tool_calls in assistant messages
        for tc in msg.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", "")
            total += len(args) // _CHARS_PER_TOKEN
    if system_prompt:
        total += len(system_prompt) // _CHARS_PER_TOKEN
    return total


# ── Internal helpers ────────────────────────────────────────────────────────


def _approx_token_count(text: str) -> int:
    """Rough token estimate for budget calculations.

    Uses byte-length in UTF-8 rather than character count to get a
    conservative bound for CJK text (where one character can be 3 bytes).
    4 bytes ≈ 1 token works for both ASCII (1 byte/char) and CJK (3 bytes/char).
    This replaces the simpler ``len(text) // 4`` which underestimates CJK by
    up to 3x, risking context overflow before proactive compaction fires.
    """
    return len(text.encode("utf-8")) // _CHARS_PER_TOKEN


def _serialize_messages(messages: List[Dict[str, Any]]) -> str:
    """Serialize conversation messages to text for the compaction prompt input.

    Each message formatted as ``[role]: content`` with double-newline
    separation, matching Codex CLI's prompt construction semantics.
    """
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict)
            )
        elif not isinstance(content, str):
            content = str(content)
        line = content
        # Append tool call details inline for context
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            line += f"\n[tool_call]: {fn.get('name', '?')}({fn.get('arguments', '')})"
        lines.append(f"[{role}]: {line}")
    return "\n\n".join(lines)


# ── Core compaction function ────────────────────────────────────────────────


async def run_compact(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str,
    http_client: httpx.AsyncClient,
    *,
    max_retries: int = 2,
) -> Optional[List[Dict[str, Any]]]:
    """Run compaction and return the compacted message list.

    Steps:
      1. Collect user messages from *messages* (skipping previous summaries)
      2. Select the most recent tail (≤20K tokens by default)
      3. Serialize the full conversation and send it with ``COMPACTION_PROMPT``
         to the LLM as a standalone turn
      4. Extract the summary from the LLM's response
      5. Assemble compacted history: [tail_user_msgs, summary_assistant_msg]

    Returns ``None`` if compaction fails (all retries exhausted) — the caller
    should fall back to the original messages.

    The *messages* list is NOT mutated — the caller receives a new list.
    """
    # 1. Collect user messages
    user_msgs = collect_user_messages(messages)
    if not user_msgs:
        logger.warning("No user messages to compact")
        return None

    # 2. Select tail
    tail = select_tail_messages(user_msgs)
    if not tail:
        logger.warning("No tail messages fit within budget")
        return None

    # 3-4. Call LLM for summary
    serialized = _serialize_messages(messages)
    compaction_input = f"{serialized}\n\n---\n\n{COMPACTION_PROMPT}"

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": compaction_input}],
                "max_tokens": 2048,
                "temperature": 0.0,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = await http_client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            if resp.status_code >= 400:
                body = resp.text[:500]
                last_error = f"HTTP {resp.status_code}: {body}"
                logger.warning("Compaction API error (attempt %d/%d): %s",
                               attempt + 1, max_retries + 1, last_error)
                continue

            data = resp.json()
            summary = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not summary:
                last_error = "Empty summary in response"
                logger.warning("Compaction empty summary (attempt %d/%d)",
                               attempt + 1, max_retries + 1)
                continue

            break

        except httpx.TimeoutException as e:
            last_error = f"Timeout: {e}"
            logger.warning("Compaction timeout (attempt %d/%d)",
                           attempt + 1, max_retries + 1)
            continue
        except httpx.HTTPError as e:
            last_error = f"HTTP error: {e}"
            logger.warning("Compaction HTTP error (attempt %d/%d)",
                           attempt + 1, max_retries + 1)
            continue
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("Compaction unexpected error (attempt %d/%d): %s",
                           attempt + 1, max_retries + 1, last_error)
            continue
    else:
        logger.error("Compaction failed after %d attempts: %s",
                     max_retries + 1, last_error)
        return None

    # 5. Build compacted history
    compacted = build_compacted_history(tail, summary)
    logger.info(
        "Compaction: %d messages -> %d messages (preserved %d of %d user msgs)",
        len(messages), len(compacted),
        len(tail), len(user_msgs),
    )
    return compacted
