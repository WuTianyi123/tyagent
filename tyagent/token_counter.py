"""Local token counting using the DeepSeek tokenizer.

Provides ``count_tokens(messages, system_prompt)`` — a fast, offline
alternative to the byte-based ``total_token_estimate`` and the dry-run
API call ``_fetch_prompt_tokens``.

The bundled tokenizer.json (DeepSeek V3) produces content-token counts
that are identical to the V4 API.  The only difference is chat-template
formatting overhead (BOS / EOS / role markers), which we approximate as:

    overhead = 4 + max(0, message_count - 2)

This is accurate to within ~1 token per turn — negligible for compaction
threshold decisions (90 % of 128K–1M context windows).
"""

from __future__ import annotations

import json as _json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_tokenizer: Optional[Tokenizer] = None
_lock = threading.Lock()


def _get_tokenizer() -> Tokenizer:
    """Lazily load the bundled DeepSeek tokenizer (thread-safe)."""
    global _tokenizer
    if _tokenizer is None:
        with _lock:
            if _tokenizer is None:
                path = Path(__file__).parent / "data" / "tokenizer.json"
                logger.debug("Loading tokenizer from %s", path)
                _tokenizer = Tokenizer.from_file(str(path))
    return _tokenizer


def count_tokens(
    messages: List[Dict[str, object]],
    *,
    system_prompt: str = "",
) -> int:
    """Return an accurate estimate of prompt tokens for *messages*.

    Parameters
    ----------
    messages:
        The conversation history (``self._messages``).  Must NOT contain
        a system message — the system prompt is passed separately.
    system_prompt:
        The system prompt string (``self._system_prompt``).

    Returns
    -------
    int
        Estimated prompt tokens, including approximate chat-template overhead.
    """
    t = _get_tokenizer()
    total = 0

    # System prompt
    if system_prompt:
        total += len(t.encode(system_prompt).ids)

    # Message contents
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and content:
            total += len(t.encode(content).ids)

        # Tool-call JSON (assistant messages with function calls)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            # Encode the JSON representation to match what the API sees
            total += len(t.encode(
                _json.dumps(tool_calls, ensure_ascii=False)
            ).ids)

    # Approximate chat-template overhead:
    #   +4 base (BOS + system markers + EOS/generation prompt)
    #   +1 per message beyond the first 2 (role delimiters)
    overhead = 4 + max(0, len(messages) - 2)
    total += overhead

    return total
