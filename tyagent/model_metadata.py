"""Model metadata lookup — context length, provider inference.

Ported from Hermes's ``agent/model_metadata.py``, stripped to the subset
tyagent needs.  Resolution order:

1. User-supplied ``context_length`` (highest priority)
2. This module's ``DEFAULT_CONTEXT_LENGTHS`` table (substring match)
3. Fallback: 128 000 tokens
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

logger = logging.getLogger(__name__)

# Provider names that appear as a prefix before a model ID (e.g.
# ``deepseek:deepseek-v4-pro``).  Stripped during lookup so the bare
# model name reaches the mapping table.
_PROVIDER_PREFIXES: frozenset[str] = frozenset({
    "openrouter", "nous", "openai-codex", "copilot",
    "gemini", "anthropic", "deepseek",
    "openai", "azure", "bedrock",
    "qwen-oauth", "zai", "kimi", "kimi-coding", "moonshot",
    "stepfun", "minimax", "xai", "groq", "local",
    "custom", "ollama",
})

# ── Hardcoded context-length table (longest-substring-first) ──────
# These fire when no explicit context_length is set in config.
# Keys are sorted longest-first so ``deepseek-v4-pro`` matches before ``deepseek``.
# Values sourced from official API docs / models.dev / Hermes defaults.
# fmt: off
DEFAULT_CONTEXT_LENGTHS: Dict[str, int] = {
    # Anthropic Claude
    "claude-opus-4-7":        1_000_000,
    "claude-opus-4.7":        1_000_000,
    "claude-opus-4-6":        1_000_000,
    "claude-opus-4.6":        1_000_000,
    "claude-sonnet-4-6":      1_000_000,
    "claude-sonnet-4.6":      1_000_000,
    "claude-sonnet-4":          200_000,
    "claude-sonnet-4-1":      1_000_000,
    "claude-sonnet-4.1":      1_000_000,
    "claude-opus-4":            200_000,
    "claude-haiku-4-6":       1_000_000,
    "claude-haiku-4.6":       1_000_000,
    "claude-haiku-4":           200_000,
    "claude":                   200_000,
    # OpenAI
    "gpt-5.5":               1_050_000,
    "gpt-5.4":               1_050_000,
    "gpt-5.4-nano":            400_000,
    "gpt-5.4-mini":            400_000,
    "gpt-5.1-chat":            128_000,
    "gpt-5":                   400_000,
    "gpt-4.1":               1_047_576,
    "gpt-4-turbo":             128_000,
    "gpt-4o":                  128_000,
    "gpt-4o-mini":             128_000,
    "gpt-4":                   128_000,
    "o4-mini":                 200_000,
    "o3-mini":                 200_000,
    "o3":                      200_000,
    "o1":                      200_000,
    "o1-mini":                 128_000,
    "o1-pro":                  200_000,
    # DeepSeek — V4 family (1M), legacy R1 (128K)
    "deepseek-v4-pro":       1_000_000,
    "deepseek-v4-flash":     1_000_000,
    "deepseek-chat":         1_000_000,   # API alias → v4-flash
    "deepseek-reasoner":     1_000_000,   # API alias → v4-flash (not the same as deepseek-r1)
    "deepseek-r1":             128_000,   # explicit R1 model — 128K context
    "deepseek-v3":              64_000,
    "deepseek":                128_000,
    # Google Gemini
    "gemini-3":                409_600,
    "gemini-2.5":            1_048_576,
    "gemini-2.0-flash":      1_048_576,
    "gemini":                1_048_576,
    # Meta Llama
    "llama-4":               1_000_000,
    "llama-3.3":                65_536,
    "llama-3.1":               131_072,
    "llama-3":                 131_072,
    "llama":                   131_072,
    # Qwen (Alibaba)
    "qwen3-coder-plus":      1_000_000,
    "qwen3-coder":             262_144,
    "qwen3-max":               262_144,
    "qwen3":                   131_072,
    "qwen-max-latest":         131_072,
    "qwen-plus-latest":        131_072,
    "qwen-turbo-latest":       131_072,
    "qwen":                    131_072,
    # MiniMax
    "minimax-m2.5":            204_800,
    "minimax-m2":              204_800,
    "minimax":                 204_800,
    # GLM
    "glm-5":                   202_752,
    "glm-4":                   202_752,
    "glm":                     202_752,
    # Kimi / Moonshot
    "kimi-k2.6":               262_144,
    "kimi-k2.5":               262_144,
    "kimi-k2-thinking":        262_144,
    "kimi-k2":                 262_144,
    "kimi":                    262_144,
    # xAI Grok
    "grok-4-1-fast":         2_000_000,
    "grok-4.20":             2_000_000,
    "grok-4-fast":           2_000_000,
    "grok-code-fast":          256_000,
    "grok-4":                  256_000,
    "grok-3":                  131_072,
    "grok-2":                  131_072,
    "grok":                    131_072,
    # Misc
    "gemma-4":                 256_000,
    "gemma-3":                 131_072,
    "gemma":                     8_192,
    "mixtral":                 131_072,
    "mistral-large":           262_144,
    "mistral-small":           131_072,
    "mistral":                 131_072,
    "nemotron":                131_072,
    "trinity":                 262_144,
    "hy3-preview":             256_000,
    "mimo-v2.5":             1_048_576,
    "mimo-v2":               1_048_576,
    "elephant":                262_144,
}
# fmt: on

# Sorted longest-first for correct substring matching.
_SORTED_KEYS = sorted(DEFAULT_CONTEXT_LENGTHS, key=len, reverse=True)

FALLBACK_CONTEXT_LENGTH = 256_000


def _strip_provider_prefix(model: str) -> str:
    """Strip a recognised provider prefix from a model string.

    ``deepseek:deepseek-v4-pro`` → ``deepseek-v4-pro``
    ``ollama:qwen3.5:27b``       → ``qwen3.5:27b``      (unchanged — model:tag)
    """
    if ":" not in model or model.startswith("http"):
        return model
    prefix, suffix = model.split(":", 1)
    if prefix.strip().lower() in _PROVIDER_PREFIXES:
        return suffix
    return model


@lru_cache(maxsize=256)  # 256 slots for commonly queried model names; evicts LRU
def get_model_context_length(model: str, *, context_length: int | None = None) -> int:
    """Return the context length for *model*.

    Resolution order:
    1. *context_length* — explicit user override
    2. ``DEFAULT_CONTEXT_LENGTHS`` table — longest-substring match
    3. ``FALLBACK_CONTEXT_LENGTH`` (256K)
    """
    if context_length is not None and isinstance(context_length, int) and context_length > 0:
        return context_length

    bare = _strip_provider_prefix(model).lower()
    for key in _SORTED_KEYS:
        idx = bare.find(key)
        if idx == -1:
            continue
        # Reject false-positives where key is embedded in a longer name
        # (e.g. "non-deepseek-v4-pro" should not match "deepseek-v4-pro")
        before_ok = idx == 0 or bare[idx - 1] in ("-", "/", ":")
        after_ok = idx + len(key) == len(bare) or bare[idx + len(key)] in ("-", "/", ":")
        if before_ok and after_ok:
            result = DEFAULT_CONTEXT_LENGTHS[key]
            logger.debug("Context length for %r resolved to %s (matched %r)", model, f"{result:,}", key)
            return result

    logger.debug("Context length for %r not found in table — using fallback %s", model, f"{FALLBACK_CONTEXT_LENGTH:,}")
    return FALLBACK_CONTEXT_LENGTH
