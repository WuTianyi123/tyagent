"""System prompt assembly for tyagent.

Layers (in order):
  1. identity.md  — profile-specific agent identity (fallback: TYAGENT_IDENTITY)
  2. user.md      — profile-specific user context (optional)
  3. User system_prompt override (from config, if non-default)
  4. Session metadata (model, provider)

Design follows Hermes: build once per session, cache, rebuild only on
context compression to maximise prefix-cache hits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYAGENT_IDENTITY = (
    "You are tyagent, an AI agent framework. "
    "You are helpful, knowledgeable, and direct. "
    "You communicate clearly and prioritise being genuinely useful."
)

# Default system_prompt value in AgentConfig.  We skip injecting the
# user's custom prompt when it still matches this default so the system
# prompt stays lean.
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_if_exists(path: Path) -> Optional[str]:
    """Read file contents if it exists, stripping whitespace."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    model: str,
    user_prompt: str = "",
    *,
    home_dir: Optional[Path] = None,
) -> str:
    """Assemble the system prompt for a session.

    Called once at session start; cached on the agent.  Rebuilt after
    context compression so the identity stays fresh.
    """
    parts: list[str] = []

    # Layer 1: identity — from identity.md or hardcoded fallback
    if home_dir is not None:
        identity_text = _read_if_exists(home_dir / "identity.md")
        if identity_text:
            parts.append(identity_text)

    if not parts:
        parts.append(TYAGENT_IDENTITY)

    # Layer 2: user context — from user.md
    if home_dir is not None:
        user_text = _read_if_exists(home_dir / "user.md")
        if user_text:
            parts.append(user_text)

    # Layer 3: user's custom system_prompt only if non-default
    if user_prompt and user_prompt != _DEFAULT_SYSTEM_PROMPT:
        parts.append(user_prompt)

    # Layer 4: session metadata
    parts.append(f"Current model: {model}")

    return "\n\n".join(parts)
