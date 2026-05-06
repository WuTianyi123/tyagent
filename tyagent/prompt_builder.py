"""System prompt assembly for tyagent.

Layers (in order):
  1. base_instructions/default.md  — foundational behaviour rules
  2. identity.md                   — profile-specific agent identity (fallback: prompts/identity/default.md)
  3. User custom system_prompt     — from config, if non-default
  4. Session metadata              — model, provider
  5. Memory blocks                 — from MemoryStore snapshot (MEMORY.md + USER.md only)

User context comes exclusively from the memory tool's USER.md store,
NOT from a separate user.md file at the profile root.  The two would
be redundant — the memory tool is the canonical source.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Prompt file loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(*parts: str) -> str:
    """Load a prompt text file from ``tyagent/prompts/``.

    Usage: ``_load_prompt("identity", "default.md")`` loads
    ``tyagent/prompts/identity/default.md``.
    """
    path = _PROMPTS_DIR.joinpath(*parts)
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Failed to load prompt file: {path}") from exc


# ---------------------------------------------------------------------------
# Constants (loaded from prompt files)
# ---------------------------------------------------------------------------

BASE_INSTRUCTIONS = _load_prompt("base_instructions", "default.md")
TYAGENT_IDENTITY = _load_prompt("identity", "default.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_if_exists(path: Path) -> Optional[str]:
    """Read file contents if it exists, stripping whitespace."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _get_memory_blocks() -> List[str]:
    """Return memory blocks for system prompt injection.

    Each target (memory / user) is rendered as an independent block
    with full entry content, using the frozen snapshot captured at
    ``MemoryStore.load_from_disk()`` time.

    Returns an empty list if no MemoryStore is available (e.g. tests,
    isolated child agents, or store not yet initialised).
    """
    try:
        from tyagent.tools.memory_tool import get_store

        store = get_store()
        if store is None:
            return []
        blocks: List[str] = []
        for target in ("memory", "user"):
            block = store.format_for_system_prompt(target)
            if block:
                blocks.append(block)
        return blocks
    except Exception:
        return []


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

    Called once at agent init; cached on the agent for prefix-cache
    stability across turns.  Compact-on-start is handled by Codex's
    ``SUMMARY_PREFIX`` — no special rebuild logic needed on compression.
    """
    parts: list[str] = []

    # Layer 1: base instructions — foundational behavioural rules
    parts.append(BASE_INSTRUCTIONS)

    # Layer 2: identity — from profile identity.md or packaged default
    identity_text = None
    if home_dir is not None:
        identity_text = _read_if_exists(home_dir / "identity.md")
    parts.append(identity_text or TYAGENT_IDENTITY)

    # Layer 3: user's custom system_prompt (skip if empty/unset)
    if user_prompt and user_prompt.strip():
        parts.append(user_prompt)

    # Layer 4: session metadata
    parts.append(f"Current model: {model}")

    # Layer 5: memory blocks — from MemoryStore snapshot
    for block in _get_memory_blocks():
        parts.append(block)

    return "\n\n".join(parts)
