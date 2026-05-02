"""System prompt assembly for tyagent.

Layers (in order):
  1. Agent identity — hardcoded TYAGENT_IDENTITY
  2. User system_prompt override (from config, if non-default)
  3. Session metadata (model, provider)

Design follows Hermes: build once per session, cache, rebuild only on
context compression to maximise prefix-cache hits.
"""

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
# Builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    model: str,
    user_prompt: str = "",
) -> str:
    """Assemble the system prompt for a session.

    Called once at session start; cached on the agent.  Rebuilt after
    context compression so the identity stays fresh.
    """
    parts = [TYAGENT_IDENTITY]

    # Append user's custom system_prompt only if it differs from default.
    if user_prompt and user_prompt != _DEFAULT_SYSTEM_PROMPT:
        parts.append(user_prompt)

    parts.append(f"Current model: {model}")

    return "\n\n".join(parts)
