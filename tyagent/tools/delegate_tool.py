"""Delegate task tool for tyagent — spawn child agents with isolated context.

Each child agent gets:
  - A fresh conversation (no parent history)
  - A restricted toolset (blocked: delegate_task, memory)
  - Summary-only result (no intermediate tool calls in parent context)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from tyagent.agent import TyAgent
from tyagent.tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Tools that children must never have access to:
# - delegate_task: no recursive delegation
# - memory: no cross-session writes to shared memory
DELEGATE_BLOCKED_TOOLS = frozenset(
    ["delegate_task", "memory"]
)

DEFAULT_SUBAGENT_MAX_TOOL_TURNS = 30


# ---------------------------------------------------------------------------
# Subagent run helper (sync wrapper around async child.chat)
# ---------------------------------------------------------------------------


def _run_child_sync(
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str,
    reasoning_effort: Optional[str],
    goal: str,
    tool_names: List[str],
    max_tool_turns: int,
    context: Optional[str],
    compression: Any = None,
) -> Dict[str, Any]:
    """Run a child agent synchronously and return a result dict."""
    t0 = time.monotonic()
    child_system = system_prompt
    if context:
        child_system = f"{system_prompt}\n\nTask context: {context}"

    child = TyAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tool_turns=max_tool_turns,
        system_prompt=child_system,
        reasoning_effort=reasoning_effort,
        compression=compression,
    )
    child_messages = [{"role": "user", "content": goal}]
    tool_defs = registry.get_definitions(names=tool_names)

    async def _run() -> str:
        try:
            return await child.chat(child_messages, tools=tool_defs)
        finally:
            try:
                await asyncio.shield(child.close())
            except Exception:
                logger.exception("delegate_task: child.close() failed")

    loop: asyncio.AbstractEventLoop | None = None
    try:
        loop = asyncio.new_event_loop()
        timeout = 600.0  # wall-clock timeout for child agent
        summary = loop.run_until_complete(asyncio.wait_for(_run(), timeout=timeout))
    except BaseException as exc:
        return {
            "success": False,
            "summary": None,
            "error": str(exc),
            "duration_seconds": round(time.monotonic() - t0, 2),
        }
    finally:
        if loop is not None:
            loop.close()

    return {
        "success": True,
        "summary": summary.strip() if summary else "",
        "error": None,
        "duration_seconds": round(time.monotonic() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def _build_parent_compression(parent: TyAgent) -> Any:
    """Return the parent's compression config for cloning to child agents."""
    return getattr(parent, "_compression_config", None)


def _handle_delegate_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Handle delegate_task tool calls.

    Spawns a child TyAgent with restricted tools, runs it synchronously,
    and returns a JSON result with summary + metadata.

    *parent_agent* is the TyAgent instance that owns this session —
    injected by registry.dispatch via chat() for race-free concurrency.
    """
    goal = args.get("goal", "").strip()
    if not goal:
        return tool_error("goal is required for delegate_task.")

    context = args.get("context") or None
    toolsets: Optional[List[str]] = args.get("toolsets") or None
    max_tool_turns = args.get("max_tool_turns", DEFAULT_SUBAGENT_MAX_TOOL_TURNS)

    # Coerce max_tool_turns safely
    try:
        max_tool_turns = int(max_tool_turns)
    except (TypeError, ValueError):
        return tool_error("max_tool_turns must be an integer.")
    if max_tool_turns < 1:
        return tool_error("max_tool_turns must be at least 1.")
    if max_tool_turns > 200:
        return tool_error("max_tool_turns must be at most 200.")

    if parent_agent is None:
        return tool_error("No parent agent context — delegate_task requires a session agent.")

    # Build allowed tool names (parent tools minus blocked)
    all_names = registry.get_all_names()
    allowed = [n for n in all_names if n not in DELEGATE_BLOCKED_TOOLS]
    if toolsets:
        allowed = [n for n in toolsets if n in allowed]

    logger.info(
        "delegate_task: spawning subagent goal=%r, tools=%d, max_turns=%d",
        goal[:80], len(allowed), max_tool_turns,
    )

    result = _run_child_sync(
        model=parent_agent.model,
        api_key=parent_agent.api_key,
        base_url=parent_agent.base_url,
        system_prompt=parent_agent.system_prompt,
        reasoning_effort=parent_agent.reasoning_effort,
        goal=goal,
        tool_names=allowed,
        max_tool_turns=max_tool_turns,
        context=context,
        compression=_build_parent_compression(parent_agent),
    )

    logger.info(
        "delegate_task: subagent done success=%s, dur=%.1fs",
        result["success"], result["duration_seconds"],
    )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA: Dict[str, Any] = {
    "name": "delegate_task",
    "description": (
        "Spawn a child agent to work on a task in isolated context. "
        "The child gets a fresh conversation, a restricted toolset (no "
        "delegate_task / memory), and returns a summary only — "
        "intermediate tool calls never enter your context window.\n\n"
        "Use this when:\n"
        "- A subtask would flood your context with intermediate data\n"
        "- You need researched synthesis, code review, or debugging\n"
        "- A reasoning-heavy task would benefit from focused attention\n\n"
        "Do NOT use for single tool calls — just call the tool directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What the child should accomplish. Be specific and self-contained.",
            },
            "context": {
                "type": "string",
                "description": "Optional background info the child needs (file paths, constraints).",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of tool names the child can use. Default: all non-blocked parent tools.",
            },
            "max_tool_turns": {
                "type": "integer",
                "description": f"Max tool-calling turns for the child (default: {DEFAULT_SUBAGENT_MAX_TOOL_TURNS}).",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["goal"],
    },
}

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="delegate_task",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    description="Spawn a child agent with restricted tools and isolated context",
    emoji="📤",
    wants_parent=True,
)
