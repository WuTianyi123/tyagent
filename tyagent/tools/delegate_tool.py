"""Delegate task tool for tyagent — spawn child agents with isolated context.

Each child agent gets:
  - A fresh conversation (no parent history)
  - A restricted toolset (blocked: spawn_task, wait_task, close_task, list_tasks, memory)
  - Summary-only result (no intermediate tool calls in parent context)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from tyagent.agent import TyAgent
from tyagent.events import EventCollector
from tyagent.tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Tools that children must never have access to:
# - All spawn/wait/close/list tools: no recursive sub-agent management
# - memory: no cross-session writes to shared memory
DELEGATE_BLOCKED_TOOLS = frozenset(
    ["spawn_task", "wait_task", "close_task", "list_tasks", "memory"]
)

DEFAULT_SUBAGENT_MAX_TOOL_TURNS = 30
DEFAULT_CHILD_TIMEOUT = 600.0  # asyncio.wait_for timeout for each child chat() call


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


async def _run_child_async(
    task_id: str,
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
    tool_progress_callback: Any = None,
    collector: Optional[EventCollector] = None,
    home_dir: Optional[Path] = None,
    context_length: Optional[int] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    http_timeout: float = 120.0,
    shutdown_timeout: float = 5.0,
) -> Dict[str, Any]:
    """Run a child agent as an asyncio task in the shared event loop.
    On completion, notifies the collector. Never raises — errors captured in result dict.
    """
    t0 = time.monotonic()
    child_system = system_prompt
    if context:
        child_system = f"{system_prompt}\n\nTask context: {context}"

    child = TyAgent(
        model=model, api_key=api_key, base_url=base_url,
        max_tool_turns=max_tool_turns, system_prompt=child_system,
        reasoning_effort=reasoning_effort, compression=compression,
        home_dir=home_dir, context_length=context_length,
        max_tokens=max_tokens,
        temperature=temperature,
        http_timeout=http_timeout,
        shutdown_timeout=shutdown_timeout,
    )
    child_messages = [{"role": "user", "content": goal}]
    tool_defs = registry.get_definitions(names=tool_names)

    try:
        summary = await asyncio.wait_for(
            child.chat(child_messages, tools=tool_defs,
                        tool_progress_callback=tool_progress_callback),
            timeout=DEFAULT_CHILD_TIMEOUT,
        )
        result = {
            "success": True,
            "summary": (summary.strip() if summary else ""),
            "error": None,
            "duration_seconds": round(time.monotonic() - t0, 2),
        }
    except asyncio.TimeoutError:
        result = {
            "success": False, "summary": None,
            "error": "Child timed out after 600s",
            "duration_seconds": round(time.monotonic() - t0, 2),
        }
    except BaseException as exc:
        result = {
            "success": False, "summary": None,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.monotonic() - t0, 2),
        }
    finally:
        try: await child.close()
        except Exception: pass

    if collector is not None:
        collector.notify_child_done(task_id, result)
    return result


def _build_parent_compression(parent: TyAgent) -> Any:
    """Return the parent's compression config for cloning to child agents."""
    return getattr(parent, "_compression_config", None)


def _build_allowed_tools(toolsets: Optional[List[str]] = None) -> List[str]:
    """Return tool names allowed for a child agent."""
    all_names = registry.get_all_names()
    allowed = [n for n in all_names if n not in DELEGATE_BLOCKED_TOOLS]
    if toolsets:
        allowed = [n for n in toolsets if n in allowed]
    return allowed


async def _handle_spawn_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Spawn a child agent as an asyncio task. Returns task_id immediately."""
    goal = args.get("goal", "").strip()
    if not goal:
        return tool_error("goal is required for spawn_task.")
    if parent_agent is None:
        return tool_error("spawn_task requires a session agent.")

    context = args.get("context") or None
    toolsets: Optional[List[str]] = args.get("toolsets") or None
    max_tool_turns = args.get("max_tool_turns", DEFAULT_SUBAGENT_MAX_TOOL_TURNS)
    try:
        max_tool_turns = int(max_tool_turns)
    except (TypeError, ValueError):
        return tool_error("max_tool_turns must be an integer.")
    if max_tool_turns < 1:
        return tool_error("max_tool_turns must be at least 1.")
    if max_tool_turns > 200:
        return tool_error("max_tool_turns must be at most 200.")

    allowed = _build_allowed_tools(toolsets)

    parent_cb = getattr(parent_agent, "_tool_progress_callback", None)
    child_cb = None
    if parent_cb is not None:
        def _child_progress(tn: str, ai: dict) -> None:
            parent_cb(tn, ai, prefix="📤 ")
        child_cb = _child_progress

    task_id = str(uuid.uuid4())[:12]  # 48-bit — negligible collision risk in practice

    # Lazy create collector
    if parent_agent._event_collector is None:
        parent_agent._event_collector = EventCollector()

    child_coro = _run_child_async(
        task_id=task_id,
        model=parent_agent.model, api_key=parent_agent.api_key,
        base_url=parent_agent.base_url,
        system_prompt=parent_agent.system_prompt,
        reasoning_effort=parent_agent.reasoning_effort,
        goal=goal, tool_names=allowed,
        max_tool_turns=max_tool_turns, context=context,
        compression=_build_parent_compression(parent_agent),
        tool_progress_callback=child_cb,
        collector=parent_agent._event_collector,
        home_dir=parent_agent.home_dir,
        context_length=parent_agent.context_length,
        max_tokens=getattr(parent_agent, '_max_tokens', 4096),
        temperature=getattr(parent_agent, '_temperature', 0.7),
        http_timeout=getattr(parent_agent, '_http_timeout', 120.0),
        shutdown_timeout=getattr(parent_agent, '_shutdown_timeout', 5.0),
    )

    child_task = asyncio.get_running_loop().create_task(child_coro)
    parent_agent._bg_tasks[task_id] = child_task

    logger.info("spawn_task: launched %s goal=%r", task_id, goal[:80])
    return json.dumps({"task_id": task_id, "status": "running"}, ensure_ascii=False)


async def _handle_wait_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Wait for spawned child agents. Uses asyncio.wait() with timeout."""
    task_ids = args.get("task_ids", [])
    if not task_ids:
        return tool_error("task_ids is required for wait_task.")
    if not isinstance(task_ids, list):
        return tool_error("task_ids must be a list.")
    timeout = args.get("timeout", 300)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        return tool_error("timeout must be a number.")
    if timeout <= 0:
        return tool_error("timeout must be positive.")
    if parent_agent is None:
        return tool_error("wait_task requires a session agent.")

    futures_map: Dict[str, asyncio.Task] = {}
    missing: List[str] = []
    for tid in task_ids:
        if not isinstance(tid, str):
            missing.append(repr(tid))
            continue
        future = parent_agent._bg_tasks.get(tid)
        if future is None:
            missing.append(tid)
        else:
            futures_map[tid] = future

    results: Dict[str, Any] = {}
    for tid in missing:
        results[tid] = {"error": f"Task not found: {tid}"}

    if futures_map:
        async def _wait_all():
            done, pending = await asyncio.wait(
                list(futures_map.values()), timeout=timeout,
            )
            done_set = set(done)
            for tid, fut in futures_map.items():
                if fut in done_set:
                    try:
                        results[tid] = fut.result()
                        parent_agent._bg_tasks.pop(tid, None)
                    except asyncio.CancelledError:
                        results[tid] = {"error": "Task was cancelled"}
                        parent_agent._bg_tasks.pop(tid, None)
                    except Exception as exc:
                        results[tid] = {"error": f"{type(exc).__name__}: {exc}"}
                        parent_agent._bg_tasks.pop(tid, None)
                else:
                    results[tid] = {"error": f"Timed out after {timeout}s"}

        await _wait_all()

    return json.dumps(results, ensure_ascii=False)


async def _handle_close_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Close (cancel) a running child agent."""
    task_id = args.get("task_id", "").strip()
    if not task_id:
        return tool_error("task_id is required for close_task.")
    if parent_agent is None:
        return tool_error("close_task requires a session agent.")

    task = parent_agent._bg_tasks.pop(task_id, None)
    if task is None:
        return json.dumps({"success": False, "error": f"Task not found: {task_id}"})

    if not task.done():
        task.cancel()
        return json.dumps({"success": True, "message": f"Cancelled {task_id}"})
    else:
        return json.dumps({"success": True, "message": f"{task_id} already completed"})


async def _handle_list_tasks(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """List all spawned child agents and their status."""
    if parent_agent is None:
        return tool_error("list_tasks requires a session agent.")

    tasks = {}
    for tid, task in parent_agent._bg_tasks.items():
        tasks[tid] = {"done": task.done(), "cancelled": task.cancelled()}
    return json.dumps(tasks, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schemas for async sub-agent tools
# ---------------------------------------------------------------------------

SPAWN_TASK_SCHEMA: Dict[str, Any] = {
    "name": "spawn_task",
    "description": (
        "Launch a child agent to work on a task in the background. "
        "Returns immediately with a task_id — the child runs independently. "
        "You can continue working and call wait_task later to collect results.\n\n"
        "If a child completes before you call wait_task, its result "
        "is automatically injected into the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What the child should accomplish."},
            "context": {"type": "string", "description": "Optional background info."},
            "toolsets": {"type": "array", "items": {"type": "string"}, "description": "Optional restricted tool list."},
            "max_tool_turns": {"type": "integer", "description": f"Max tool turns (default: {DEFAULT_SUBAGENT_MAX_TOOL_TURNS}).", "minimum": 1, "maximum": 200},
        },
        "required": ["goal"],
    },
}

WAIT_TASK_SCHEMA: Dict[str, Any] = {
    "name": "wait_task",
    "description": (
        "Wait for one or more spawned child agents to complete. "
        "Returns results keyed by task_id. Uses asyncio.wait — "
        "multiple children are waited on simultaneously."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_ids": {"type": "array", "items": {"type": "string"}, "description": "Task IDs from spawn_task."},
            "timeout": {"type": "number", "description": "Max seconds (default: 300)."},
        },
        "required": ["task_ids"],
    },
}

CLOSE_TASK_SCHEMA: Dict[str, Any] = {
    "name": "close_task",
    "description": (
        "Close (cancel) a running child agent. Use this if a child "
        "is taking too long or producing irrelevant results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task ID from spawn_task."},
        },
        "required": ["task_id"],
    },
}

LIST_TASKS_SCHEMA: Dict[str, Any] = {
    "name": "list_tasks",
    "description": (
        "List all spawned child agents and their current status "
        "(running, completed, or cancelled)."
    ),
    "parameters": {"type": "object", "properties": {}},
}

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(name="spawn_task", schema=SPAWN_TASK_SCHEMA, handler=_handle_spawn_task,
                  description="Launch a child agent in background (non-blocking)", emoji="🚀", wants_parent=True)
registry.register(name="wait_task", schema=WAIT_TASK_SCHEMA, handler=_handle_wait_task,
                  description="Wait for background child agents to complete", emoji="⏳", wants_parent=True)
registry.register(name="close_task", schema=CLOSE_TASK_SCHEMA, handler=_handle_close_task,
                  description="Cancel a running child agent", emoji="🛑", wants_parent=True)
registry.register(name="list_tasks", schema=LIST_TASKS_SCHEMA, handler=_handle_list_tasks,
                  description="List running/completed child agents", emoji="📋", wants_parent=True)
