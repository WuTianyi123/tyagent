"""Delegate task tool for tyagent — Codex v2 sub-agent framework.

This is a complete rewrite of the sub-agent system, aligning with Codex's
v2 model:

- **Hierarchical task paths** — /root/task_a/sub_b  instead of random UUIDs.
- **Persistent agent threads** — each child runs its own ``_agent_loop()``
  and can receive mid-task messages from the parent.
- **Full-duplex mailbox** — parent ↔ child bidirectional communication.
- **8 tools** — spawn_task, wait_task, close_task, list_tasks,
  send_input, send_message, followup_task, resume_agent.

Architecture
------------

.. code-block::

    spawn_task("db_query", goal="...")
      → creates independent TyAgent with _agent_loop
      → registers in parent._task_tree as /root/main/db_query
      → injects initial goal into child's inbox
      → returns task_path immediately

    Parent ← send_message / followup_task / send_input → Child
      via cross-mailbox messages (InterAgentMessage)

    Child completes
      → sends FinalNotification to parent._mailbox
      → parent._agent_loop drains mailbox → injects into _messages
      → parent._agent_loop optionally triggers a new turn
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tyagent.agent import TyAgent
from tyagent.types import InboxMessage
from tyagent.subagent.mailbox import (
    FinalNotification,
    InterAgentMessage,
    Mailbox,
)
from tyagent.subagent.task_tree import TaskTree
from tyagent.tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DEFAULT_SUBAGENT_MAX_TOOL_TURNS = 30
DEFAULT_CHILD_TIMEOUT = 600.0
MAX_CONCURRENT_SUBAGENTS = 5  # hard cap, aligns with Codex's default
ROOT_PATH = "/root"

# Sub-agents can spawn their own sub-agents (nested spawn is allowed).
# Only memory is blocked for sub-agents (should not write cross-session).
CHILD_BLOCKED_TOOLS = frozenset(["memory"])


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _child_tool_names(toolsets: Optional[List[str]] = None) -> List[str]:
    """Return tool names that a child agent is allowed to use."""
    all_names = registry.get_all_names()
    allowed = [n for n in all_names if n not in CHILD_BLOCKED_TOOLS]
    if toolsets:
        allowed = [n for n in toolsets if n in allowed]
    return allowed


def _build_parent_compression(parent: TyAgent) -> Any:
    """Cloneable compression config snapshot for child agents."""
    return getattr(parent, "_compression_config", None)


async def _launch_child_agent(
    *,
    task_path: str,
    goal: str,
    context: Optional[str],
    parent_agent: TyAgent,
    toolsets: Optional[List[str]],
    max_tool_turns: int,
) -> None:
    """Create and start a child agent running its own ``_agent_loop()``.

    The child runs indefinitely until explicitly stopped (``close_task``)
    or the parent shuts down.  After each turn the child notifies the
    parent via ``InterAgentMessage``; on exit it sends a
    ``FinalNotification``.
    """
    # ── Build system prompt with sub-agent identity ──────────────
    parent_path = parent_agent._task_path or ROOT_PATH
    child_system = parent_agent.system_prompt
    identity = (
        f"\n\n## Sub-agent identity\n"
        f"You are a sub-agent spawned by [{parent_path}].\n"
        f"Your task path: {task_path}\n"
        f"You can spawn your own sub-agents (nested spawn is allowed).\n"
        f"When you finish a task, notify your parent via the mailbox "
        f"(this happens automatically after each turn).\n"
        f"Your parent can send you follow-up tasks while you are running.\n"
        f"Use list_tasks to see your own sub-agents."
    )
    child_system = child_system + identity
    if context:
        child_system = f"{child_system}\n\nTask context: {context}"

    # ── Create child agent ───────────────────────────────────
    child = TyAgent(
        model=parent_agent.model,
        api_key=parent_agent.api_key,
        base_url=parent_agent.base_url,
        max_tool_turns=max_tool_turns,
        system_prompt=child_system,
        reasoning_effort=parent_agent.reasoning_effort,
        compression=_build_parent_compression(parent_agent),
        home_dir=parent_agent.home_dir,
        context_length=parent_agent.context_length,
        http_timeout=getattr(parent_agent, "_http_timeout", 120.0),
        shutdown_timeout=getattr(parent_agent, "_shutdown_timeout", 5.0),
    )

    # ── Child-specific configuration ──────────────────────────
    child._task_path = task_path
    child._parent_mailbox = parent_agent._mailbox
    child._child_mode = True
    child._check_inbox_between_turns = True  # receive send_input
    child._task_tree = TaskTree("root")
    child._allowed_tool_names = _child_tool_names(toolsets)

    # Register child in parent's child_agents dict (for send_input lookup)
    parent_agent._child_agents[task_path] = child

    # Set up tool progress callback forwarding
    parent_cb = getattr(parent_agent, "_tool_progress_callback", None)
    if parent_cb is not None:
        def _child_progress(tn: str, ai: dict) -> None:
            parent_cb(tn, ai, prefix="📤 ")
        child._tool_progress_callback = _child_progress

    # ── Inject initial goal and start permanent agent loop ───
    child._running = True
    child._stop_event.clear()
    await child._inbox.put(InboxMessage(
        text=goal,
        reply_target=None,
        tool_progress_cb=None,
        turn_done_cb=None,
    ))

    loop_task = asyncio.create_task(_child_agent_loop(
        child=child, task_path=task_path, parent_agent=parent_agent,
    ))
    parent_agent._bg_tasks[task_path] = loop_task
    child._loop_task = loop_task

    logger.info(
        "spawn_task: launched %s goal=%r",
        task_path, goal[:80],
    )


async def _child_agent_loop(
    child: TyAgent,
    task_path: str,
    parent_agent: TyAgent,
) -> None:
    """Run child._agent_loop() with cleanup on exit.

    The child's own ``_agent_loop()`` handles turn-by-turn notification
    and final notification.  This wrapper handles cleanup (close, unregister).
    """
    try:
        await child._agent_loop()
    finally:
        try:
            await child.close()
        except Exception:
            pass
        parent_agent._child_agents.pop(task_path, None)
        parent_agent._bg_tasks.pop(task_path, None)
        if parent_agent._task_tree is not None:
            parent_agent._task_tree.set_status(task_path, "shutdown")


# ═══════════════════════════════════════════════════════════════
# Tool Handlers
# ═══════════════════════════════════════════════════════════════


async def _handle_spawn_task(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Spawn a child agent.  Returns the canonical task_path immediately."""
    task_name = args.get("task_name", "").strip()
    goal = args.get("goal", "").strip()

    if not task_name:
        return tool_error("task_name is required for spawn_task.")
    if not goal:
        return tool_error("goal is required for spawn_task.")
    if parent_agent is None:
        return tool_error("spawn_task requires a session agent.")

    # ── Enforce max concurrent limit ──────────────────────────
    if parent_agent._task_tree is not None:
        running = sum(
            1 for p in parent_agent._task_tree.all_paths()
            if p != ROOT_PATH and parent_agent._task_tree.path_status(p) == "running"
        )
        if running >= MAX_CONCURRENT_SUBAGENTS:
            return tool_error(
                f"Cannot spawn: {running} agents already running "
                f"(max {MAX_CONCURRENT_SUBAGENTS}). "
                f"Close completed agents with close_task first."
            )

    # ── Resolve task path ────────────────────────────────────
    if parent_agent._task_tree is None:
        parent_agent._task_tree = TaskTree()
    parent_path = parent_agent._task_path or ROOT_PATH

    try:
        task_path = parent_agent._task_tree.register(
            parent_path=parent_path,
            task_name=task_name,
            agent=None,  # agent will be set in _launch_child_agent
        )
    except ValueError as exc:
        return tool_error(str(exc))

    context = args.get("context") or None
    toolsets: Optional[List[str]] = args.get("toolsets") or None
    fork_turns = args.get("fork_turns", "none")
    max_tool_turns = args.get("max_tool_turns", DEFAULT_SUBAGENT_MAX_TOOL_TURNS)

    try:
        max_tool_turns = int(max_tool_turns)
    except (TypeError, ValueError):
        return tool_error("max_tool_turns must be an integer.")
    if max_tool_turns < 1:
        return tool_error("max_tool_turns must be at least 1.")
    if max_tool_turns > 200:
        return tool_error("max_tool_turns must be at most 200.")

    # ── Fork context (future: actually fork history) ──────────
    if fork_turns not in ("none", "all"):
        try:
            int(fork_turns)
        except ValueError:
            return tool_error(
                "fork_turns must be 'none', 'all', or a number."
            )

    # ── Launch child ──────────────────────────────────────────
    await _launch_child_agent(
        task_path=task_path,
        goal=goal,
        context=context,
        parent_agent=parent_agent,
        toolsets=toolsets,
        max_tool_turns=max_tool_turns,
    )

    return json.dumps(
        {"task_path": task_path, "status": "running"},
        ensure_ascii=False,
    )


async def _handle_wait_task(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Wait for any mailbox update — Codex v2 semantics.

    No *targets* parameter.  Blocks until a mailbox message arrives
    (or timeout).  Returns a summary of what arrived.
    """
    if parent_agent is None:
        return tool_error("wait_task requires a session agent.")

    timeout = args.get("timeout", 300)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        return tool_error("timeout must be a number.")
    if timeout <= 0:
        return tool_error("timeout must be positive.")

    mailbox = parent_agent._mailbox

    # Check if messages are already available
    if mailbox.peek():
        items = mailbox.drain()
    else:
        arrived = await mailbox.wait_next(timeout=timeout)
        if not arrived:
            return json.dumps(
                {"message": "Timed out waiting for mailbox update.", "timed_out": True},
                ensure_ascii=False,
            )
        items = mailbox.drain()

    # Build summary
    completions = []
    messages = []
    for item in items:
        if isinstance(item, FinalNotification):
            completions.append({
                "task_path": item.task_path,
                "success": item.success,
                "summary": item.summary,
                "error": item.error,
                "duration_seconds": item.duration_seconds,
            })
        else:
            messages.append({
                "author": item.author,
                "content": item.content,
            })

    result: Dict[str, Any] = {"timed_out": False}
    if completions:
        result["completions"] = completions
    if messages:
        result["messages"] = messages

    return json.dumps(result, ensure_ascii=False)


async def _handle_close_task(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Close a running child agent and all its descendants."""
    target = args.get("target", "").strip()
    if not target:
        return tool_error("target is required for close_task.")
    if parent_agent is None:
        return tool_error("close_task requires a session agent.")

    # ── Resolve the target ─────────────────────────────────────
    tree = parent_agent._task_tree
    if tree is None:
        return json.dumps(
            {"success": False, "error": f"Task not found: {target}"}
        )

    resolved = tree.resolve(parent_agent._task_path or ROOT_PATH, target)
    if resolved is None:
        return json.dumps(
            {"success": False, "error": f"Task not found: {target}"}
        )

    previous_status = tree.path_status(resolved) or "not_found"

    # ── Cascade-close descendants ──────────────────────────────
    descendant_paths = tree.descendants(resolved)
    closed_count = 0
    for path in descendant_paths:
        child_agent = parent_agent._child_agents.get(path)
        bg_task = parent_agent._bg_tasks.get(path)

        if child_agent is not None:
            # Graceful stop: sets _stop_event, _agent_loop exits
            # naturally, _child_agent_loop finally block runs cleanup
            try:
                await child_agent.stop()
                closed_count += 1
            except Exception:
                pass
        elif bg_task is not None and not bg_task.done():
            # Fallback if agent already popped but task still running
            bg_task.cancel()
            closed_count += 1

    # ── Unregister from task_tree ─────────────────────────────
    tree.unregister(resolved)

    return json.dumps(
        {
            "previous_status": previous_status,
            "closed_count": closed_count,
        },
        ensure_ascii=False,
    )


async def _handle_list_tasks(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """List all spawned agents with their status and task paths."""
    if parent_agent is None:
        return tool_error("list_tasks requires a session agent.")

    path_prefix = (args.get("path_prefix") or "").strip()
    tree = parent_agent._task_tree

    if tree is None:
        return json.dumps({"agents": []}, ensure_ascii=False)

    if path_prefix:
        paths = tree.filter_by_prefix(path_prefix)
    else:
        paths = tree.all_paths()
        # Exclude root itself
        paths = [p for p in paths if p != ROOT_PATH]

    agents = []
    for path in paths:
        status = tree.path_status(path) or "unknown"
        agents.append({
            "agent_name": path,
            "agent_status": status,
        })

    return json.dumps({"agents": agents}, ensure_ascii=False)


async def _handle_send_input(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Send a message to a running child agent, optionally interrupting."""
    target = args.get("target", "").strip()
    message = args.get("message", "").strip()
    interrupt = args.get("interrupt", False)

    if not target:
        return tool_error("target is required for send_input.")
    if not message:
        return tool_error("message is required for send_input.")
    if parent_agent is None:
        return tool_error("send_input requires a session agent.")

    if not isinstance(interrupt, bool):
        return tool_error("interrupt must be a boolean.")

    # ── Resolve ────────────────────────────────────────────────
    tree = parent_agent._task_tree
    caller_path = parent_agent._task_path or ROOT_PATH
    resolved = tree.resolve(caller_path, target) if tree else None

    if resolved is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not found: {target}",
        })

    child = parent_agent._child_agents.get(resolved)
    if child is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not running: {resolved}",
        })

    # ── Send ───────────────────────────────────────────────────
    prefix = f"（来自主代理[{caller_path}]的指导）"
    await child.send_message(f"{prefix}{message}")
    status = "interrupted" if interrupt else "queued"

    logger.info(
        "send_input: %s → %s (%s, %d chars)",
        status, resolved, len(message),
    )

    return json.dumps({
        "success": True,
        "target": resolved,
        "status": status,
    }, ensure_ascii=False)


async def _handle_send_message(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Send a message WITHOUT triggering a new turn in the target."""
    target = args.get("target", "").strip()
    message = args.get("message", "").strip()

    if not target:
        return tool_error("target is required for send_message.")
    if not message:
        return tool_error("message is required for send_message.")
    if parent_agent is None:
        return tool_error("send_message requires a session agent.")

    tree = parent_agent._task_tree
    caller_path = parent_agent._task_path or ROOT_PATH
    resolved = tree.resolve(caller_path, target) if tree else None

    if resolved is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not found: {target}",
        })

    child = parent_agent._child_agents.get(resolved)
    if child is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not running: {resolved}",
        })

    # Queue without triggering a turn
    prefix = f"[{caller_path}]"
    await child.send_message(f"{prefix} {message}")

    return json.dumps({
        "success": True,
        "target": resolved,
        "status": "delivered",
    }, ensure_ascii=False)


async def _handle_followup_task(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Send a message AND trigger a new turn in the target agent."""
    target = args.get("target", "").strip()
    message = args.get("message", "").strip()

    if not target:
        return tool_error("target is required for followup_task.")
    if not message:
        return tool_error("message is required for followup_task.")
    if parent_agent is None:
        return tool_error("followup_task requires a session agent.")

    tree = parent_agent._task_tree
    caller_path = parent_agent._task_path or ROOT_PATH
    resolved = tree.resolve(caller_path, target) if tree else None

    if resolved is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not found: {target}",
        })

    child = parent_agent._child_agents.get(resolved)
    if child is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not running: {resolved}",
        })

    # Send to child's mailbox as InterAgentMessage with trigger_turn=True
    prefix = f"[{caller_path}] followup"
    if child._mailbox is not None:
        child._mailbox.send(InterAgentMessage(
            author=caller_path,
            recipient=resolved,
            content=f"{prefix} {message}",
            trigger_turn=True,
        ))
    else:
        # Fallback: send to inbox (old path)
        await child.send_message(f"{prefix} {message}")

    return json.dumps({
        "success": True,
        "target": resolved,
        "status": "followup_queued",
    }, ensure_ascii=False)


async def _handle_resume_agent(
    args: Dict[str, Any], parent_agent: Any = None
) -> str:
    """Re-open a previously closed agent so it can receive messages again.

    In the current in-memory model, this is a no-op — agents cannot be
    resumed after close_task.  Reserved for persistent thread support.
    """
    target = args.get("id", "").strip()
    if not target:
        return tool_error("id is required for resume_agent.")

    tree = parent_agent._task_tree if parent_agent else None
    if tree is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not found: {target}",
        })

    caller_path = parent_agent._task_path if parent_agent else ROOT_PATH
    resolved = tree.resolve(caller_path, target)
    if resolved is None:
        return json.dumps({
            "success": False,
            "error": f"Agent not found: {target}",
        })

    status = tree.path_status(resolved)
    if status not in ("completed", "shutdown"):
        return json.dumps({
            "success": False,
            "error": (
                f"Cannot resume agent '{resolved}' "
                f"(status: {status}). Only completed/shutdown "
                f"agents can be resumed."
            ),
        })

    # Currently a no-op — agents are in-memory and cannot be restored.
    return json.dumps({
        "success": False,
        "error": (
            f"resume_agent is not yet supported for '{resolved}'. "
            f"Re-spawn a new agent instead."
        ),
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# Tool Schemas (Codex v2 aligned)
# ═══════════════════════════════════════════════════════════════

SPAWN_TASK_SCHEMA: Dict[str, Any] = {
    "name": "spawn_task",
    "description": (
        "Spawn a sub-agent for a well-scoped task. "
        "Spawned agents inherit your model and tools, and can spawn "
        "their own sub-agents. "
        "Provide a descriptive task_name — it forms a hierarchical "
        "path. If your path is /root/task1 and you spawn with "
        "task_name='db_query', the agent's canonical path is "
        "/root/task1/db_query. "
        "You can reference it by relative name ('db_query') or "
        "absolute path ('/root/task1/db_query'). "
        "Returns the canonical task_path immediately — the child "
        "runs independently. "
        "Use when: the task is well-scoped, can run in parallel, "
        "and you don't want to wait for it inline. "
        "IMPORTANT: Do NOT spawn agents just to explore or read. "
        "Close agents with close_task when results are integrated. "
        "Hard limit: 5 concurrent agents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_name": {
                "type": "string",
                "description": (
                    "A descriptive name for this task. "
                    "Use lowercase letters, digits, and underscores. "
                    "Forms part of the hierarchical task path."
                ),
            },
            "goal": {
                "type": "string",
                "description": "What the child agent should accomplish.",
            },
            "context": {
                "type": "string",
                "description": "Optional background information.",
            },
            "fork_turns": {
                "type": "string",
                "description": (
                    "How many conversation turns to fork into the child. "
                    "'none' = fresh context (default). "
                    "'all' = full parent history. "
                    "A number like '3' = fork last 3 turns."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional restricted tool list.",
            },
            "max_tool_turns": {
                "type": "integer",
                "description": (
                    f"Max tool-call turns (default: "
                    f"{DEFAULT_SUBAGENT_MAX_TOOL_TURNS})."
                ),
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["task_name", "goal"],
    },
}

WAIT_TASK_SCHEMA: Dict[str, Any] = {
    "name": "wait_task",
    "description": (
        "Wait for a mailbox update from any live sub-agent. "
        "This includes mid-task progress messages and final completion "
        "notifications. "
        "Returns a summary of what arrived — does NOT return the full "
        "content. "
        "The full content is injected into your conversation "
        "automatically. "
        "Use when: you've spawned sub-agents and want to wait for "
        "results or progress updates before continuing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timeout": {
                "type": "number",
                "description": (
                    "Max seconds to wait (default: 300). "
                    "Prefer longer waits to avoid busy-polling."
                ),
            },
        },
    },
}

CLOSE_TASK_SCHEMA: Dict[str, Any] = {
    "name": "close_task",
    "description": (
        "Close a running child agent and all of its descendants. "
        "Returns the agent's previous status before shutdown. "
        "The *target* can be a relative task_name (e.g. 'db_query') "
        "or an absolute task_path (e.g. '/root/main/db_query'). "
        "Use when: a sub-agent is no longer needed or is producing "
        "irrelevant results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Agent task_name or task_path to close "
                    "(from spawn_task return value or list_tasks)."
                ),
            },
        },
        "required": ["target"],
    },
}

LIST_TASKS_SCHEMA: Dict[str, Any] = {
    "name": "list_tasks",
    "description": (
        "List all spawned sub-agents and their status. "
        "Optionally filter by task-path prefix. "
        "Each entry includes the canonical task_name and status "
        "(running, completed, error, shutdown)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path_prefix": {
                "type": "string",
                "description": (
                    "Optional task-path prefix to filter results. "
                    "e.g. '/root/main' to see only agents in that subtree."
                ),
            },
        },
    },
}

SEND_INPUT_SCHEMA: Dict[str, Any] = {
    "name": "send_input",
    "description": (
        "Send a message to a running child agent. "
        "When interrupt=true, the message is prioritised — the child "
        "will handle it before continuing its current work. "
        "When interrupt=false (default), the message is queued and "
        "handled between tool calls. "
        "Use when: you discover new information the child needs, "
        "the user refines requirements, or the child's current "
        "direction needs correction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Agent task_name or task_path to message."
                ),
            },
            "message": {
                "type": "string",
                "description": "Message text to send.",
            },
            "interrupt": {
                "type": "boolean",
                "description": (
                    "When true, the child handles this immediately. "
                    "When false (default), queued for next tool turn."
                ),
            },
        },
        "required": ["target", "message"],
    },
}

SEND_MESSAGE_SCHEMA: Dict[str, Any] = {
    "name": "send_message",
    "description": (
        "Send a message to a running child agent WITHOUT triggering "
        "a new turn. "
        "The message is delivered promptly but the child continues "
        "its current work — it will see the message on its next "
        "natural turn. "
        "Use when: you want to provide context to a child without "
        "interrupting its current flow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Agent task_name or task_path to message.",
            },
            "message": {
                "type": "string",
                "description": "Message text to send.",
            },
        },
        "required": ["target", "message"],
    },
}

FOLLOWUP_TASK_SCHEMA: Dict[str, Any] = {
    "name": "followup_task",
    "description": (
        "Send a message to a child agent AND trigger a new turn. "
        "If the child is mid-turn, the message is queued and will "
        "start the child's next turn after the current one completes. "
        "Use when: you want to give the child a new task after its "
        "current work finishes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Agent task_name or task_path to message.",
            },
            "message": {
                "type": "string",
                "description": "Message text to send.",
            },
        },
        "required": ["target", "message"],
    },
}

RESUME_AGENT_SCHEMA: Dict[str, Any] = {
    "name": "resume_agent",
    "description": (
        "Resume a previously closed agent by id so it can receive "
        "send_input and wait_task calls. "
        "Currently not supported (agents are ephemeral)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Agent task_path to resume.",
            },
        },
        "required": ["id"],
    },
}


# ═══════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════

registry.register(
    name="spawn_task",
    schema=SPAWN_TASK_SCHEMA,
    handler=_handle_spawn_task,
    description="Launch a child agent in background (non-blocking, returns task_path)",
    emoji="🚀",
    wants_parent=True,
)
registry.register(
    name="wait_task",
    schema=WAIT_TASK_SCHEMA,
    handler=_handle_wait_task,
    description="Wait for mailbox update from any live sub-agent",
    emoji="⏳",
    wants_parent=True,
)
registry.register(
    name="close_task",
    schema=CLOSE_TASK_SCHEMA,
    handler=_handle_close_task,
    description="Close a running child agent and its descendants",
    emoji="🛑",
    wants_parent=True,
)
registry.register(
    name="list_tasks",
    schema=LIST_TASKS_SCHEMA,
    handler=_handle_list_tasks,
    description="List all spawned agents with task_path and status",
    emoji="📋",
    wants_parent=True,
)
registry.register(
    name="send_input",
    schema=SEND_INPUT_SCHEMA,
    handler=_handle_send_input,
    description="Send message to child agent (optionally interrupt)",
    emoji="📨",
    wants_parent=True,
)
registry.register(
    name="send_message",
    schema=SEND_MESSAGE_SCHEMA,
    handler=_handle_send_message,
    description="Send message to child agent without triggering turn",
    emoji="💬",
    wants_parent=True,
)
registry.register(
    name="followup_task",
    schema=FOLLOWUP_TASK_SCHEMA,
    handler=_handle_followup_task,
    description="Send message + trigger a new turn in child agent",
    emoji="🔄",
    wants_parent=True,
)
registry.register(
    name="resume_agent",
    schema=RESUME_AGENT_SCHEMA,
    handler=_handle_resume_agent,
    description="Resume a previously closed agent (not yet supported)",
    emoji="🔓",
    wants_parent=True,
)
