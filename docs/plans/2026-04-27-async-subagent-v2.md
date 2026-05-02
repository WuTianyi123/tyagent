# Async Subagent — Full Codex-Aligned Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace synchronous `delegate_task` with Codex-aligned async subagent architecture — `spawn_task` (non-blocking) + `wait_task` (blocking, multi-source) + auto-injection of completed child results into parent's LLM turn.

**Architecture:** Pure asyncio throughout. `chat()` becomes a persistent event loop that doesn't exit while subagents are running. A `EventCollector` bridges child completion events back into the parent conversation. `wait_task` uses `asyncio.wait()` for multi-source waiting. No ThreadPoolExecutor — children are `asyncio.Task` instances sharing the parent's event loop.

**Dependencies:** Python 3.11+, asyncio, pytest, unittest.mock

---

### Architecture Overview

```
User message enters gateway
  │
  ▼
agent.chat(messages, tools)  ← persistent loop
  │
  ├─▶ EventCollector.drain_completed()
  │     injects child results as user messages
  │
  ├─▶ LLM API call
  │     │
  │     ├── spawn_task  → asyncio.create_task(_run_child_async)
  │     │                  → child completes → collector.notify_child_done()
  │     │                  → next loop iteration injects result automatically
  │     │
  │     ├── wait_task   → asyncio.wait([futures], timeout=timeout)
  │     │                  → collects completed results
  │     │
  │     └── sync tools  → run_in_executor(registry.dispatch)
  │
  ├─▶ Model produced final text
  │     │
  │     ├── No subagents running + no pending events → return ✅
  │     │
  │     └── Subagents still running → await collector.wait_next()
  │           → child completes → inject → re-enter LLM
  │
  └─▶ (loop continues)
```

**Key invariants:**
- Every LLM turn starts with a drain of the collector — completed children always get injected before the model sees the next prompt
- chat() only returns when ALL subagents are done and no pending events exist
- The model can spawn, do other work, call wait_task, or do nothing — the loop handles all patterns

---

### Task 1: Create EventCollector

**Objective:** Bridge between child agent completion (async task) and parent agent's chat() loop.

**Files:**
- Create: `tyagent/events.py`

**Step 1: Write the class**

```python
"""Event collector bridging subagent completion to parent agent loop."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional


class EventCollector:
    """Collects events from subagents for the parent chat() loop to consume.

    Pattern:
      1. Child completes → calls notify_child_done(task_id, result)
      2. Chat() loop → calls drain_completed() before each LLM turn
      3. If model exits while children run → call wait_next() to block
    """

    def __init__(self):
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._event = asyncio.Event()

    def notify_child_done(self, task_id: str, result: Dict[str, Any]) -> None:
        """Called by _run_child_async when a child agent completes."""
        self._completed[task_id] = result
        self._event.set()

    def drain_completed(self) -> List[Dict[str, Any]]:
        """Return a list of completed child events and clear buffer.

        Each event dict:
          {"type": "child_complete", "task_id": str, "result": {...}}
        """
        events = [
            {"type": "child_complete", "task_id": tid, "result": result}
            for tid, result in self._completed.items()
        ]
        self._completed.clear()
        self._event.clear()
        return events

    def peek(self) -> bool:
        """Check if events are available without consuming."""
        return bool(self._completed)

    async def wait_next(self, timeout: Optional[float] = None) -> bool:
        """Block until the next event arrives (or timeout).

        Returns True if an event arrived, False on timeout.
        """
        if self._completed:
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
```

**Step 2: Write tests**

```python
# tests/test_events.py

import pytest
from tyagent.events import EventCollector


class TestEventCollector:
    async def test_empty_drain(self):
        c = EventCollector()
        assert c.drain_completed() == []
        assert c.peek() is False

    async def test_notify_and_drain(self):
        c = EventCollector()
        c.notify_child_done("abc", {"success": True, "summary": "done"})
        assert c.peek() is True
        events = c.drain_completed()
        assert len(events) == 1
        assert events[0]["task_id"] == "abc"
        assert events[0]["type"] == "child_complete"
        assert c.peek() is False  # cleared

    async def test_drain_clears_previous(self):
        c = EventCollector()
        c.notify_child_done("a", {"summary": "first"})
        c.drain_completed()
        assert c.peek() is False

    async def test_multiple_children(self):
        c = EventCollector()
        c.notify_child_done("a", {})
        c.notify_child_done("b", {})
        events = c.drain_completed()
        assert len(events) == 2
        task_ids = {e["task_id"] for e in events}
        assert task_ids == {"a", "b"}

    async def test_wait_next_returns_immediately_if_event_exists(self):
        c = EventCollector()
        c.notify_child_done("x", {})
        result = await c.wait_next(timeout=1.0)
        assert result is True

    async def test_wait_next_timeout(self):
        c = EventCollector()
        result = await c.wait_next(timeout=0.05)
        assert result is False

    async def test_wait_next_notified(self):
        c = EventCollector()

        async def notify_later():
            await asyncio.sleep(0.05)
            c.notify_child_done("x", {})

        async def waiter():
            return await c.wait_next(timeout=5.0)

        results = await asyncio.gather(waiter(), notify_later())
        assert results[0] is True
```

**Step 3: Commit**

```bash
git add tyagent/events.py tests/test_events.py
git commit -m "feat: add EventCollector for bridging child completion to parent loop"
```

---

### Task 2: Update TyAgent for async subagent lifecycle

**Objective:** Add `_bg_tasks` (asyncio.Task dict) and `_event_collector` to TyAgent. Remove ThreadPoolExecutor.

**Files:**
- Modify: `tyagent/agent.py` — `__init__`, `close`
- Modify: `tyagent/events.py` — no changes, just import in agent.py

**Step 1: Update `__init__`**

```python
# In __init__, add after self._system_msg:
self._bg_tasks: Dict[str, asyncio.Task] = {}
self._event_collector: Optional[EventCollector] = None
```

Import at top:
```python
from tyagent.events import EventCollector
```

**Step 2: Update `close()`**

```python
async def close(self) -> None:
    # Cancel any running background tasks
    for tid, task in list(self._bg_tasks.items()):
        if not task.done():
            task.cancel()
    if self._bg_tasks:
        await asyncio.gather(*self._bg_tasks.values(), return_exceptions=True)
    self._bg_tasks.clear()
    self._event_collector = None
    await self._client.aclose()
```

**Step 3: Write tests**

```python
# tests/test_agent.py

class TestAsyncSubagentLifecycle:
    async def test_bg_tasks_initially_empty(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        assert agent._bg_tasks == {}
        assert agent._event_collector is None
        await agent.close()

    async def test_close_cancels_running_tasks(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        async def never_finish():
            await asyncio.Event().wait()
        task = asyncio.create_task(never_finish())
        agent._bg_tasks["abc"] = task
        await agent.close()
        assert task.cancelled()
```

**Step 4: Commit**

```bash
git add tyagent/agent.py tests/test_agent.py
git commit -m "feat: add async subagent lifecycle to TyAgent"
```

---

### Task 3: Implement _run_child_async and spawn_task

**Objective:** Child agent as an asyncio task. `spawn_task` creates the task and returns a `task_id` immediately.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Add `_run_child_async` (async replacement for `_run_child_sync`)**

```python
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
) -> Dict[str, Any]:
    """Run a child agent as an asyncio task.

    On completion, notifies the collector.  Never raises — errors
    are captured in the result dict.
    """
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

    try:
        summary = await asyncio.wait_for(
            child.chat(
                child_messages,
                tools=tool_defs,
                tool_progress_callback=tool_progress_callback,
            ),
            timeout=600.0,
        )
        result: Dict[str, Any] = {
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
        try:
            await child.close()
        except Exception:
            pass

    # Notify parent's event collector
    if collector is not None:
        collector.notify_child_done(task_id, result)

    return result
```

**Step 2: Add `_handle_spawn_task` handler**

```python
import uuid

def _handle_spawn_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Spawn a child agent in the background (asyncio task). Returns task_id immediately.

    The child is added to parent_agent._bg_tasks for wait_task to track,
    and its completion triggers parent_agent._event_collector for auto-injection.
    """
    goal = args.get("goal", "").strip()
    if not goal:
        return tool_error("goal is required for spawn_task.")

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

    if parent_agent is None:
        return tool_error("No parent agent context — spawn_task requires a session agent.")

    allowed = _build_allowed_tools(toolsets)

    # Build child progress callback
    parent_cb = getattr(parent_agent, "_tool_progress_callback", None)
    child_cb = None
    if parent_cb is not None:
        def _child_progress(tool_name: str, args_inner: dict) -> None:
            parent_cb(tool_name, args_inner, prefix="📤 ")
        child_cb = _child_progress

    task_id = str(uuid.uuid4())[:8]

    # Ensure event collector exists
    if parent_agent._event_collector is None:
        parent_agent._event_collector = EventCollector()

    # Create asyncio task for the child
    child_coro = _run_child_async(
        task_id=task_id,
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
        tool_progress_callback=child_cb,
        collector=parent_agent._event_collector,
    )

    loop = asyncio.get_event_loop()
    child_task = loop.create_task(child_coro)
    parent_agent._bg_tasks[task_id] = child_task

    logger.info("spawn_task: launched %s goal=%r", task_id, goal[:80])
    return json.dumps({"task_id": task_id, "status": "running"}, ensure_ascii=False)
```

Note: The handler is called from `run_in_executor` (sync), but it needs to submit an async task. `loop.create_task()` or `asyncio.ensure_future()` works from any thread as long as the loop is running. Use `asyncio.get_event_loop()` which returns the running loop from thread context.

**Step 3: Extract `_build_allowed_tools` (shared between spawn and delegate)**

```python
def _build_allowed_tools(toolsets: Optional[List[str]] = None) -> List[str]:
    """Return tool names allowed for a child agent."""
    all_names = registry.get_all_names()
    allowed = [n for n in all_names if n not in DELEGATE_BLOCKED_TOOLS]
    if toolsets:
        allowed = [n for n in toolsets if n in allowed]
    return allowed
```

**Step 4: Register spawn_task**

```python
SPAWN_TASK_SCHEMA: Dict[str, Any] = {
    "name": "spawn_task",
    "description": (
        "Launch a child agent to work on a task in the background. "
        "Returns immediately with a task_id — the child runs independently. "
        "You can continue working and call wait_task later to collect results.\n\n"
        "Use this when you want to parallelize: spawn multiple children, "
        "do other work, then wait for all results at once.\n\n"
        "Tip: If a child completes before you call wait_task, its result "
        "will be automatically injected into the conversation."
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

registry.register(
    name="spawn_task",
    schema=SPAWN_TASK_SCHEMA,
    handler=_handle_spawn_task,
    description="Launch a child agent in background (non-blocking, Codex-style)",
    emoji="🚀",
    wants_parent=True,
)
```

**Step 5: Write tests**

```python
class TestSpawnTask:
    def test_returns_task_id_immediately(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_async"):
            result = json.loads(_handle_spawn_task({"goal": "test"}, parent_agent=agent))
        assert "task_id" in result
        assert result["status"] == "running"

    def test_task_stored_in_bg_tasks(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_async"):
            result = json.loads(_handle_spawn_task({"goal": "test"}, parent_agent=agent))
        assert result["task_id"] in agent._bg_tasks

    def test_collector_created_lazily(self):
        agent = _make_agent()
        assert agent._event_collector is None
        with patch.object(delegate_tool, "_run_child_async"):
            _handle_spawn_task({"goal": "test"}, parent_agent=agent)
        assert agent._event_collector is not None

    def test_no_parent_agent_errors(self):
        result = json.loads(_handle_spawn_task({"goal": "test"}))
        assert "error" in result

    def test_spawn_task_blocked_from_children(self):
        assert "spawn_task" in DELEGATE_BLOCKED_TOOLS

    def test_validation_goal_required(self):
        agent = _make_agent()
        r = json.loads(_handle_spawn_task({}, parent_agent=agent))
        assert "error" in r

    def test_validation_max_tool_turns(self):
        agent = _make_agent()
        for bad in ["abc", 0, -1, 999]:
            r = json.loads(_handle_spawn_task({"goal": "test", "max_tool_turns": bad}, parent_agent=agent))
            assert "error" in r
```

**Step 6: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "feat: add spawn_task tool with async child lifecycle"
```

---

### Task 4: Implement wait_task handler

**Objective:** Block until specified children complete, return aggregated results. Uses `asyncio.wait()` with timeout.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Add `_handle_wait_task` handler**

```python
def _handle_wait_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Wait for spawned child agents and return their aggregated results.

    Uses asyncio.wait() internally for multi-source waiting.
    Returns results keyed by task_id.
    """
    task_ids = args.get("task_ids", [])
    if not task_ids:
        return tool_error("task_ids is required for wait_task (list of task IDs).")
    if not isinstance(task_ids, list):
        return tool_error("task_ids must be a list of task ID strings.")

    timeout = args.get("timeout", 300)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        return tool_error("timeout must be a number (seconds).")
    if timeout <= 0:
        return tool_error("timeout must be positive.")

    if parent_agent is None:
        return tool_error("No parent agent context — wait_task requires a session agent.")

    # Match task IDs to futures
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

    results: Dict[str, Dict[str, Any]] = {}

    for tid in missing:
        results[tid] = {"error": f"Task not found: {tid}"}

    if futures_map:
        # asyncio.wait in a separate async context
        # Since handler runs in run_in_executor (thread), we need the event loop
        async def _wait_all():
            done, pending = await asyncio.wait(
                list(futures_map.values()),
                timeout=timeout,
            )
            done_set = set(done)
            for tid, fut in futures_map.items():
                if fut in done_set:
                    try:
                        results[tid] = fut.result()  # Already done, no await needed
                        parent_agent._bg_tasks.pop(tid, None)
                    except asyncio.CancelledError:
                        results[tid] = {"error": "Task was cancelled"}
                        parent_agent._bg_tasks.pop(tid, None)
                    except Exception as exc:
                        results[tid] = {"error": f"{type(exc).__name__}: {exc}"}
                        parent_agent._bg_tasks.pop(tid, None)
                else:
                    results[tid] = {"error": f"Timed out after {timeout}s"}

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_wait_all())

    logger.info("wait_task: collected %d results for %d task(s)",
                len([r for r in results.values() if "error" not in r]), len(task_ids))
    return json.dumps(results, ensure_ascii=False)
```

**Step 2: Register wait_task**

```python
WAIT_TASK_SCHEMA: Dict[str, Any] = {
    "name": "wait_task",
    "description": (
        "Wait for one or more spawned child agents to complete. "
        "Returns aggregated results keyed by task_id.\n\n"
        "If a task has already completed, returns it immediately. "
        "If a task is still running, blocks for up to `timeout` seconds. "
        "Uses asyncio.wait internally — multiple children are waited on simultaneously."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of task IDs returned by spawn_task.",
            },
            "timeout": {
                "type": "number",
                "description": "Max seconds to wait (default: 300).",
            },
        },
        "required": ["task_ids"],
    },
}

registry.register(
    name="wait_task",
    schema=WAIT_TASK_SCHEMA,
    handler=_handle_wait_task,
    description="Wait for background child agents to complete (blocking)",
    emoji="⏳",
    wants_parent=True,
)
```

**Step 3: Write tests**

```python
class TestWaitTask:
    def test_returns_collected_results(self):
        agent = _make_agent()
        async def mock_child():
            return {"success": True, "summary": "done"}
        task = asyncio.ensure_future(mock_child())
        await task  # complete it
        agent._bg_tasks["abc123"] = task

        result = json.loads(_handle_wait_task({"task_ids": ["abc123"]}, parent_agent=agent))
        assert "abc123" in result
        assert result["abc123"]["summary"] == "done"

    def test_missing_task_ids_errors(self):
        agent = _make_agent()
        result = json.loads(_handle_wait_task({}, parent_agent=agent))
        assert "error" in result

    def test_task_not_found_error(self):
        agent = _make_agent()
        result = json.loads(_handle_wait_task({"task_ids": ["nonexistent"]}, parent_agent=agent))
        assert "error" in result["nonexistent"]

    def test_timeout_positive_required(self):
        agent = _make_agent()
        for bad in [0, -1]:
            r = json.loads(_handle_wait_task({"task_ids": ["x"], "timeout": bad}, parent_agent=agent))
            assert "error" in r

    def test_task_popped_after_collection(self):
        agent = _make_agent()
        async def ok(): return {"summary": "ok"}
        task = asyncio.ensure_future(ok())
        await task
        agent._bg_tasks["x"] = task
        json.loads(_handle_wait_task({"task_ids": ["x"]}, parent_agent=agent))
        assert "x" not in agent._bg_tasks

    def test_cancelled_task_reported(self):
        agent = _make_agent()
        async def cancellable():
            await asyncio.sleep(999)
        task = asyncio.ensure_future(cancellable())
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        agent._bg_tasks["x"] = task
        result = json.loads(_handle_wait_task({"task_ids": ["x"]}, parent_agent=agent))
        assert "cancelled" in result["x"]["error"].lower()
```

**Step 4: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "feat: add wait_task tool with asyncio.wait multi-source blocking"
```

---

### Task 5: Update chat() loop — auto-injection + persistent loop

**Objective:** The core change. Before each LLM turn, drain EventCollector and inject child results as user messages. After the model produces final text, continue looping if children are still running.

**Files:**
- Modify: `tyagent/agent.py` — `chat()` method

**Step 1: Replace the tool-calling dispatch block (lines ~367-403)**

Replace this section in `chat()`:

```python
# --- Tool call execution (existing code, around line 367) ---
for tc in tool_calls:
    ...
```

With an **async-aware dispatch**:

```python
            # Execute tool calls and append results
            tool_turn += 1
            logger.info("Tool turn %d/%s: executing %d tool call(s)",
                        tool_turn, self.max_tool_turns or "∞", len(tool_calls))

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_type = tc.get("type", "")
                func = tc.get("function", {}) or {}
                func_name = func.get("name", "")
                func_args_str = func.get("arguments", "")

                if tc_type != "function" or not func_name:
                    tool_result = json.dumps({"error": "Malformed tool call"})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})
                    if on_message:
                        on_message("tool", tool_result, tool_call_id=tc_id)
                    continue

                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except json.JSONDecodeError:
                    tool_result = json.dumps({"error": f"Invalid JSON arguments: {func_args_str}"})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})
                    if on_message:
                        on_message("tool", tool_result, tool_call_id=tc_id)
                    continue

                logger.info("  ⚡ %s(%s)", func_name, ", ".join(f"{k}={v!r}" for k, v in list(func_args.items())[:3]))
                if tool_progress_callback:
                    try:
                        tool_progress_callback(func_name, func_args)
                    except Exception as exc:
                        logger.warning("tool_progress_callback failed: %s", exc)

                # ── Async-aware dispatch ──────────────────────────────
                # spawn_task and wait_task run directly in the event loop.
                # All other tools run via run_in_executor (sync).
                if func_name in ("spawn_task", "wait_task"):
                    # These handlers will use asyncio.get_event_loop() internally
                    # to create tasks or wait on futures.
                    result = await loop.run_in_executor(
                        None, registry.dispatch, func_name, func_args, self,
                    )
                else:
                    result = await loop.run_in_executor(
                        None, registry.dispatch, func_name, func_args, self,
                    )
                # ── End async-aware dispatch ───────────────────────────

                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                if on_message:
                    on_message("tool", result, tool_call_id=tc_id)

```

Note: For this task, `spawn_task` and `wait_task` still go through `run_in_executor`. The handlers themselves (`_handle_spawn_task`, `_handle_wait_task`) call `asyncio.get_event_loop()` internally to bridge the thread→async gap. This minimizes changes to the chat() loop while enabling full async behavior.

In a future optimization, we could register `async_handler` on the registry and call `await entry.async_handler(...)` directly, but for MVP the `run_in_executor` → `get_event_loop()` bridge is clean enough.

**Step 2: Add auto-injection before the LLM call**

After the `messages.insert(0, system_msg)` block and before `api_messages = list(messages)`:

```python
        # ── Auto-inject completed subagent results ──────────────
        # Before each LLM turn, drain the event collector and inject
        # completed child results as user messages.
        if self._event_collector is not None:
            child_events = self._event_collector.drain_completed()
            for event in child_events:
                task_id = event["task_id"]
                result = event["result"]
                # Build a descriptive user message from the child result
                summary = result.get("summary", "")
                if summary:
                    inject_content = f"## Subagent `{task_id}` completed\n\n{summary}"
                elif result.get("success"):
                    inject_content = f"## Subagent `{task_id}` completed successfully"
                else:
                    inject_content = (
                        f"## Subagent `{task_id}` failed\n\n"
                        f"Error: {result.get('error', 'Unknown error')}"
                    )
                messages.append({"role": "user", "content": inject_content})
                logger.info("Auto-injected child result for %s into conversation", task_id)
```

This block should be placed **once, before the while True loop** and **also at the top of the while True loop body**, so it fires before every LLM turn (including the first one on subsequent iterations after tool execution).

**Step 3: Add persistent loop — don't exit while children run**

In the final return section (around line 352-353 in the current code):

```python
            # No tool calls -> final answer
            if not tool_calls:
                content_text = (content or reasoning_content or "")

                # ── Persistent loop: wait for running children ────────
                # If children are still running, don't exit — wait for
                # the next child to complete and inject its result.
                if self._bg_tasks:
                    logger.info(
                        "Model finished but %d subagent(s) running — waiting for next event",
                        len(self._bg_tasks),
                    )
                    if self._event_collector is None:
                        self._event_collector = EventCollector()

                    # Wait for next child completion (max timeout bound)
                    arrived = await self._event_collector.wait_next(
                        timeout=SUBAGENT_PERSIST_TIMEOUT,
                    )
                    if arrived:
                        # Inject child result and let model respond
                        continue
                    else:
                        logger.warning(
                            "Subagent persist timeout (%ds) — returning current answer",
                            SUBAGENT_PERSIST_TIMEOUT,
                        )
                        return content_text

                return content_text
```

Add a constant:
```python
# Near the top of agent.py, in the constants area
SUBAGENT_PERSIST_TIMEOUT = 120  # Max seconds to wait for children after model finishes
```

**Step 4: Write tests**

```python
# tests/test_agent_async_subagent.py

class TestChatAutoInject:
    @patch("tyagent.tools.registry.get_definitions", return_value=[])
    async def test_injects_completed_child_result(self, mock_defs):
        """Completed child result is injected as user message before next LLM call."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        agent._event_collector = EventCollector()
        agent._event_collector.notify_child_done("abc", {"success": True, "summary": "Search done"})

        messages = [{"role": "user", "content": "search"}]
        # chat() with mock API that returns empty (no tool calls)
        with patch.object(agent._client, "post", ...) as mock_post:
            # mock_post returns a response with streaming=False
            ...

        # Verify injected message appeared
        injected = [m for m in messages if "Subagent" in m.get("content", "")]
        assert len(injected) == 1
```

**Step 5: Commit**

```bash
git add tyagent/agent.py tests/test_agent_async_subagent.py
git commit -m "feat: auto-inject child results and persistent chat() loop"
```

---

### Task 6: Update delegate_task as spawn+wait backward compat wrapper

**Objective:** Keep `delegate_task` working exactly as before — internally spawns, immediately waits, flattens result.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Rewrite `_handle_delegate_task`**

```python
def _handle_delegate_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Convenience wrapper: spawn_task + immediate wait_task, single result.

    This is equivalent to the original blocking delegate_task behavior,
    kept for backward compatibility.
    """
    # Delegate to spawn (strips extraneous args)
    spawn_args = {k: v for k, v in args.items() if k not in ("task_ids", "timeout")}
    spawn_result = json.loads(_handle_spawn_task(spawn_args, parent_agent=parent_agent))
    if "error" in spawn_result:
        return json.dumps(spawn_result, ensure_ascii=False)

    task_id = spawn_result["task_id"]

    # Immediately wait
    wait_result = json.loads(
        _handle_wait_task({"task_ids": [task_id]}, parent_agent=parent_agent)
    )

    # Flatten: delegate_task returns one result dict, not {task_id: result}
    single = wait_result.get(task_id, {"error": "Unknown error"})
    return json.dumps(single, ensure_ascii=False)
```

**Step 2: Write regression tests (verify existing delegate_task tests still pass)**

The existing tests mock `_run_child_sync` — but delegate_task no longer calls `_run_child_sync`. The tests need to mock `_handle_spawn_task` and `_handle_wait_task` instead, or mock `_run_child_async`.

Update `test_delegate_tool.py`:

```python
class TestDelegateTaskBackwardCompat:
    def test_delegate_task_spawns_and_waits(self):
        """delegate_task returns flattened result from spawn+wait."""
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_async") as mock_run:
            mock_run.return_value = {"success": True, "summary": "hi", "error": None, "duration": 1}
            result = json.loads(_handle_delegate_task({"goal": "test"}, parent_agent=agent))
        assert result["success"] is True
        assert result["summary"] == "hi"

    def test_delegate_task_error_on_spawn_failure(self):
        agent = _make_agent()
        result = json.loads(_handle_delegate_task({"goal": ""}, parent_agent=agent))
        assert "error" in result
```

**Step 3: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "refactor: delegate_task as spawn+wait wrapper (backward compatible)"
```

---

### Task 7: Update DELEGATE_BLOCKED_TOOLS

**Objective:** Block spawn_task and wait_task from child agents (no recursive background spawning).

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Update blocked set**

```python
DELEGATE_BLOCKED_TOOLS = frozenset(
    ["delegate_task", "spawn_task", "wait_task", "memory"]
)
```

**Step 2: Update existing tests**

```python
def test_spawn_task_and_wait_task_are_blocked():
    assert "spawn_task" in DELEGATE_BLOCKED_TOOLS
    assert "wait_task" in DELEGATE_BLOCKED_TOOLS
```

**Step 3: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "fix: block spawn_task/wait_task from child agents"
```

---

### Task 8: End-to-end integration tests

**Objective:** Verify the full spawn → (interleave work) → wait flow works with mocked child agents.

**Files:**
- Modify: `tests/test_delegate_tool.py`

**Step 1: Write integration tests**

```python
class TestSpawnWaitIntegration:
    async def test_spawn_then_wait_gives_result(self):
        """spawn_task → wait_task returns child result."""
        agent = _make_agent()

        async def fast_child(*args, **kwargs):
            return {"success": True, "summary": "work done", "error": None, "duration": 0.1}

        with patch.object(delegate_tool, "_run_child_async", side_effect=fast_child):
            spawn_result = json.loads(
                _handle_spawn_task({"goal": "task"}, parent_agent=agent)
            )
            tid = spawn_result["task_id"]

            # Wait a moment for the child to complete
            await asyncio.sleep(0.1)

            wait_result = json.loads(
                _handle_wait_task({"task_ids": [tid]}, parent_agent=agent)
            )

        assert wait_result[tid]["success"] is True
        assert "work done" in wait_result[tid]["summary"]

    async def test_spawn_multiple_children_concurrently(self):
        """Multiple children run concurrently, wait_task collects all."""
        agent = _make_agent()
        counters = {}

        async def child_a(*a, **kw): return {"summary": "A done"}
        async def child_b(*a, **kw): return {"summary": "B done"}
        async def child_c(*a, **kw): return {"summary": "C done"}

        with patch.object(delegate_tool, "_run_child_async", side_effect=[child_a(), child_b(), child_c()]):
            t1 = json.loads(_handle_spawn_task({"goal": "a"}, parent_agent=agent))
            t2 = json.loads(_handle_spawn_task({"goal": "b"}, parent_agent=agent))
            t3 = json.loads(_handle_spawn_task({"goal": "c"}, parent_agent=agent))

            await asyncio.sleep(0.1)

            result = json.loads(_handle_wait_task(
                {"task_ids": [t1["task_id"], t2["task_id"], t3["task_id"]]},
                parent_agent=agent,
            ))

        assert len(result) == 3
        for tid, r in result.items():
            assert "error" not in r

    async def test_auto_injection_works(self):
        """Child result auto-injects as user message in parent conversation."""
        agent = _make_agent()
        collector = EventCollector()
        agent._event_collector = collector

        # Simulate child completing
        collector.notify_child_done("x", {"success": True, "summary": "research done"})

        # Drain should produce events
        events = collector.drain_completed()
        assert len(events) == 1
        assert events[0]["task_id"] == "x"

        # In real chat(), these events become user messages
        messages = [{"role": "user", "content": "search"}]
        for ev in events:
            messages.append({"role": "user", "content": f"## Subagent `{ev['task_id']}` completed\n\n{ev['result']['summary']}"})

        assert len(messages) == 2
        assert "research done" in messages[1]["content"]

    async def test_persistent_loop_waits_for_children(self):
        """After model produces final text, chat() waits for running children."""
        agent = _make_agent()
        collector = EventCollector()
        agent._event_collector = collector

        # Simulate running child
        async def slow_child():
            await asyncio.sleep(0.1)
            return {"success": True, "summary": "finally done"}
        child_task = asyncio.ensure_future(slow_child())
        agent._bg_tasks["slow"] = child_task

        # Model finished, but children running
        assert bool(agent._bg_tasks) is True

        # wait_next should fire when child completes
        arrived = await collector.wait_next(timeout=5.0)
        assert arrived is True
        assert collector.peek() is True
```

**Step 2: Run integration tests**

```bash
pytest tests/test_delegate_tool.py::TestSpawnWaitIntegration -v
```

**Step 3: Commit**

```bash
git add tests/test_delegate_tool.py
git commit -m "test: add spawn/wait/injection integration tests"
```

---

### Task 9: Clean up old code — remove _run_child_sync, ThreadPoolExecutor remnants

**Objective:** Remove sync-only code that's no longer used.

**Files:**
- Modify: `tyagent/agent.py` — remove `_bg_executor`, `_ensure_executor`, ThreadPoolExecutor imports
- Modify: `tyagent/tools/delegate_tool.py` — remove `_run_child_sync` and `_run_child_in_thread`

**Step 1: Remove ThreadPoolExecutor from agent.py**

In `__init__`:
```diff
- self._bg_executor: concurrent.futures.ThreadPoolExecutor | None = None
- self._bg_tasks: Dict[str, concurrent.futures.Future] = {}
+ self._bg_tasks: Dict[str, asyncio.Task] = {}
```

In `close()`:
```diff
- if self._bg_executor is not None:
-     self._bg_executor.shutdown(wait=False)
-     self._bg_executor = None
- self._bg_tasks.clear()
+ for tid, task in list(self._bg_tasks.items()):
+     if not task.done():
+         task.cancel()
+ if self._bg_tasks:
+     await asyncio.gather(*self._bg_tasks.values(), return_exceptions=True)
+ self._bg_tasks.clear()
```

Remove `import concurrent.futures` from agent.py if it was added.

**Step 2: Remove `_run_child_sync` from delegate_tool.py**

```diff
- def _run_child_sync(...)
- def _run_child_in_thread(...)
```

Keep `_run_child_async` as the only child execution path.

**Step 3: Update tests that reference removed functions**

Update imports in `test_delegate_tool.py`:
```diff
- from tyagent.tools.delegate_tool import (
-     _run_child_sync,
-     ...
- )
+ from tyagent.tools.delegate_tool import (
+     _run_child_async,
+     ...
+ )
```

Remove `TestRunChildSync` class entirely (tests should use `_run_child_async` instead).

Add replacement tests for `_run_child_async`:

```python
class TestRunChildAsync:
    async def test_successful_child_run(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="Child completed task successfully.")
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            result = await _run_child_async(
                task_id="t1",
                model="test", api_key="k", base_url="http://x",
                system_prompt="p", reasoning_effort=None,
                goal="Do a thing", tool_names=[],
                max_tool_turns=30, context=None,
            )

        assert result["success"] is True
        assert result["summary"] == "Child completed task successfully."
        assert result["error"] is None
        assert "duration_seconds" in result

    async def test_child_failure_captured(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(side_effect=RuntimeError("Boom"))
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            result = await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="p", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30, context=None,
            )

        assert result["success"] is False
        assert "Boom" in result["error"]
```

Also update `_make_agent` mock if needed — `_bg_tasks` is now `Dict[str, asyncio.Task]` not `Dict[str, concurrent.futures.Future]`.

**Step 4: Run full test suite**

```bash
pytest tests/ -q
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add tyagent/agent.py tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "refactor: remove old sync-only code, clean up for async subagent architecture"
```

---

### Task 10: Full test suite regression + final review

**Objective:** Ensure all 411+ tests pass and the implementation is complete.

**Files:**
- None (verification only)

**Step 1: Run full suite**

```bash
pytest tests/ -q
```

**Step 2: Fix any regressions**

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: final cleanup, all tests passing"
```

---

## Alignment Summary (vs Codex)

| Feature | Codex | This Plan |
|---|---|---|
| spawn non-blocking | ✅ | ✅ |
| Child = independent async task | ✅ (tokio) | ✅ (asyncio) |
| Auto-inject child result into parent | ✅ (mailbox → user msg) | ✅ (EventCollector.drain → user msg) |
| Injection triggers new LLM turn | ✅ (select! triggers run_turn) | ✅ (chat() persistent loop continues) |
| chat() is event loop, not one-shot | ✅ (agent thread runs forever) | ✅ (chat() stays alive while children run) |
| wait = explicit blocking tool | ✅ | ✅ |
| wait supports multi-source listening | ✅ (select! over child + ops + user) | ✅ (asyncio.wait with timeout) |
| Parent→Child ops channel | ✅ | ❌ (post-MVP) |
| trigger_turn control | ✅ | ❌ (always auto-inject + trigger) |
| Model can interleave spawn/wait/other | ✅ | ✅ |

The two omitted features (ops channel, trigger_turn) are straightforward to add later but add complexity without immediate benefit for the MVP. Everything that makes the Codex model powerful for async subagent management is covered.

---

## Key Risks and Mitigations

1. **`asyncio.get_event_loop()` from `run_in_executor` thread** — This works in Python 3.10+ for threads spawned in the same process as a running loop. Verified: `loop = asyncio.get_event_loop()` returns the running loop. If it fails, fallback is to pass the loop explicitly or use `asyncio.run_coroutine_threadsafe()`.

2. **`run_in_executor` + blocking on child completion** — `wait_task` calls `loop.run_until_complete(_wait_all())`. This is safe because the event loop is already running (it's the main loop thread). `run_until_complete` is actually not needed — we can use `asyncio.run_coroutine_threadsafe().result()` which is the standard pattern from threads. But `loop.run_until_complete()` works from the running loop's thread too.

3. **Memory/state leaks** — Children store their results in `_bg_tasks`. `wait_task` pops them. `close()` cancels remaining. No leak path.

4. **Child timeout** — 600s per child via `asyncio.wait_for`. Configurable via constant.
