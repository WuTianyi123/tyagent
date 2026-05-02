# Async Subagent — spawn_task / wait_task Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Split `delegate_task` into `spawn_task` (non-blocking, returns task_id immediately) + `wait_task` (blocks until results ready), enabling the same Codex-style "spawn → do other work → wait" pattern within OpenAI's native tool-calling protocol.

**Architecture:** `concurrent.futures.ThreadPoolExecutor` on `TyAgent` runs child agents in background threads. `spawn_task` submits and returns a `task_id` immediately. `wait_task` calls `future.result()` to block until completion. `delegate_task` becomes a backward-compatible spawn+wait wrapper. No mailbox injection needed for MVP; the model explicitly calls `wait_task`.

**Tech Stack:** Python 3.11+, `concurrent.futures`, pytest, unittest.mock

---

### Task 1: Add ThreadPoolExecutor to TyAgent

**Objective:** Give TyAgent a place to track background tasks.

**Files:**
- Modify: `tyagent/agent.py` — `__init__` and `close`

**Step 1: Add imports and init**

```python
# In agent.py __init__, after self._client = ...:
import concurrent.futures

self._bg_executor: concurrent.futures.ThreadPoolExecutor | None = None
self._bg_tasks: Dict[str, concurrent.futures.Future] = {}
```

**Step 2: Add a `_ensure_executor()` helper**

```python
def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
    if self._bg_executor is None:
        self._bg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="tyagent-bg"
        )
    return self._bg_executor
```

**Step 3: Shutdown in `close()`**

```python
# In close(), before or after self._client.aclose():
if self._bg_executor is not None:
    self._bg_executor.shutdown(wait=False)
    self._bg_executor = None
self._bg_tasks.clear()
```

**Step 4: Write test — executor is created lazily**

```python
# tests/test_agent.py
def test_bg_executor_is_lazy():
    agent = TyAgent(model="test", api_key="k", base_url="http://x")
    assert agent._bg_executor is None
    ex = agent._ensure_executor()
    assert ex is not None
    assert agent._bg_executor is ex  # cached

def test_bg_executor_shutdown_in_close():
    agent = TyAgent(model="test", api_key="k", base_url="http://x")
    ex = agent._ensure_executor()
    agent.close()
    # After close, executor is cleaned up
    assert agent._bg_executor is None
```

**Step 5: Commit**

```bash
git add tyagent/agent.py tests/test_agent.py
git commit -m "feat: add ThreadPoolExecutor to TyAgent for background tasks"
```

---

### Task 2: Refactor _run_child_sync → add _run_child (thread-safe)

**Objective:** Extract the child-agent creation and execution into a function callable from a thread — no API changes, just internal rename and minor cleanup.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Keep `_run_child_sync` as-is (it already works in threads)**

The current `_run_child_sync` creates its own `asyncio.new_event_loop()`, so it's already thread-safe. No refactoring needed — just confirm the signature is compatible with `executor.submit()` (it is — all args are serializable primitives).

**Step 2: Add a thin `_run_child_in_thread` wrapper that calls it**

```python
def _run_child_in_thread(
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
) -> Dict[str, Any]:
    """Thread-safe entry point for executor.submit()."""
    return _run_child_sync(
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        goal=goal,
        tool_names=tool_names,
        max_tool_turns=max_tool_turns,
        context=context,
        compression=compression,
        tool_progress_callback=tool_progress_callback,
    )
```

(Optional — if `_run_child_sync` signature stays stable, we can submit it directly. This wrapper is for clarity only.)

**Step 3: Update `_handle_delegate_task` to use `_run_child_in_thread`**

Just a rename in the call — no logic changes.

```python
result = _run_child_in_thread(
    model=parent_agent.model,
    ...
)
```

**Step 4: Write test — `_run_child_in_thread` gives same result as `_run_child_sync`**

```python
def test_run_child_in_thread_same_as_sync():
    with patch(...):
        r1 = _run_child_sync(...)
        r2 = _run_child_in_thread(...)
    assert r1 == r2
```

**Step 5: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "refactor: add _run_child_in_thread wrapper for executor compatibility"
```

---

### Task 3: Implement spawn_task tool

**Objective:** Non-blocking child agent launch — returns `{task_id, status}` immediately.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Add `_handle_spawn_task` handler**

```python
import uuid

def _handle_spawn_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Spawn a child agent in the background. Returns task_id immediately."""
    goal = args.get("goal", "").strip()
    if not goal:
        return tool_error("goal is required for spawn_task.")

    context = args.get("context") or None
    toolsets: Optional[List[str]] = args.get("toolsets") or None
    max_tool_turns = args.get("max_tool_turns", DEFAULT_SUBAGENT_MAX_TOOL_TURNS)

    # Coerce max_tool_turns (same logic as delegate_task)
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

    # Build allowed tool names
    all_names = registry.get_all_names()
    allowed = [n for n in all_names if n not in DELEGATE_BLOCKED_TOOLS]
    if toolsets:
        allowed = [n for n in toolsets if n in allowed]

    # Build child progress callback
    parent_cb = getattr(parent_agent, "_tool_progress_callback", None)
    child_cb = None
    if parent_cb is not None:
        def _child_progress(tool_name: str, args_inner: dict) -> None:
            parent_cb(tool_name, args_inner, prefix="📤 ")
        child_cb = _child_progress

    # Submit to background executor
    executor = parent_agent._ensure_executor()
    future = executor.submit(
        _run_child_sync,
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
    )

    task_id = str(uuid.uuid4())[:8]
    parent_agent._bg_tasks[task_id] = future

    logger.info("spawn_task: launched %s goal=%r", task_id, goal[:80])
    return json.dumps({"task_id": task_id, "status": "running"}, ensure_ascii=False)
```

**Step 2: Add schema and register**

```python
SPAWN_TASK_SCHEMA: Dict[str, Any] = {
    "name": "spawn_task",
    "description": (
        "Launch a child agent to work on a task in the background. "
        "Returns immediately with a task_id — the child runs independently. "
        "You can continue working and call wait_task later to collect results.\n\n"
        "Use this when you want to parallelize: spawn multiple children, "
        "do other work, then wait for all results at once."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What the child should accomplish.",
            },
            "context": {
                "type": "string",
                "description": "Optional background info (file paths, constraints).",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional restricted tool list.",
            },
            "max_tool_turns": {
                "type": "integer",
                "description": f"Max tool turns (default: {DEFAULT_SUBAGENT_MAX_TOOL_TURNS}).",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["goal"],
    },
}

registry.register(
    name="spawn_task",
    schema=SPAWN_TASK_SCHEMA,
    handler=_handle_spawn_task,
    description="Launch a child agent in background (non-blocking)",
    emoji="🚀",
    wants_parent=True,
)
```

**Step 3: Write tests**

```python
class TestSpawnTask:
    def test_returns_task_id_immediately(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_sync", return_value={...}):
            result = _call_spawn({"goal": "test"}, parent_agent=agent)
        assert "task_id" in result
        assert result["status"] == "running"
        assert "error" not in result

    def test_task_stored_in_agent(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_sync"):
            result = _call_spawn({"goal": "test"}, parent_agent=agent)
        tid = result["task_id"]
        assert tid in agent._bg_tasks

    def test_no_parent_agent_errors(self):
        result = _call_spawn({"goal": "test"}, parent_agent=None)
        assert "error" in result
```

**Step 4: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "feat: add spawn_task tool (non-blocking child agent launch)"
```

---

### Task 4: Implement wait_task tool

**Objective:** Block until spawned child agents complete, return aggregated results.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Add `_handle_wait_task` handler**

```python
def _handle_wait_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Wait for spawned child agents and return their results."""
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
        return tool_error("No parent agent context.")

    results = {}
    for tid in task_ids:
        if not isinstance(tid, str):
            results[str(tid)] = {"error": f"Invalid task_id type: {type(tid).__name__}"}
            continue
        future = parent_agent._bg_tasks.pop(tid, None)
        if future is None:
            results[tid] = {"error": f"Task not found: {tid}"}
            continue
        try:
            results[tid] = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            results[tid] = {"error": f"Task timed out after {timeout}s"}
            # Put it back so it can be waited again
            parent_agent._bg_tasks[tid] = future
        except Exception as exc:
            results[tid] = {"error": f"{type(exc).__name__}: {exc}"}

    logger.info("wait_task: collected %d results", len(results))
    return json.dumps(results, ensure_ascii=False)
```

**Step 2: Add schema and register**

```python
WAIT_TASK_SCHEMA: Dict[str, Any] = {
    "name": "wait_task",
    "description": (
        "Wait for one or more spawned child agents to complete. "
        "Returns aggregated results keyed by task_id. "
        "Use this after calling spawn_task to collect the child's summary.\n\n"
        "If a task has already completed, wait_task returns it immediately. "
        "If a task is still running, wait_task blocks for up to `timeout` seconds."
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
                "description": "Max seconds to wait per task (default: 300).",
            },
        },
        "required": ["task_ids"],
    },
}

registry.register(
    name="wait_task",
    schema=WAIT_TASK_SCHEMA,
    handler=_handle_wait_task,
    description="Wait for background child agents to complete",
    emoji="⏳",
    wants_parent=True,
)
```

**Step 3: Write tests**

```python
class TestWaitTask:
    def test_collects_completed_tasks(self):
        agent = _make_agent()
        future = MagicMock()
        future.result.return_value = {"success": True, "summary": "done"}
        agent._bg_tasks["abc123"] = future

        result = _call_wait({"task_ids": ["abc123"]}, parent_agent=agent)
        assert "abc123" in result
        assert result["abc123"]["summary"] == "done"

    def test_task_not_found_error(self):
        agent = _make_agent()
        result = _call_wait({"task_ids": ["nonexistent"]}, parent_agent=agent)
        assert "error" in result["nonexistent"]

    def test_missing_task_ids_errors(self):
        agent = _make_agent()
        result = _call_wait({}, parent_agent=agent)
        assert "error" in result

    def test_timeout_positive_required(self):
        agent = _make_agent()
        r1 = _call_wait({"task_ids": ["x"], "timeout": 0}, parent_agent=agent)
        assert "error" in r1
        r2 = _call_wait({"task_ids": ["x"], "timeout": -1}, parent_agent=agent)
        assert "error" in r2

    def test_task_popped_after_collection(self):
        agent = _make_agent()
        future = MagicMock()
        future.result.return_value = {"success": True, "summary": "ok"}
        agent._bg_tasks["x"] = future

        _call_wait({"task_ids": ["x"]}, parent_agent=agent)
        assert "x" not in agent._bg_tasks  # consumed

    def test_timed_out_task_stays(self):
        agent = _make_agent()
        future = MagicMock()
        future.result.side_effect = concurrent.futures.TimeoutError()
        agent._bg_tasks["x"] = future

        result = _call_wait({"task_ids": ["x"], "timeout": 1}, parent_agent=agent)
        assert "timed out" in result["x"]["error"]
        assert "x" in agent._bg_tasks  # still there for retry
```

**Step 4: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "feat: add wait_task tool (blocking result collection)"
```

---

### Task 5: Update delegate_task as spawn+wait wrapper

**Objective:** Keep backward compatibility — `delegate_task` should still work exactly as before, but internally it uses spawn+wait.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Rewrite `_handle_delegate_task` as spawn + immediate wait**

```python
def _handle_delegate_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Convenience wrapper: spawn_task + immediate wait_task."""
    # Remove task_ids/timeout from args if present (delegate_task doesn't take them)
    spawn_args = {k: v for k, v in args.items() if k not in ("task_ids", "timeout")}

    # Step 1: spawn
    spawn_result = json.loads(_handle_spawn_task(spawn_args, parent_agent=parent_agent))
    if "error" in spawn_result:
        return json.dumps(spawn_result, ensure_ascii=False)

    task_id = spawn_result["task_id"]

    # Step 2: wait
    wait_args = {"task_ids": [task_id]}
    wait_result = json.loads(_handle_wait_task(wait_args, parent_agent=parent_agent))

    # Flatten: delegate_task returns one result, not {task_id: result}
    single = wait_result.get(task_id, {})
    return json.dumps(single, ensure_ascii=False)
```

**Step 2: Verify old delegate_task schema unchanged**

No schema changes needed — `DELEGATE_TASK_SCHEMA` stays as-is.

**Step 3: Write regression tests**

```python
class TestDelegateTaskBackwardCompat:
    def test_delegate_task_still_works(self):
        """Existing delegate_task behavior is preserved."""
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_sync",
                          return_value={"success": True, "summary": "hi", "error": None, "duration_seconds": 1}):
            result = _call_handler({"goal": "test"}, parent_agent=agent)
        assert result["success"] is True
        assert result["summary"] == "hi"

    def test_delegate_task_errors_on_spawn_failure(self):
        agent = _make_agent()
        result = _call_handler({"goal": ""}, parent_agent=agent)
        assert "error" in result
```

**Step 4: Run full test suite**

```bash
pytest tests/ -q
```

Expected: all existing delegate_task tests pass, new spawn/wait tests pass.

**Step 5: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "refactor: delegate_task as spawn+wait wrapper (backward compatible)"
```

---

### Task 6: Update DELEGATE_BLOCKED_TOOLS

**Objective:** Block `spawn_task` and `wait_task` from child agents (no recursive background spawning for MVP).

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`

**Step 1: Update blocked set**

```python
DELEGATE_BLOCKED_TOOLS = frozenset(
    ["delegate_task", "spawn_task", "wait_task", "memory"]
)
```

**Step 2: Add verification test**

```python
def test_spawn_task_is_blocked_from_children():
    assert "spawn_task" in DELEGATE_BLOCKED_TOOLS
    assert "wait_task" in DELEGATE_BLOCKED_TOOLS
```

**Step 3: Commit**

```bash
git add tyagent/tools/delegate_tool.py tests/test_delegate_tool.py
git commit -m "fix: block spawn_task/wait_task from child agents"
```

---

### Task 7: End-to-end test — spawn, interleave work, wait

**Objective:** Verify the full pattern works end-to-end with a mocked child agent.

**Files:**
- Create/Modify: `tests/test_delegate_tool.py`

**Step 1: Write integration test**

```python
import time
import threading

class TestSpawnWaitIntegration:
    def test_spawn_then_wait_gives_result(self):
        """spawn_task → wait_task returns child result."""
        agent = _make_agent()

        def slow_child(*args, **kwargs):
            time.sleep(0.1)  # Simulate work
            return {"success": True, "summary": "slow work done", "error": None, "duration_seconds": 0.1}

        with patch.object(delegate_tool, "_run_child_sync", side_effect=slow_child):
            spawn_result = _call_spawn({"goal": "slow task"}, parent_agent=agent)
            tid = spawn_result["task_id"]

            wait_result = _call_wait({"task_ids": [tid]}, parent_agent=agent)

        assert wait_result[tid]["success"] is True
        assert wait_result[tid]["summary"] == "slow work done"

    def test_spawn_multiple_then_wait_all(self):
        """spawn 3 tasks, wait for all."""
        agent = _make_agent()

        def child_work(*args, **kwargs):
            time.sleep(0.05)
            return {"success": True, "summary": f"result-{threading.current_thread().name}", "error": None, "duration_seconds": 0.05}

        with patch.object(delegate_tool, "_run_child_sync", side_effect=child_work):
            r1 = _call_spawn({"goal": "a"}, parent_agent=agent)
            r2 = _call_spawn({"goal": "b"}, parent_agent=agent)
            r3 = _call_spawn({"goal": "c"}, parent_agent=agent)

            results = _call_wait(
                {"task_ids": [r1["task_id"], r2["task_id"], r3["task_id"]]},
                parent_agent=agent,
            )

        assert len(results) == 3
        for tid, r in results.items():
            assert r["success"] is True

    def test_spawn_child_runs_concurrently_with_parent(self):
        """Child runs in background thread — parent can do other work."""
        agent = _make_agent()
        child_started = threading.Event()
        parent_proceed = threading.Event()

        def slow_child(*args, **kwargs):
            child_started.set()
            parent_proceed.wait(timeout=5)  # Wait for parent signal
            return {"success": True, "summary": "done", "error": None, "duration_seconds": 1}

        with patch.object(delegate_tool, "_run_child_sync", side_effect=slow_child):
            result = _call_spawn({"goal": "slow"}, parent_agent=agent)
            tid = result["task_id"]

            # Child is running in background — parent can continue
            assert child_started.wait(timeout=2), "Child should have started by now"

            # Parent does "other work" while child runs
            # (in real scenario, this would be more tool calls)

            # Now let child finish
            parent_proceed.set()

            # Collect result
            wait_result = _call_wait({"task_ids": [tid]}, parent_agent=agent)

        assert wait_result[tid]["success"] is True
```

**Step 2: Run integration tests**

```bash
pytest tests/test_delegate_tool.py::TestSpawnWaitIntegration -v
```

**Step 3: Commit**

```bash
git add tests/test_delegate_tool.py
git commit -m "test: add spawn/wait integration tests"
```

---

### Task 8: Run full test suite and verify no regressions

**Objective:** Ensure all existing 411 tests pass with the new code.

**Files:**
- None (verification only)

**Step 1: Run full suite**

```bash
pytest tests/ -q
```

**Expected:** 411+ tests passing (original 411 + new spawn/wait tests).

**Step 2: Fix any regressions**

Debug and fix. Commit fixes.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: final cleanup, all tests passing"
```

---

## Design Notes

### Why ThreadPoolExecutor instead of asyncio tasks?

- **Simplicity**: `future.result(timeout)` blocks the calling thread (which is already in `run_in_executor`), no event-loop bridging needed.
- **Isolation**: Each child agent creates its own event loop (`asyncio.new_event_loop()`), so no shared loop state.
- **Compatibility**: Works identically to the current `_run_child_sync` — just submitted to a pool instead of waited immediately.

### Why no mailbox injection for MVP?

- Codex's mailbox injection ensures child results eventually surface even if the model forgets to call `wait_agent`. But tyagent's `chat()` loop runs tool turns until the model produces a final answer. If the model spawns and never waits, those results are orphaned — but this is the model's mistake, not a system failure.
- Adding mailbox injection would require modifying the `chat()` loop to interleave background results, which is a larger change. If we find models consistently forgetting to wait, we can add it later.

### What the model sees:

```
Turn N:   Tool call: spawn_task(goal="search papers") → {"task_id":"a1b2","status":"running"}
Turn N+1: Tool call: read_file(path="main.py") → ...
Turn N+2: Tool call: wait_task(task_ids=["a1b2"]) → {"a1b2": {"success":true,"summary":"Found 3 papers..."}}
```
