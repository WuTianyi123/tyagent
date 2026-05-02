# Async Subagent — Fully Codex-Aligned Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace synchronous `delegate_task` with Codex-aligned async subagent architecture — `spawn_task` (non-blocking) + `wait_task` (blocking, multi-source) + auto-injection of completed child results into parent's LLM turn + user message interleaving during child execution + parent→child ops tools.

**Architecture:** Pure asyncio throughout. `chat()` becomes a persistent event loop that listens on **three sources** concurrently — child completion events (`EventCollector`), new user messages (`_user_queue`), and interrupts (`_interrupt_event`). Gateway enqueues subsequent user messages instead of discarding them. `close_task` and `list_tasks` tools provide parent→child ops (simplified ops channel).

---

## Three Gaps Closed

### Gap 1: User input blocked during chat()

**Codex:** Agent thread runs `select!` over user messages + child completions + ops. User can always send messages.

**tyagent fix:** Agent holds a `_user_queue: asyncio.Queue`. The gateway, when it detects a session is already processing, enqueues the new message instead of discarding it. chat() in its wait loop `asyncio.wait()`s on both `collector.wait_next()` and `_user_queue.get()`. When a user message arrives while children run, it's consumed immediately — the model sees the new message and decides how to respond.

Architecture change: Gateway needs `_session_key_to_agent` mapping (exists via `_agent_cache`), and `_handle_message_event` checks `_active_sessions` before the main processing block.

### Gap 2: No ops channel

**Codex:** `tx_ops` channel for parent→child commands (close, send_message, cancel).

**tyagent fix:** Two tools: `close_task(task_id)` — cancels the child's asyncio task; `list_tasks()` — returns running/task status. Plus a `_interrupt_event` on the agent that gateway can set to wake up a waiting chat() loop.

This is a simplified ops channel covering the critical parent→child operations (cancel, inspect). Full bidirectional messaging is post-MVP.

### Gap 3: Missing tools

**Codex tools:** `close_agent`, `list_agents`, `send_agent_message`

**tyagent fix:** `close_task(task_id)` + `list_tasks()`. `send_agent_message` deferred to post-MVP.

---

## Architecture

```
gateway._on_message(event)
  │
  ├── session_key in _active_sessions?
  │     ├── No  → normal processing (create agent, call chat())
  │     └── Yes → agent._user_queue.put(msg)  ← gap 1 fix
  │                  └── return (don't call chat())
  │
  agent.chat(messages, tools)
  │
  ├── ▶ Auto-inject: drain EventCollector → inject as user messages
  │
  ├── ▶ Drain _user_queue → inject as user messages
  │
  ├── ▶ LLM API call
  │     │
  │     ├── spawn_task    → asyncio.create_task(_run_child_async)
  │     │                    → child completes → collector.notify_child_done()
  │     │
  │     ├── wait_task     → asyncio.wait([futures], timeout=timeout)
  │     │                    → multi-source (children + timeout)
  │     │
  │     ├── close_task    → task.cancel()
  │     │
  │     ├── list_tasks    → return task states
  │     │
  │     └── sync tools    → run_in_executor(registry.dispatch)
  │
  ├── ▶ Model produced final text
  │     │
  │     ├── No children + no pending events → return ✅
  │     │
  │     └── Children running or pending events
  │           └── asyncio.wait([
  │                 collector.wait_next(),    ── child complete
  │                 _user_queue.get(),        ── user message  ← gap 1
  │               ], timeout=timeout)
  │               │
  │               ├── child_complete → inject → continue LLM
  │               ├── user_message   → inject → continue LLM
  │               └── timeout        → return current answer
  │
  └──▶ (loop continues until no children + no pending)
```

---

## Task Breakdown

### Task 1: EventCollector class

**Objective:** Bridge between child agent completion (async task) and parent agent's chat() loop.

**Files:** Create `tyagent/events.py`

```python
class EventCollector:
    """Collects events from subagents for the parent chat() loop to consume."""

    def __init__(self):
        self._completed: Dict[str, Dict] = {}
        self._event = asyncio.Event()

    def notify_child_done(self, task_id: str, result: Dict) -> None:
        self._completed[task_id] = result
        self._event.set()

    def drain_completed(self) -> List[Dict]:
        events = [
            {"type": "child_complete", "task_id": tid, "result": result}
            for tid, result in self._completed.items()
        ]
        self._completed.clear()
        self._event.clear()
        return events

    def peek(self) -> bool:
        return bool(self._completed)

    async def wait_next(self, timeout: Optional[float] = None) -> bool:
        if self._completed:
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
```

**Tests:** Verify drain/peek/wait lifecycle, multiple children, timeout behavior.

**Commit:** `git commit -m "feat: add EventCollector for bridging child completion to parent loop"`

---

### Task 2: TyAgent lifecycle — _bg_tasks, _event_collector, _user_queue, _interrupt_event

**Objective:** Give TyAgent the infrastructure to manage async child tasks, receive user messages mid-chat(), and respond to external interrupts.

**Files:** Modify `tyagent/agent.py`

**Step 1: Add instance variables in `__init__`**

```python
self._bg_tasks: Dict[str, asyncio.Task] = {}
self._event_collector: Optional[EventCollector] = None
self._user_queue: asyncio.Queue[Dict] = asyncio.Queue()
self._interrupt_event: asyncio.Event = asyncio.Event()
```

**Step 2: Update `close()` to clean up**

```python
async def close(self) -> None:
    # Cancel running background tasks
    for tid, task in list(self._bg_tasks.items()):
        if not task.done():
            task.cancel()
    if self._bg_tasks:
        await asyncio.gather(*self._bg_tasks.values(), return_exceptions=True)
    self._bg_tasks.clear()
    self._event_collector = None
    await self._client.aclose()
```

Note: `_user_queue` and `_interrupt_event` are plain asyncio primitives — no cleanup needed (no resources to free).

**Tests:**

```python
async def test_bg_tasks_initially_empty():
    agent = TyAgent(model="test", api_key="k", base_url="http://x")
    assert agent._bg_tasks == {}
    assert agent._event_collector is None
    assert agent._interrupt_event is not None
    assert agent._user_queue is not None
    await agent.close()

async def test_close_cancels_running_tasks():
    agent = TyAgent(...)
    async def never_finish():
        await asyncio.Event().wait()
    task = asyncio.create_task(never_finish())
    agent._bg_tasks["abc"] = task
    await agent.close()
    assert task.cancelled()
```

**Commit:** `git commit -m "feat: add async subagent lifecycle to TyAgent"`

---

### Task 3: _run_child_async and spawn_task tool

**Objective:** Child agent as an asyncio task. `spawn_task` creates the task and returns `task_id` immediately.

**Files:**
- Modify: `tyagent/tools/delegate_tool.py`
- Import: `from tyagent.events import EventCollector`

**Step 1: Add `_run_child_async`**

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

    On completion (or failure), notifies the collector.
    Never raises — all errors captured in result dict.
    """
    t0 = time.monotonic()
    child_system = system_prompt
    if context:
        child_system = f"{system_prompt}\n\nTask context: {context}"

    child = TyAgent(
        model=model, api_key=api_key, base_url=base_url,
        max_tool_turns=max_tool_turns, system_prompt=child_system,
        reasoning_effort=reasoning_effort, compression=compression,
    )
    child_messages = [{"role": "user", "content": goal}]
    tool_defs = registry.get_definitions(names=tool_names)

    try:
        summary = await asyncio.wait_for(
            child.chat(child_messages, tools=tool_defs,
                        tool_progress_callback=tool_progress_callback),
            timeout=600.0,
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
        try:
            await child.close()
        except Exception:
            pass

    if collector is not None:
        collector.notify_child_done(task_id, result)
    return result
```

**Step 2: Add `_build_allowed_tools` helper**

```python
def _build_allowed_tools(toolsets: Optional[List[str]] = None) -> List[str]:
    """Return tool names allowed for a child agent (blocked tools excluded)."""
    all_names = registry.get_all_names()
    allowed = [n for n in all_names if n not in DELEGATE_BLOCKED_TOOLS]
    if toolsets:
        allowed = [n for n in toolsets if n in allowed]
    return allowed
```

**Step 3: Add `_handle_spawn_task`**

```python
import uuid

def _handle_spawn_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Spawn a child agent as an asyncio task. Returns task_id immediately.

    The child is stored in parent_agent._bg_tasks for wait_task to track.
    On completion, it notifies parent_agent._event_collector for auto-injection.
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
        return tool_error("spawn_task requires a session agent.")

    allowed = _build_allowed_tools(toolsets)

    # Child progress callback
    parent_cb = getattr(parent_agent, "_tool_progress_callback", None)
    child_cb = None
    if parent_cb is not None:
        def _child_progress(tn: str, ai: dict) -> None:
            parent_cb(tn, ai, prefix="📤 ")
        child_cb = _child_progress

    task_id = str(uuid.uuid4())[:8]

    # Lazily create collector
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
    )

    loop = asyncio.get_event_loop()
    child_task = loop.create_task(child_coro)
    parent_agent._bg_tasks[task_id] = child_task

    logger.info("spawn_task: launched %s goal=%r", task_id, goal[:80])
    return json.dumps({"task_id": task_id, "status": "running"}, ensure_ascii=False)
```

**Step 4: Schema and registration**

```python
SPAWN_TASK_SCHEMA = {
    "name": "spawn_task",
    "description": (
        "Launch a child agent in the background. Returns immediately with "
        "a task_id — the child runs independently. You can continue working "
        "and call wait_task later to collect results.\n\n"
        "Note: If a child completes before you call wait_task, its result "
        "is automatically injected into the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "context": {"type": "string"},
            "toolsets": {"type": "array", "items": {"type": "string"}},
            "max_tool_turns": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["goal"],
    },
}

registry.register(
    name="spawn_task", schema=SPAWN_TASK_SCHEMA,
    handler=_handle_spawn_task,
    description="Launch a child agent in background (non-blocking)",
    emoji="🚀", wants_parent=True,
)
```

**Tests:**

```python
class TestSpawnTask:
    def test_returns_task_id(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_async"):
            r = json.loads(_handle_spawn_task({"goal": "test"}, parent_agent=agent))
        assert "task_id" in r and r["status"] == "running"

    def test_stored_in_bg_tasks(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_run_child_async"):
            r = json.loads(_handle_spawn_task({"goal": "test"}, parent_agent=agent))
        assert r["task_id"] in agent._bg_tasks

    def test_collector_created_lazily(self):
        agent = _make_agent()
        assert agent._event_collector is None
        with patch.object(delegate_tool, "_run_child_async"):
            _handle_spawn_task({"goal": "test"}, parent_agent=agent)
        assert agent._event_collector is not None

    def test_errors(self):
        assert "error" in json.loads(_handle_spawn_task({"goal": ""}, parent_agent=_make_agent()))
        assert "error" in json.loads(_handle_spawn_task({"goal": "x"}, parent_agent=None))
```

**Commit:** `git commit -m "feat: add spawn_task tool with async child lifecycle"`

---

### Task 4: wait_task tool

**Objective:** Block until specified children complete, return aggregated results. Uses `asyncio.wait()` with timeout for multi-source waiting (Codex-align: wait as a dedicated blocking tool).

**Files:** Modify `tyagent/tools/delegate_tool.py`

**Step 1: Add `_handle_wait_task`**

```python
def _handle_wait_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Wait for spawned child agents. Uses asyncio.wait() with timeout.

    Results keyed by task_id. Timed-out tasks are left in _bg_tasks for retry.
    """
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

    results: Dict[str, Dict] = {}
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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_wait_all())

    return json.dumps(results, ensure_ascii=False)
```

**Step 2: Schema and registration**

```python
WAIT_TASK_SCHEMA = {
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

registry.register(
    name="wait_task", schema=WAIT_TASK_SCHEMA,
    handler=_handle_wait_task,
    description="Wait for background child agents to complete",
    emoji="⏳", wants_parent=True,
)
```

**Tests:** Verify result collection, missing tasks, timeout, cancelled tasks.

**Commit:** `git commit -m "feat: add wait_task tool with asyncio.wait"`

---

### Task 5: close_task + list_tasks tools (ops channel — Gap 2 & 3)

**Objective:** Parent→child operations: close (cancel) a running child, inspect all running children.

**Files:** Modify `tyagent/tools/delegate_tool.py`

**Step 1: Add `_handle_close_task`**

```python
def _handle_close_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Close (cancel) a running child agent by cancelling its asyncio task."""
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
        logger.info("close_task: cancelled %s", task_id)
        return json.dumps({"success": True, "message": f"Cancelled {task_id}"})
    else:
        logger.info("close_task: %s already completed", task_id)
        return json.dumps({"success": True, "message": f"{task_id} already completed"})
```

**Step 2: Schema and registration for close_task**

```python
CLOSE_TASK_SCHEMA = {
    "name": "close_task",
    "description": (
        "Close (cancel) a running child agent. Use this if a child "
        "is taking too long or producing irrelevant results. "
        "Idempotent — safe to call multiple times."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task ID from spawn_task."},
        },
        "required": ["task_id"],
    },
}

registry.register(
    name="close_task", schema=CLOSE_TASK_SCHEMA,
    handler=_handle_close_task,
    description="Cancel a running child agent",
    emoji="🛑", wants_parent=True,
)
```

**Step 3: Add `_handle_list_tasks`**

```python
def _handle_list_tasks(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """List all running/completed child agents and their status."""
    if parent_agent is None:
        return tool_error("list_tasks requires a session agent.")

    tasks = {}
    for tid, task in parent_agent._bg_tasks.items():
        tasks[tid] = {
            "done": task.done(),
            "cancelled": task.cancelled(),
        }
    return json.dumps(tasks, ensure_ascii=False)
```

**Step 4: Schema and registration for list_tasks**

```python
LIST_TASKS_SCHEMA = {
    "name": "list_tasks",
    "description": (
        "List all spawned child agents and their current status "
        "(running, completed, or cancelled). Use this to check "
        "on background work before calling wait_task."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

registry.register(
    name="list_tasks", schema=LIST_TASKS_SCHEMA,
    handler=_handle_list_tasks,
    description="List running/completed child agents",
    emoji="📋", wants_parent=True,
)
```

**Tests:**

```python
class TestCloseTask:
    def test_cancel_running_task(self):
        agent = _make_agent()
        async def inf(): await asyncio.Event().wait()
        task = asyncio.ensure_future(inf())
        agent._bg_tasks["x"] = task
        r = json.loads(_handle_close_task({"task_id": "x"}, parent_agent=agent))
        assert r["success"] is True
        assert "x" not in agent._bg_tasks

    def test_missing_task(self):
        r = json.loads(_handle_close_task({"task_id": "x"}, parent_agent=_make_agent()))
        assert r["success"] is False

class TestListTasks:
    def test_empty(self):
        r = json.loads(_handle_list_tasks({}, parent_agent=_make_agent()))
        assert r == {}

    def test_lists_running(self):
        agent = _make_agent()
        async def inf(): await asyncio.Event().wait()
        agent._bg_tasks["x"] = asyncio.ensure_future(inf())
        r = json.loads(_handle_list_tasks({}, parent_agent=agent))
        assert "x" in r
        assert r["x"]["done"] is False
```

**Commit:** `git commit -m "feat: add close_task and list_tasks tools (ops channel)"`

---

### Task 6: chat() loop transformation — auto-injection + persistent loop + multi-source wait (Gap 1 core)

**Objective:** The core architectural change. Before each LLM turn: drain EventCollector AND _user_queue, inject as user messages. After model produces final text: if children are running, wait on BOTH collector and _user_queue simultaneously.

**Files:** Modify `tyagent/agent.py`

**Step 1: Add the auto-injection block BEFORE the while True loop (and at the top of each iteration)**

Insert just before `api_messages = list(messages)` and again at the top of the while loop body:

```python
        # ── Auto-inject events ──────────────────────────────────────
        # Before each LLM turn, drain child completions and queued
        # user messages into the conversation.
        if self._event_collector is not None:
            child_events = self._event_collector.drain_completed()
            for event in child_events:
                task_id = event["task_id"]
                result = event["result"]
                summary = result.get("summary", "")
                if summary:
                    inject = f"## Subagent `{task_id}` completed\n\n{summary}"
                elif result.get("success"):
                    inject = f"## Subagent `{task_id}` completed successfully"
                else:
                    inject = f"## Subagent `{task_id}` failed\n\n{result.get('error', 'Unknown')}"
                messages.append({"role": "user", "content": inject})
                logger.info("Auto-injected child result for %s", task_id)

        # Drain queued user messages (from gateway while children ran)
        while not self._user_queue.empty():
            try:
                queued = self._user_queue.get_nowait()
                messages.append(queued)
                logger.info("Drained queued user message into conversation")
            except asyncio.QueueEmpty:
                break
```

**Step 2: Replace the final return with persistent wait loop**

Replace `return (content or reasoning_content or "")` (line ~353) with:

```python
            # No tool calls -> final answer
            if not tool_calls:
                content_text = (content or reasoning_content or "")

                # ── Persistent loop: wait for events ─────────────────
                # If children are running or events are pending, don't
                # return — wait for the next event and continue.
                if self._bg_tasks or (self._event_collector and self._event_collector.peek()):
                    # Ensure collector exists
                    if self._event_collector is None:
                        self._event_collector = EventCollector()

                    logger.info(
                        "Model finished but %d subagent(s) running — waiting for next event",
                        len(self._bg_tasks),
                    )

                    # Multi-source wait: child completion OR user message
                    child_done_task = asyncio.create_task(
                        self._event_collector.wait_next(timeout=SUBAGENT_PERSIST_TIMEOUT)
                    )
                    user_msg_task = asyncio.create_task(
                        self._user_queue.get()
                    )

                    done, pending = await asyncio.wait(
                        [child_done_task, user_msg_task],
                        return_when=FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

                    if child_done_task in done and child_done_task.result():
                        # Child completed — inject and re-enter LLM
                        logger.info("Child completed while waiting — re-entering LLM")
                        continue

                    if user_msg_task in done:
                        # User sent new message — inject and re-enter LLM
                        logger.info("User message received while waiting — re-entering LLM")
                        continue

                    # Timeout — return current answer
                    logger.warning("Persist timeout — returning current answer")
                    return content_text

                return content_text
```

Add constant:
```python
SUBAGENT_PERSIST_TIMEOUT = 120  # seconds
```

**Step 3: Async-aware tool dispatch**

Replace the dispatch section in the tool execution loop (around line ~391-399):

```python
                # ── Async-aware dispatch ──────────────────────────────
                # spawn_task and wait_task handlers run via run_in_executor
                # and internally use asyncio.get_event_loop() to bridge
                # thread→async. close_task/list_tasks use same path.
                result = await loop.run_in_executor(
                    None, registry.dispatch, func_name, func_args, self,
                )
```

No special-casing needed because all four async tools (`spawn_task`, `wait_task`, `close_task`, `list_tasks`) use the same bridge pattern: `asyncio.get_event_loop()` inside the handler.

**Tests:** Verify auto-injection, persistent loop, multi-source wait (with `_user_queue` events).

**Commit:** `git commit -m "feat: transform chat() loop with auto-injection, persistent wait, multi-source"`

---

### Task 7: Gateway modifications — enqueue messages for active sessions (Gap 1)

**Objective:** When `_on_message` detects a session is already in `_active_sessions`, enqueue the message to the agent's `_user_queue` instead of discarding it.

**Files:** Modify `tyagent/gateway/gateway.py`

**Step 1: Add active-session check at the beginning of processing**

Insert right after `session_key = adapter.build_session_key(event)` but before the draining check (or right after it, but definitely before `self._active_sessions.add(session_key)`):

```python
        session_key = adapter.build_session_key(event)

        # ── If session is active, enqueue message instead of blocking ──
        if session_key in self._active_sessions:
            logger.info(
                "Session %s is active — enqueuing message to agent queue",
                session_key,
            )
            agent = self._agent_cache.get(session_key)
            if agent is not None:
                await agent._user_queue.put(
                    {"role": "user", "content": event.text}
                )
                # Persist message to DB as well
                session = self.session_store.get(session_key)
                if session:
                    session.add_message("user", event.text)
                # Send a quick acknowledgement
                await adapter.send_message(
                    event.chat_id or "",
                    "📩 收到，正在处理中...",
                    reply_to_message_id=event.message_id,
                )
            else:
                # Fallback: agent not found (shouldn't happen)
                logger.warning("Agent not found for active session %s", session_key)
            return None
```

**Step 2: Add `_get_agent` helper**

```python
    def _get_agent(self, session_key: str) -> Optional[TyAgent]:
        """Get cached agent without creating."""
        return self._agent_cache.get(session_key)
```

**Step 3: Handle session cleanup**

When the active session completes (in the `finally` block), check if there are queued messages that should trigger a new chat():

```python
        finally:
            self._active_sessions.discard(session_key)
            self._active_chat_ids.pop(session_key, None)
            self._session_to_adapter.pop(session_key, None)

            # If user messages were queued during chat(), process the next one
            agent = self._agent_cache.get(session_key)
            if agent is not None and not agent._user_queue.empty():
                # Fire-and-forget: process the next queued message
                asyncio.create_task(
                    self._process_queued_next(session_key, agent, adapter)
                )
```

Add the helper:
```python
    async def _process_queued_next(
        self, session_key: str, agent: TyAgent, adapter: BasePlatformAdapter,
    ) -> None:
        """Process the next queued user message for an active session."""
        try:
            queued = await agent._user_queue.get()
            # Create a synthetic MessageEvent
            event = MessageEvent(
                platform=adapter.platform_name,
                chat_id=self._active_chat_ids.get(session_key, ""),
                user_id="", text=queued.get("content", ""),
                message_type=MessageType.USER,
            )
            await self._handle_message_event(event)
        except Exception:
            logger.exception("Failed to process queued message for %s", session_key)
```

**Optimization:** This `_process_queued_next` creates a synthetic event to re-enter `_handle_message_event`. The downside is that the new chat() call blocks again. A cleaner approach: just let the user's next message (which they'll naturally send after receiving the "processing" ack) re-enter `_handle_message_event` naturally. The queue draining happens inside chat() itself.

Simplified approach: **don't auto-trigger the next chat.** Let the user's follow-up message (which they'll see as part of the conversation) naturally re-enter gateway processing. The queue acts as a buffer that chat() drains on each turn.

So the finally block stays as-is:
```python
        finally:
            self._active_sessions.discard(session_key)
            # No auto-trigger — user's next message handles it
```

**Tests:** Verify enqueue behavior, acknowledge message sent, session correctly tracked.

**Commit:** `git commit -m "feat: enqueue user messages for active sessions instead of blocking"`

---

### Task 8: Update delegate_task as spawn+wait backward compat wrapper

**Objective:** `delegate_task` internally does spawn+immediate-wait. Exact same behavior as before.

**Files:** Modify `tyagent/tools/delegate_tool.py`

**Step 1: Rewrite handler**

```python
def _handle_delegate_task(args: Dict[str, Any], parent_agent: Any = None) -> str:
    """Convenience wrapper: spawn_task + immediate wait_task (flattened result)."""
    spawn_args = {k: v for k, v in args.items() if k not in ("task_ids", "timeout")}
    spawn_result = json.loads(_handle_spawn_task(spawn_args, parent_agent=parent_agent))
    if "error" in spawn_result:
        return json.dumps(spawn_result, ensure_ascii=False)
    task_id = spawn_result["task_id"]
    wait_result = json.loads(
        _handle_wait_task({"task_ids": [task_id]}, parent_agent=parent_agent)
    )
    single = wait_result.get(task_id, {"error": "Unknown error"})
    return json.dumps(single, ensure_ascii=False)
```

**Tests:** Backward compat regression. All existing delegate_task tests should pass with mocks on `_run_child_async`.

**Commit:** `git commit -m "refactor: delegate_task as spawn+wait wrapper (backward compat)"`

---

### Task 9: Update DELEGATE_BLOCKED_TOOLS

**Objective:** Block all async subagent tools from children.

**Files:** Modify `tyagent/tools/delegate_tool.py`

```python
DELEGATE_BLOCKED_TOOLS = frozenset(
    ["delegate_task", "spawn_task", "wait_task", "close_task", "list_tasks", "memory"]
)
```

**Commit:** `git commit -m "fix: block all async subagent tools from children"`

---

### Task 10: Clean up old code — remove _run_child_sync, old patterns

**Objective:** Remove sync-only code. Update tests that reference removed functions.

**Files:** Modify `tyagent/agent.py`, `tyagent/tools/delegate_tool.py`, `tests/test_delegate_tool.py`

- Remove `_run_child_sync` from delegate_tool.py
- Remove `_run_child_in_thread` if it was added
- Remove any ThreadPoolExecutor imports
- Add `TestRunChildAsync` class covering successful/failed child runs
- Update `_make_agent` mock to support `_bg_tasks: Dict[str, asyncio.Task]`

**Commit:** `git commit -m "refactor: remove old sync-only code"`

---

### Task 11: Integration tests + full regression

**Objective:** End-to-end tests verifying the complete spawn→work→wait→inject→interleave flow.

**Files:** Create `tests/test_async_subagent.py`

Key test scenarios:
1. **Basic flow:** spawn_task → wait_task returns child result
2. **Multiple children:** spawn 3 → wait all → all collected
3. **Auto-injection:** child completes while waiting → result injected as user message
4. **User message interleaving:** child runs → user sends message → chat() picks it up
5. **close_task:** spawn → close before completion → task cancelled
6. **list_tasks:** spawn → list → verify status
7. **Persistent loop:** model finishes with children running → chat() stays alive → child completes → new turn
8. **delegate_task backward compat:** unchanged behavior
9. **Full regression:** `pytest tests/ -q` → all pass

**Commit:** `git commit -m "test: integration tests for async subagent architecture"`

---

## Architecture Alignment Table (vs Codex)

| Feature | Codex | This Plan | Status |
|---|---|---|---|
| spawn non-blocking | ✅ | ✅ | Task 3 |
| Child = async task | ✅ (tokio) | ✅ (asyncio) | Task 3 |
| Auto-inject child result | ✅ (mailbox → user msg) | ✅ (EventCollector.drain → user msg) | Task 6 |
| Injection triggers new LLM turn | ✅ (select! → run_turn) | ✅ (persistent while loop continues) | Task 6 |
| chat() = event loop (not one-shot) | ✅ (agent thread forever) | ✅ (stays alive while children run) | Task 6 |
| User input during children | ✅ (select! over user + child) | ✅ (asyncio.wait over collector + user_queue) | Task 6+7 |
| wait = explicit blocking tool | ✅ | ✅ | Task 4 |
| Parent→Child ops | ✅ (tx_ops) | ✅ (close_task + list_tasks) | Task 5 |
| Gateway enqueues vs discards | ✅ (incoming on channel) | ✅ (enqueue to _user_queue) | Task 7 |
| trigger_turn control | ✅ | ❌ post-MVP | — |
| send_agent_message | ✅ | ❌ post-MVP | — |
| Multi-adapter parallel sessions | — (single agent) | ✅ (arch ready via _user_queue) | Task 7 |

---

## Risk Areas

1. **`asyncio.get_event_loop()` from `run_in_executor` thread**: Works in Python 3.10+ for threads spawned while a loop is running. If not, fallback: pass loop explicitly or use `asyncio.run_coroutine_threadsafe()`.

2. **`_user_queue` drain timing**: If gateway enqueues a message while chat() is in the API call (not in wait loop), the message sits in the queue until the next wait point. This is OK — same as Codex where messages sit in a channel until select! picks them up.

3. **Session→agent mapping**: Gateway already has `_agent_cache` dict. Accessing it to get the agent for an active session is safe because the chat() holds the only reference.
