# TyAgent + Async Subagent — Full Codex Architecture Rethink

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Transform tyagent from a request-response architecture to a Codex-aligned **actor model** with a permanent agent event loop. The agent loop runs `select!` over user messages, child completions, and ops events — exactly like Codex's tokio `select!`. Gateway sends messages to the agent's inbox; the agent replies via outbox/future. Sub-agent tools (spawn, wait, close, list) are first-class citizens driving the event loop.

**Architecture:** TyAgent gets a `start()` → `_agent_loop()` → `stop()` lifecycle. `_agent_loop()` runs `asyncio.wait()` over three event sources:
1. `_inbox.get()` — user messages from gateway
2. `_event_collector.wait_next()` — child agent completions
3. `_stop_event.wait()` — graceful shutdown

Each event triggers `_run_turn()` — extracted from the current `chat()` — which executes the LLM + tool call loop and returns the final text. Sub-agents are `asyncio.Task`s that call `collector.notify_child_done()` on completion.

**Gateway changes:** Instead of `await agent.chat(messages, tools)`, gateway calls `await agent.send_message(text)`. Session history is loaded before starting the loop. Out-of-band child completion replies go through `_auto_response_queue`.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       Gateway (per session)                      │
│                                                                  │
│  _on_message(event)                                              │
│    ├─ session.active? → agent.send_message(text)                 │
│    │                    → await future → reply                   │
│    │                    (initial message starts the loop)        │
│    └─ !session.active? → session_store.get_messages()            │
│                          agent.start(history)                     │
│                          agent.send_message(text) → await reply  │
│                                                                  │
│  _consume_auto_responses()                                       │
│    └─ while running:                                             │
│         response = agent._auto_response_queue.get()               │
│         adapter.send_message(chat_id, response)                  │
└──────────────┬──────────────────────────────────┬────────────────┘
               │ send_message(text)               │ _auto_response_queue
               ▼                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TyAgent._agent_loop()                          │
│                                                                  │
│  while _running:                                                 │
│    select! over:                                                 │
│    ┌─────────────────────────────────────────────────┐           │
│    │  asyncio.wait([                                  │           │
│    │    inbox.get(),           ← 用户消息              │           │
│    │    collector.wait_next(), ← 子代理完成            │           │
│    │    stop_event.wait(),    ← 优雅关闭              │           │
│    │  ], return_when=FIRST_COMPLETED)                │           │
│    └─────────────────────────────────────────────────┘           │
│         │                                                       │
│         ▼                                                       │
│    Drain collector → inject child results as user messages      │
│    Process inbox → append user message to messages              │
│    Run _run_turn()                                              │
│         │                                                       │
│         ▼                                                       │
│    [LLM call] [tool dispatch] [LLM call] ... → final text       │
│         │                                                       │
│         ▼                                                       │
│    Set inbox future result OR push to _auto_response_queue      │
│    Continue loop                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Before / After Comparison

| | Before | After |
|---|---|---|
| Agent lifecycle | Created per request, destroyed after | `start()` → permanent loop → `stop()` |
| Message flow | gateway → `agent.chat(messages)` → return | gateway → `agent.send_message(text)` → future |
| Event loop | Inside chat(), temporary, per turn | Permanent, owned by agent |
| Child complete trigger | Only if chat() is waiting | First-class event source in select! |
| User input during children | Blocked (gateway serial) | Enqueued in inbox, processed in select! |
| Session history | Loaded per chat() call | Loaded once at start(), accumulated in loop |
| chat() function | Monolithic (history + tool loop) | Extracted to `_run_turn()` (pure tool loop) |

---

## Task by Task

### Task 1: TyAgent loop infrastructure (start, stop, inbox, outbox)

**Objective:** Add the actor-model lifecycle to TyAgent.

**Files:** Modify `tyagent/agent.py`

**Step 1: Add instance variables in `__init__`**

```python
# ── Actor model lifecycle ─────────────────────────────────
self._running: bool = False
self._loop_task: Optional[asyncio.Task] = None
self._inbox: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()   # {text, future}
self._auto_response_queue: asyncio.Queue[str] = asyncio.Queue()
self._stop_event: asyncio.Event = asyncio.Event()
# ── Child agent management ────────────────────────────────
self._bg_tasks: Dict[str, asyncio.Task] = {}
self._event_collector: Optional[EventCollector] = None
```

**Step 2: Add `start()`, `stop()`, `send_message()` methods**

```python
async def start(
    self,
    history: Optional[List[Dict[str, Any]]] = None,
    on_message: Optional[OnMessageCallback] = None,
) -> None:
    """Start the permanent agent event loop.

    Args:
        history: Initial messages to load (from session store).
        on_message: Callback for message persistence (same as chat()'s).
    """
    if self._running:
        logger.warning("Agent loop already running")
        return

    self._messages = list(history) if history else []
    self._on_message = on_message
    self._running = True
    self._stop_event.clear()
    self._loop_task = asyncio.create_task(self._agent_loop())
    logger.info("Agent loop started")

async def stop(self) -> None:
    """Stop the agent loop and clean up."""
    if not self._running:
        return
    self._running = False
    self._stop_event.set()

    # Cancel running children
    for tid, task in list(self._bg_tasks.items()):
        if not task.done():
            task.cancel()
    if self._bg_tasks:
        await asyncio.gather(*self._bg_tasks.values(), return_exceptions=True)
    self._bg_tasks.clear()

    # Wait for loop to stop
    if self._loop_task is not None:
        try:
            await asyncio.wait_for(self._loop_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None
    logger.info("Agent loop stopped")

async def send_message(self, text: str) -> str:
    """Send a user message to the agent loop and wait for the response.

    Returns the agent's final text response.
    """
    if not self._running:
        raise RuntimeError("Agent loop is not running. Call start() first.")

    future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    await self._inbox.put({"text": text, "future": future})
    return await future
```

**Step 3: Add `_agent_loop()` skeleton**

```python
async def _agent_loop(self) -> None:
    """Permanent agent event loop — equivalent to Codex's tokio select!.

    Event sources:
    1. inbox.get()       — user messages from gateway
    2. collector.wait_next() — child agent completion
    3. stop_event.wait() — graceful shutdown
    """
    loop = asyncio.get_event_loop()
    messages = self._messages  # Mutable list shared across turns
    current_response_future: Optional[asyncio.Future[str]] = None
    tools = registry.get_definitions()  # Loaded once

    while self._running:
        # ── 1. Select! over event sources ────────────────────
        inbox_task = loop.create_task(self._inbox.get())
        child_task = loop.create_task(self._event_collector.wait_next())
        stop_task = loop.create_task(self._stop_event.wait())

        done, pending = await asyncio.wait(
            [inbox_task, child_task, stop_task],
            return_when=FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        # ── 2. Check for stop signal ─────────────────────────
        if stop_task in done:
            break

        # ── 3. Drain completed child events ──────────────────
        # Inject child results as user messages before processing
        # any other event. This ensures the model always sees
        # the latest child results.
        if self._event_collector is not None:
            child_events = self._event_collector.drain_completed()
            for event in child_events:
                summary = event["result"].get("summary", "")
                if summary:
                    inject = f"## Subagent `{event['task_id']}` completed\n\n{summary}"
                elif event["result"].get("success"):
                    inject = f"## Subagent `{event['task_id']}` completed successfully"
                else:
                    inject = f"## Subagent `{event['task_id']}` failed\n\n{event['result'].get('error', 'Unknown')}"
                messages.append({"role": "user", "content": inject})

        # ── 4. Process inbox message ─────────────────────────
        if inbox_task in done:
            inbox_msg = inbox_task.result()
            messages.append({"role": "user", "content": inbox_msg["text"]})
            current_response_future = inbox_msg["future"]

        # ── 5. Run turn ──────────────────────────────────────
        content = await self._run_turn(messages, tools=tools)

        # ── 6. Deliver response ──────────────────────────────
        if current_response_future is not None and not current_response_future.done():
            current_response_future.set_result(content)
            current_response_future = None
        elif child_task in done and child_task.result():
            # Child completion triggered this turn with no user message
            # → push to auto-response queue for out-of-band delivery
            await self._auto_response_queue.put(content)
```

**Constants:**
```python
SUBAGENT_PERSIST_TIMEOUT = 120  # Max seconds waiting for children
```

**Commit:** `git commit -m "feat: add actor-model lifecycle to TyAgent (start/stop/loop)"`

---

### Task 2: Extract _run_turn() from chat()

**Objective:** `_run_turn(messages, tools) → str` runs one complete LLM + tool-calling cycle and returns the final text. This is the core loop extracted from `chat()`.

**Files:** Modify `tyagent/agent.py`

**Step 1: Add `_run_turn()` method**

This is the LLM request + tool call execution loop from the current `chat()`, extracted as a standalone method:

```python
async def _run_turn(
    self,
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Run one turn: LLM call + tool call execution until final text.

    This is the core loop extracted from chat().
    messages is mutated in place (appended with assistant + tool results).
    """
    if self._system_msg is None:
        self._system_msg = {"role": "system", "content": self.system_prompt}
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, self._system_msg)

    self._prev_msg_count = 0
    self._token_history = []
    api_messages = list(messages)
    self._prev_msg_count = len(messages)

    payload_base: Dict[str, Any] = {
        "model": self.model,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    if self.reasoning_effort:
        payload_base["reasoning_effort"] = self.reasoning_effort
    if tools:
        payload_base["tools"] = tools
        payload_base["tool_choice"] = "auto"

    tool_turn = 0
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls = None

    while True:
        if self.max_tool_turns is not None and self.max_tool_turns > 0 and tool_turn >= self.max_tool_turns:
            logger.warning("Max tool turns (%d) reached", self.max_tool_turns)
            break

        # Append-only: add new messages since last API call
        if tool_turn > 0:
            api_messages.extend(messages[self._prev_msg_count:])
        self._prev_msg_count = len(messages)
        payload_base["messages"] = api_messages

        # Context overflow retry loop
        _compressed = False
        while True:
            self.last_usage = None
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "KimiCLI/1.30.0",
                    },
                    json=payload_base,
                )
                if resp.status_code >= 400:
                    body_str = resp.text[:2000]
                    if _is_context_overflow(resp.status_code, body_str):
                        raise ContextOverflow(body_str)
                    raise AgentError(f"LLM API returned {resp.status_code}: {body_str}")
                data = resp.json()
                usage = data.get("usage")
                if usage:
                    self.last_usage = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                reasoning_content = message.get("reasoning_content")
                break

            except ContextOverflow:
                if not _compressed:
                    _compressed = True
                    compressed = await compress_context(...)
                    if compressed is not None:
                        api_messages = compressed
                        self._prev_msg_count = len(messages)
                        self._token_history = []
                        payload_base["messages"] = api_messages
                        continue
                raise AgentError("Context too long even after compression.")

        # Record token history
        if self.last_usage and self.last_usage.get("prompt_tokens"):
            self._token_history.append(
                (len(api_messages), self.last_usage["prompt_tokens"])
            )

        # Append assistant message
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if content is not None:
            assistant_msg["content"] = content
        if reasoning_content is not None:
            assistant_msg["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Persist via callback
        if self._on_message:
            msg_kwargs: Dict = {}
            if tool_calls:
                msg_kwargs["tool_calls"] = tool_calls
            if reasoning_content:
                msg_kwargs["reasoning"] = reasoning_content
            self._on_message("assistant", content or "", **msg_kwargs)

        # No tool calls → final answer
        if not tool_calls:
            return (content or reasoning_content or "")

        # Execute tool calls
        tool_turn += 1
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            func = tc.get("function", {}) or {}
            func_name = func.get("name", "")
            func_args_str = func.get("arguments", "")

            if not func_name:
                messages.append({"role": "tool", "tool_call_id": tc_id,
                                 "content": json.dumps({"error": "Malformed tool call"})})
                continue

            try:
                func_args = json.loads(func_args_str) if func_args_str else {}
            except json.JSONDecodeError:
                messages.append({"role": "tool", "tool_call_id": tc_id,
                                 "content": json.dumps({"error": f"Invalid JSON: {func_args_str}"})})
                continue

            logger.info("  ⚡ %s(...)", func_name)
            if self._tool_progress_callback:
                try: self._tool_progress_callback(func_name, func_args)
                except Exception: pass

            # Dispatch via run_in_executor (sync handlers + async-bridge handlers)
            _loop = asyncio.get_running_loop()
            result = await _loop.run_in_executor(
                None, registry.dispatch, func_name, func_args, self,
            )
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            if self._on_message:
                self._on_message("tool", result, tool_call_id=tc_id)

    return (content or reasoning_content or "")
```

**Step 2: Simplify `chat()` to be a backward-compat wrapper**

```python
async def chat(self, messages, *, tools=None, stream=False, on_message=None,
               stream_delta_callback=None, on_segment_break=None,
               reasoning_callback=None, tool_progress_callback=None) -> str:
    """Backward-compatible one-shot chat session.

    This is equivalent to calling start() + send_message() but
    creates a temporary event loop. Mainly used by sub-agent children
    and existing tests.

    For the permanent agent loop (actor model), use:
      await agent.start(history=history)
      response = await agent.send_message("hello")
      await agent.stop()
    """
    # Store callbacks (used by _run_turn)
    self._tool_progress_callback = tool_progress_callback
    self._on_message = on_message

    # Run as a one-shot _run_turn (no permanent loop)
    self._event_collector = EventCollector()
    result = await self._run_turn(messages, tools=tools)

    self._tool_progress_callback = None
    self._on_message = None
    return result
```

**Commit:** `git commit -m "refactor: extract _run_turn() from chat(); chat() becomes backward-compat wrapper"`

---

### Task 3: Sub-agent tools integration with agent loop (spawn, wait, close, list)

**Objective:** Sub-agent tools store/retrieve children from `agent._bg_tasks`, notify `agent._event_collector`, and work transparently whether called from the permanent loop or one-shot `chat()`.

**Files:** Modify `tyagent/tools/delegate_tool.py`

**Step 1: `_run_child_async` — async child agent as asyncio task**

```python
async def _run_child_async(
    task_id: str,
    model: str, api_key: str, base_url: str,
    system_prompt: str, reasoning_effort: Optional[str],
    goal: str, tool_names: List[str], max_tool_turns: int,
    context: Optional[str], compression: Any = None,
    tool_progress_callback: Any = None,
    collector: Optional[EventCollector] = None,
) -> Dict[str, Any]:
    """Run a child agent. Notifies collector on completion. Never raises."""
    t0 = time.monotonic()
    child_system = system_prompt
    if context:
        child_system = f"{system_prompt}\n\nTask context: {context}"

    child = TyAgent(model=model, api_key=api_key, base_url=base_url,
                    max_tool_turns=max_tool_turns, system_prompt=child_system,
                    reasoning_effort=reasoning_effort, compression=compression)
    child_messages = [{"role": "user", "content": goal}]
    tool_defs = registry.get_definitions(names=tool_names)

    try:
        summary = await asyncio.wait_for(
            child.chat(child_messages, tools=tool_defs,
                        tool_progress_callback=tool_progress_callback),
            timeout=600.0,
        )
        result = {"success": True,
                  "summary": (summary.strip() if summary else ""),
                  "error": None,
                  "duration_seconds": round(time.monotonic() - t0, 2)}
    except asyncio.TimeoutError:
        result = {"success": False, "summary": None,
                  "error": "Child timed out after 600s",
                  "duration_seconds": round(time.monotonic() - t0, 2)}
    except BaseException as exc:
        result = {"success": False, "summary": None,
                  "error": f"{type(exc).__name__}: {exc}",
                  "duration_seconds": round(time.monotonic() - t0, 2)}
    finally:
        try: await child.close()
        except Exception: pass

    if collector is not None:
        collector.notify_child_done(task_id, result)
    return result
```

**Step 2: `spawn_task` handler**

```python
import uuid

def _handle_spawn_task(args, parent_agent=None):
    goal = args.get("goal", "").strip()
    if not goal:
        return tool_error("goal is required for spawn_task.")
    if parent_agent is None:
        return tool_error("spawn_task requires a session agent.")

    # ... validation (context, toolsets, max_tool_turns) ...

    task_id = str(uuid.uuid4())[:8]

    # Lazy init collector
    if parent_agent._event_collector is None:
        parent_agent._event_collector = EventCollector()

    child_coro = _run_child_async(
        task_id=task_id, ...,
        collector=parent_agent._event_collector,
    )

    loop = asyncio.get_event_loop()
    child_task = loop.create_task(child_coro)
    parent_agent._bg_tasks[task_id] = child_task

    return json.dumps({"task_id": task_id, "status": "running"}, ensure_ascii=False)
```

**Step 3: `wait_task`, `close_task`, `list_tasks` handlers**

Same design as V3 plan — `asyncio.wait()` in `wait_task`, `task.cancel()` in `close_task`, iterate `_bg_tasks` in `list_tasks`.

**Step 4: Registration**

Register all four tools:
- `spawn_task` 🚀
- `wait_task` ⏳
- `close_task` 🛑
- `list_tasks` 📋

**Step 5: `DELEGATE_BLOCKED_TOOLS`** includes all four + `delegate_task` + `memory`.

**Commit:** `git commit -m "feat: integrate sub-agent tools (spawn/wait/close/list) with agent loop"`

---

### Task 4: Gateway transformation — actor model integration

**Objective:** Gateway starts a permanent agent loop per session, sends messages via `send_message()`, and consumes the auto-response queue for out-of-band child-completion replies.

**Files:** Modify `tyagent/gateway/gateway.py`

**Step 1: Session→agent management in Gateway.__init__**

```python
# In __init__:
self._session_agents: Dict[str, TyAgent] = {}  # session_key → TyAgent
self._session_tasks: Dict[str, asyncio.Task] = {}  # session_key → consumer task
```

**Step 2: Add session agent lifecycle methods**

```python
async def _ensure_session_agent(
    self, session_key: str, session,
) -> TyAgent:
    """Get or create the agent for a session, starting its loop."""
    if session_key in self._session_agents:
        return self._session_agents[session_key]

    agent = TyAgent.from_config(self.config.agent)
    history = session.messages  # Load existing messages

    # Start the permanent agent loop
    def persist_message(role, content, **extras):
        self.session_store.add_message(
            session_key, role, content,
            session_id=..., **extras,
        )
    await agent.start(history=history, on_message=persist_message)

    self._session_agents[session_key] = agent
    return agent


async def _stop_session_agent(self, session_key: str) -> None:
    """Stop and clean up a session's agent."""
    agent = self._session_agents.pop(session_key, None)
    if agent is not None:
        consumer = self._session_tasks.pop(session_key, None)
        if consumer:
            consumer.cancel()
        await agent.stop()


async def _consume_auto_responses(
    self, session_key: str, agent: TyAgent,
    adapter: BasePlatformAdapter, chat_id: str,
) -> None:
    """Consume out-of-band agent responses (child-completion triggered)."""
    try:
        while self._running:
            response = await agent._auto_response_queue.get()
            await adapter.send_message(chat_id, response)
    except asyncio.CancelledError:
        pass
```

**Step 3: Transform `_handle_message_event`**

Replace the `async def _handle_message_event` body:

```python
    async def _handle_message_event(self, event: MessageEvent) -> Optional[str]:
        adapter = self._find_adapter_for_event(event)
        if not adapter:
            return None

        session_key = adapter.build_session_key(event)
        if self._draining:
            await adapter.send_message(event.chat_id or "",
                                       "⏳ Gateway is restarting. Please try again shortly.",
                                       reply_to_message_id=event.message_id)
            return None

        # Session management (existing: suspended/archived logic stays)
        ...

        # Build tool definitions
        tool_defs = registry.get_definitions()

        # ── Ensure agent loop is running ──────────────────────
        agent = await self._ensure_session_agent(session_key, session)

        # ── Send message and wait for response ────────────────
        try:
            text = event.text
            # ... (media handling same as before) ...
            response = await agent.send_message(text)
        except Exception as exc:
            logger.exception("Agent error for %s", session_key)
            response = f"❌ 错误: {exc}"
```

**Step 4: Cleanup on restart/drain**

In the supervisor's cleanup logic, call `_stop_session_agent` for all sessions.

**Commit:** `git commit -m "feat: transform gateway to actor model with permanent agent loop"`

---

### Task 5: Session store integration

**Objective:** Agent loop loads history at start, persists messages via `_on_message` callback during `_run_turn()`.

**Files:** Modify `tyagent/agent.py`, `tyagent/gateway/gateway.py`

**Step 1: Confirm `_on_message` callback in `_run_turn()`**

Already included in Task 2's `_run_turn()`:
```python
if self._on_message:
    self._on_message("assistant", content or "", ...)
```

**Step 2: Gateway passes persist callback to `agent.start()`**

```python
def persist_message(role, content, **extras):
    self.session_store.add_message(session_key, role, content, ...)

await agent.start(history=session.messages, on_message=persist_message)
```

**Step 3: Auto-response messages also persisted**

In `_consume_auto_responses`:
```python
response = await agent._auto_response_queue.get()
self.session_store.add_message(session_key, "assistant", response, ...)
await adapter.send_message(chat_id, response)
```

**Commit:** `git commit -m "feat: integrate session store with agent loop lifecycle"`

---

### Task 6: EventCollector + tests

**Objective:** Same as V3 Task 1 — the EventCollector class.

**Files:** Create `tyagent/events.py`

Same implementation as V3 plan. Full test coverage.

**Commit:** `git commit -m "feat: EventCollector for bridging child completion to parent loop"`

---

### Task 7: Backward compatibility — delegate_task wrapper, existing tests

**Objective:** `delegate_task` becomes spawn+wait wrapper. Existing tests pass without changes (mocking `_run_child_async`).

**Files:** Modify `tyagent/tools/delegate_tool.py`, `tests/test_delegate_tool.py`

**Step 1: delegate_task handler**

```python
def _handle_delegate_task(args, parent_agent=None):
    spawn_args = {k: v for k, v in args.items() if k not in ("task_ids", "timeout")}
    spawn_result = json.loads(_handle_spawn_task(spawn_args, parent_agent=parent_agent))
    if "error" in spawn_result:
        return json.dumps(spawn_result)
    task_id = spawn_result["task_id"]
    wait_result = json.loads(_handle_wait_task({"task_ids": [task_id]}, parent_agent=parent_agent))
    single = wait_result.get(task_id, {"error": "Unknown"})
    return json.dumps(single)
```

**Step 2: Update test mocks**

Existing tests mock `_run_child_sync` — change to mock `_run_child_async`.

**Step 3: Verify `chat()` backward compat**

Existing tests that call `agent.chat(messages)` directly should still work (Task 2's `chat()` wrapper delegates to `_run_turn()`).

**Commit:** `git commit -m "refactor: delegate_task as spawn+wait wrapper; update tests"`

---

### Task 8: Clean up old code — remove ThreadPoolExecutor, old patterns

**Objective:** Remove any ThreadPoolExecutor remnants, old `_run_child_sync`, ensure code is clean.

**Files:** `tyagent/agent.py`, `tyagent/tools/delegate_tool.py`

**Commit:** `git commit -m "chore: remove old sync-only patterns"`

---

### Task 9: Integration tests + full regression

**Files:** Create `tests/test_agent_loop.py`

Key scenarios:
1. **Basic flow:** `start()` → `send_message("hello")` → response returned
2. **Child spawn + wait:** `send_message("search X")` → model spawns → waits → returns result
3. **Child completion triggers new turn (no user message):** model spawns children, they complete, auto-response sent
4. **User input during children:** message 1 spawns child → message 2 queued → processed by agent loop
5. **close_task:** spawn → close → child cancelled
6. **list_tasks:** spawn → list → verify status
7. **Multi-turn conversation:** send_message × 3, verify thread maintained
8. **Agent stop:** stops cleanly, cancels children, no leaks

Full regression: `pytest tests/ -q` → all pass

**Commit:** `git commit -m "test: integration tests for agent loop architecture"`

---

## Alignment vs Codex (Final)

| Feature | Codex | tyagent Final | Status |
|---|---|---|---|
| Permanent agent event loop | ✅ | ✅ | Task 1 |
| `select!` over event sources | ✅ (tokio) | ✅ (asyncio.wait) | Task 1 |
| User messages via channel | ✅ (rx_input) | ✅ (_inbox queue) | Task 1 |
| Child completion as source | ✅ (rx_comm) | ✅ (_event_collector) | Task 1+6 |
| Ops channel (close/list) | ✅ (rx_ops) | ✅ (close_task+list_tasks) | Task 3 |
| Child completion triggers turn | ✅ | ✅ | Task 1 loop |
| Gateway = actor, not call | ✅ | ✅ | Task 4 |
| Extra reply channel | ✅ (msg stream) | ✅ (_auto_response_queue) | Task 1+4 |
| Session history load | ✅ | ✅ | Task 5 |
| trigger_turn control | ✅ | ❌ post-MVP | — |
| send_agent_message | ✅ | ❌ post-MVP | — |

**One remaining difference:** Codex's select! can **accept events from the same source while processing another** (events go to mailbox/channel, dequeued by select!). tyagent's implementation processes one event fully (runs _run_turn to completion) before selecting the next. This means if 3 children complete within milliseconds of each other, they're all drained together before the next select!. This is functionally equivalent but slightly different in timing — closer to Codex's behavior than any of the previous designs, since the loop never exits.
