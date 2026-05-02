# Final Plan: TyAgent → Codex-Aligned Architecture

> **Principle:** Never design a two-pipe system when a one-pipe system with structured messages suffices.

## Architecture at a Glance

```
                       ┌──────────────┐
  gateway ──put──▶     │  _inbox      │  (InboxMessage: text + reply_target)
                       │              │
                       ▼              │
              ┌────────────────────┐  │
              │  _agent_loop()     │  │  select! { inbox, collector, stop }
              │  while running:    │  │
              │    select!{        │  │
              │      inbox.get()   │──┘
              │      collector     │
              │      stop_event    │
              │    }               │
              │    → inject child  │
              │    → _run_turn()   │
              │    → output.put()  │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  _output_queue     │  (AgentOutput: text + reply_target)
              │                    │
              ▼                    │
  ┌──────────────────────┐         │
  │  _consume_output()   │─────────┘  gateway 长期消费者
  │  while running:      │
  │    output = queue.get│──▶ adapter.send_message(...)
  │    routing logic     │
  └──────────────────────┘
```

**两个队列，一个输出。没有future绑定，没有auto_response_queue。**

---

## Key Design Decisions

### 1. Gateway send_message = fire-and-forget

```python
async def send_message(self, text: str, reply_target: ReplyTarget) -> None:
    """Put message to inbox. Don't wait for response."""
    await self._inbox.put(InboxMessage(text=text, reply_target=reply_target))
```

不再返回future。gateway_send_message只负责把消息放进inbox，然后立即返回。回复从_output_queue消费。

### 2. 回复 = 结构化消息，不是裸字符串

```python
@dataclass
class InboxMessage:
    text: str
    reply_target: ReplyTarget

@dataclass
class AgentOutput:
    text: str
    reply_target: Optional[ReplyTarget]  # 有=回复到指定对话; 无=子代理触发的自动回复
```

两个类结构一致。**输入和输出是对称的**——都有路由信息。

### 3. Agent loop = 转换器，不关心路由

```python
async def _agent_loop(self):
    while self._running:
        inbox_task = asyncio.create_task(self._inbox.get())
        child_task = asyncio.create_task(self._event_collector.wait_next())
        stop_task = asyncio.create_task(self._stop_event.wait())

        done, pending = await asyncio.wait(
            [inbox_task, child_task, stop_task],
            return_when=FIRST_COMPLETED,
        )
        for t in pending: t.cancel()

        if stop_task in done: break

        # Inject child completions
        for event in self._event_collector.drain_completed():
            self._messages.append(user_msg(inject_template(event)))
            reply_target = None  # auto-reply: no route back

        # Process user message  
        current_reply = None
        if inbox_task in done:
            msg = inbox_task.result()
            self._messages.append({"role": "user", "content": msg.text})
            current_reply = msg.reply_target

        # Run LLM turn
        content = await self._run_turn()

        # Single output pipe — agent doesn't know or care where this goes
        await self._output_queue.put(AgentOutput(
            text=content,
            reply_target=current_reply,
        ))
```

Agent loop**只知道**：输入来了 → turn → 输出。它不知道输出去了哪里，不需要知道。reply_target只是从输入消息中复制过来的路由信息——agent本身不关心它。

### 4. Gateway消费者做路由决策

```python
async def _consume_output(self, session_key: str):
    agent = self._session_agents[session_key]
    adapter = self._session_adapter[session_key]
    chat_id = self._session_chat_ids[session_key]

    while self._running:
        output = await agent._output_queue.get()

        if output.reply_target:
            # 用户消息驱动的回复 → 回复到那条消息下面
            await adapter.send_message(
                output.reply_target.chat_id,
                output.text,
                reply_to=output.reply_target.message_id,
            )
        else:
            # 子代理完成触发的回复 → 发新消息
            await adapter.send_message(chat_id, output.text)
```

---

## Task Breakdown (Final)

### Phase 1: Infrastructure (3 days)

| Task | What | Outcome |
|---|---|---|
| 1 | `EventCollector` class | `tyagent/events.py` |
| 2 | TyAgent loop: `_inbox`, `_output_queue`, `start()`, `stop()`, `_agent_loop()` | 常驻loop框架 |
| 3 | Extract `_run_turn()` from `chat()` | 纯LLM+tool循环 |
| 4 | Sub-agent tools: `spawn_task`, `wait_task`, `close_task`, `list_tasks` | 所有工具就绪 |

### Phase 2: Gateway Integration (2 days)

| Task | What | Outcome |
|---|---|---|
| 5 | `InboxMessage`/`AgentOutput` types | 结构化路由 |
| 6 | Gateway: `send_message()` fire-and-forget, `_consume_output()`, `_ensure_session_agent()` | 新架构 |
| 7 | Session store integration (history load, persist callback) | 消息不丢 |

### Phase 3: Compatibility & Quality (2 days)

| Task | What | Outcome |
|---|---|---|
| 8 | `chat()` backward compat (for sub-agents & tests) | 旧代码不坏 |
| 9 | `delegate_task` as spawn+wait wrapper | 向后兼容 |
| 10 | Clean up old code (ThreadPoolExecutor, _run_child_sync) | 整洁 |
| 11 | Integration tests + full regression | 全部通过 |

---

## 和你之前讨论过的Hermes架构设计原理的关系

这个改造本质上是**从"函数调用模型"到"Actor模型"的转变**：

- 当前：`gateway 调 agent.chat() = 函数调用`。agent是被动的、临时的
- 改造后：`gateway 发消息给 agent loop = Actor通信`。agent是自主的、常驻的

这和 Hermes 核心循环的设计哲学一致——gateway 和 agent 之间是**消息通道**的连接，不是**函数调用栈**的连接。agent 拥有自己的生命周期，gateway 通过消息与其通信。这也是你之前对 Hermes 架构设计感兴趣的方向。

要推进执行吗？按 Phase 1→2→3 顺序，每阶段一个SDD循环。
