# tyagent 测试系统重构设计方案

## 问题诊断

现有 581 个测试用例的核心问题：**Mock 边界太深**，全部绕过 agent/gateway 的核心执行路径。

### 现有测试的问题

```
真实路径: 用户 → Gateway → agent.start() → agent loop → _run_turn()
           → HTTP POST → LLM → 解析 → _execute_tool_calls() → 真实工具
           → 再调 LLM → 输出 → _consume_output() → adapter.send_message()

现有测试: MagicMock agent → agent.chat=AsyncMock(return_value="Hi")
           → 断言 "Hi" 被返回 [没有 agent loop, 没有工具分发, 没有错误处理]
```

### 三个致命缺陷

1. **Mock agent._client.post** — 绕过了 `_run_turn()` 的全部逻辑
   （工具调用循环、消息累积、compaction、error handling）
2. **Gateway 测试用 MagicMock agent** — 绕过了 agent.start/send_message/actor model loop
3. **Delegate tool 测试 Mock _run_child_async** — 绕过了真实的子进程 spawn/wait/close

## 新设计：四层测试金字塔

### Layer 1: 纯单元测试（无外部依赖）
- 工具注册、schema 生成、消息 sanitization
- Config 解析、SessionStore CRUD
- 保留现有的好测试（config, db, session, sanitize 等）

### Layer 2: Agent 集成测试（FakeLLM at HTTP boundary）
- **Mock 边界只设在 HTTP 层**
- 真实 TyAgent → 真实 _run_turn() → 真实工具分发 → FakeLLM
- FakeLLM 可编程：设置 "收到 X → 返回 Y"

### Layer 3: Gateway 集成测试（Fake Agent + Fake Adapter）
- 真实 Gateway → 真实 actor model → agent loop → 输出消费
- FakeAdapter：收集发送的消息，不真连飞书
- FakeLLM 在 agent 内部

### Layer 4: 端到端测试（Full stack）
- Gateway + Agent + FakeLLM + TestAdapter
- 模拟：用户发消息 → 整个链路的处理 → 响应到达 adapter

## FakeLLM 设计

```python
class FakeLLM:
    """Programmable LLM that lives at HTTP boundary.

    不是 mock agent._client.post，而是替换 agent._client 为 FakeLLM。
    TyAgent 的全部代码都真实运行。
    """

    def respond(self, text: str):     # 简单文本响应
    def tool_call(self, name, args):  # 工具调用响应
    def error(self, status, msg):     # 错误响应
    def chain(self, *responses):      # 多轮响应序列

    # 内部实现 httpx.AsyncClient 的 post() 和 stream() 接口
```

## FakeAdapter 设计

```python
class FakeAdapter(BasePlatformAdapter):
    """收集消息的测试适配器，不连真实平台。"""
    sent_messages: List[str]       # 所有 send_message 调用的文本
    last_message_id: str
    build_session_key → "test:chat_id"
```

## 测试文件重组

```
tests/
  conftest.py           ← 全局 fixtures (fake_llm, test_agent, test_gateway)
  test_agent_live.py    ← Layer 2: Agent 真实 loop 测试（替代 test_agent.py + test_agent_loop.py）
  test_gateway_live.py  ← Layer 3: Gateway 真实编排测试（替代 test_gateway.py）
  test_delegate_live.py ← Layer 2: 真实子代理 spawn/wait/close
  test_e2e.py           ← Layer 4: 端到端全链路测试
  # 保留并增强的现有文件
  test_config.py        ← 保留（配置解析测试，合理）
  test_session.py       ← 保留（SQLite CRUD 测试，合理）
  test_db.py            ← 保留
  test_compaction.py    ← 需要增强（加真实 LLM 调用的 compaction 测试）
  test_search_tool.py   ← 保留
  test_memory_tool.py   ← 保留
  test_browser_tools.py ← 保留
  test_feishu_*.py      ← 保留（飞书特定逻辑）
```

## 核心测试场景

### Agent 层（关键新测试）
1. 单轮文本对话 — agent 收到消息，调用 LLM，返回文本
2. 工具调用循环 — LLM 要求 read_file → 真实工具执行 → 结果回 LLM → 最终文本
3. 多工具调用 — LLM 同时要求 read_file + search_files
4. 工具错误恢复 — 工具失败 → 错误 JSON 返回 LLM
5. Stream 输出 — stream=True，delta callback 被正确调用
6. Agent 生命周期 — start / send_message / stop / double start
7. Compaction 触发 — 超大消息列表 → 自动压缩
8. 子代理完成触发 turn — mailbox 收到 FinalNotification → agent 自动跑下一轮

### Gateway 层（关键新测试）
1. 消息到达 → 创建 session → agent 处理 → 输出发送
2. /new 命令 → 归档旧 session → 创建新 session
3. /status 命令 → 返回真实状态信息
4. Agent 错误 → 错误消息发送给用户
5. 多 session 并发 — 两个不同 chat_id 同时发消息

### Delegate 工具层
1. spawn_task → 真实 child agent 启动 → 运行 → 完成 → FinalNotification
2. wait_task → block 直到 child 完成 → 收集结果
3. close_task → 停止运行中的 child
4. list_tasks → 列出所有活跃 child

### 端到端
1. 用户发消息 → 整个流程 → response 到达 adapter
2. 工具调用 → 进度通知 → 最终回复
3. 错误恢复全链路
