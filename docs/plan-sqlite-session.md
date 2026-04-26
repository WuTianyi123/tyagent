# 重构计划：SQLite 会话存储 + 新压缩策略

> 创建：2026-04-24
> 状态：待实施

---

## 背景

当前 tyagent 的会话管理基于 JSON 文件：每个 session 一个 JSON，启动时全量加载到内存，
每次消息都完整写回磁盘。随着对话增长，问题逐渐显现：

1. **读写开销**：长对话每次保存要序列化整个消息列表
2. **无法按需查询**：要么全加载、要么不加载，没有中间地带
3. **压缩与存储耦合**：当前的 `context.py` 在内存里做压缩，依赖 `session.messages` 这个完整列表存在内存中

同时，经过讨论确定了新的压缩策略：**保留最后一个 user 消息之后的全部 tool 调用链，丢掉之前的 tool 消息；assistant 的文本回复和 user 消息始终保留。** 这比 Hermes 的辅助 LLM 摘要方案更轻量、更确定，因为 assistant 回复本身就是 tool 结果的自然摘要。

## 目标

将 session 存储从 JSON 文件迁移到 SQLite，同时实现新的压缩策略，为 tyagent 的全功能定位打下基础。

---

## 架构设计

### 核心原则

- **消息永不删除**：SQLite 是唯一的消息源，所有原始消息完整保留
- **发送时压缩**：构建 API 消息列表时按策略过滤，不动 DB 里的数据
- **接口不变**：gateway 和 agent 的调用方式尽量保持一致，减少上层改动

### 数据模型

```sql
-- sessions 表：会话元信息
CREATE TABLE sessions (
    session_key TEXT PRIMARY KEY,
    created_at  REAL NOT NULL DEFAULT (strftime('%s','now')),
    updated_at  REAL NOT NULL DEFAULT (strftime('%s','now')),
    metadata    TEXT DEFAULT '{}'  -- JSON
);

-- messages 表：消息明细
CREATE TABLE messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key  TEXT NOT NULL REFERENCES sessions(session_key),
    role         TEXT NOT NULL,      -- system / user / assistant / tool
    content      TEXT DEFAULT '',
    tool_calls   TEXT DEFAULT NULL,  -- JSON array, only for assistant
    tool_call_id TEXT DEFAULT NULL,  -- only for tool
    reasoning    TEXT DEFAULT NULL,  -- reasoning_content, for supported models
    created_at   REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- 索引：按 session + 时间查消息（最高频查询）
CREATE INDEX idx_messages_session_time
    ON messages(session_key, created_at);
```

### SessionStore 重构

当前 `SessionStore` 持有 `Dict[str, Session]`，每个 Session 持有完整 `messages` 列表。
重构后：

- `SessionStore` 持有一个 SQLite 连接（`~/.tyagent/sessions.db`）
- `Session` 变成轻量对象，只有 `session_key`、`created_at`、`updated_at`，**不再持有 messages**
- 消息操作全部走 SQL：
  - `add_message(session_key, role, content, **extras)` → INSERT
  - `get_messages(session_key)` → SELECT 全部（用于构建压缩视图）
  - `get_message_count(session_key)` → SELECT COUNT

### 压缩策略（新 context.py）

替换现有的头-中-尾摘要方案，改为基于 tool 消息丢弃的确定性策略：

```
build_api_messages(session_key):
    1. 从 DB 查全部消息，按 created_at 排序
    2. 找到最后一个 role='user' 的位置（last_user_idx）
    3. 遍历消息：
       - 消息序号 > last_user_idx → 原样保留（当前交互的 tool 链）
       - 消息序号 ≤ last_user_idx：
         - role='tool' → 跳过
         - role='assistant' → 保留 content 和 reasoning，去掉 tool_calls
         - role='user' / 'system' → 原样保留
    4. 检查压缩后的 token 估算，如果仍超预算，日志警告（暂不做二级压缩）
```

这个策略不需要调 LLM、不需要辅助模型、没有摘要失败的风险。

### archive 行为

`/new` 命令的 archive 语义不变：将旧 session 标记为归档（metadata 里加 `archived_at`），
下次 `get()` 同一个 key 会创建新的 session。归档的 session 消息仍在 DB 里，可查询。

---

## 实施步骤

### 第一步：新建 `tyagent/db.py`

SQLite 存储层，职责：
- 数据库初始化（建表、索引）
- Session CRUD（创建、查询、归档、列表）
- Message CRUD（追加、按 session 查询、计数）
- 连接管理（WAL 模式、线程安全）

不依赖现有 `session.py` 和 `context.py`，全新文件。

### 第二步：重写 `tyagent/session.py`

`Session` 数据类简化为不含 messages 的轻量对象。
`SessionStore` 改为基于 `db.py` 的薄封装，对外接口尽量兼容：

| 旧接口 | 新实现 |
|--------|--------|
| `get(key) → Session` | 查 DB，没有则 INSERT 一行 |
| `add_message(key, role, ...)` | 直接 INSERT INTO messages |
| `save(key)` | 不再需要（每条消息实时写入） |
| `archive(key)` | UPDATE metadata 加 archived_at |
| `reset(key)` | 同 archive（保留旧方案兼容） |
| `all_session_keys()` | SELECT session_key FROM sessions |
| `prune_old_sessions()` | 删除（不再自动清理） |

`Session.messages` 属性改为方法或 property，从 DB 按需查询。

### 第三步：重写 `tyagent/context.py`

删除现有的 `compress_messages`（头-中-尾摘要）和 `_summarize_middle`，
替换为新的 `build_api_messages(store, session_key)` 函数：

1. 调用 `store.get_messages(session_key)` 获取全部消息
2. 找到最后一个 user 消息的位置
3. 按规则过滤 tool 消息和 assistant 的 tool_calls
4. 返回过滤后的消息列表

保留 `estimate_tokens` 和 `should_compress`（改用新的过滤后列表检查）。

### 第四步：适配 `tyagent/agent.py`

`agent.chat()` 的 `messages` 参数语义变化：
- **旧**：传入 session.messages 的引用，tool loop 直接 append 到这个列表
- **新**：传入 `session_key`，agent 从 SessionStore 查消息、追加消息

或者更简单的方案：gateway 层在调 `agent.chat()` 前构建好 `api_messages`，
tool loop 中产生的 assistant/tool 消息同时写入 DB 和临时 `api_messages`。

选择后者，改动最小：agent.py 的 `chat()` 签名不变，仍然接收 `messages` 列表。
但 gateway 的 `_on_message` 改为：用 `build_api_messages()` 构建发送列表，
tool loop 结束后 assistant 和 tool 消息已经通过 `session.add_message()` 写入 DB。

这需要重构 `agent.chat()` 中 tool loop 的消息追加逻辑：
当前是直接 `messages.append(...)`，需要改为同时写 DB。

**具体方案**：agent.chat() 增加一个可选的 `on_message` 回调参数。
每产生一条 assistant 或 tool 消息时，除了 append 到临时 messages 列表外，
还调用 `on_message(role, content, **extras)` 让 gateway 负责持久化。

### 第五步：适配 `tyagent/gateway.py`

`_on_message` 中：
1. `session.add_message("user", ...)` — 写入 DB
2. `api_messages = build_api_messages(session_key)` — 构建压缩视图
3. `response = await self.agent.chat(api_messages, tools=..., on_message=session.add_message)` — 发送 + 回调持久化
4. 发送回复给用户

`save(session_key)` 调用可以删除（每条消息实时写入了）。

### 第六步：迁移与兼容

- 提供迁移脚本：读取旧 JSON session 文件，批量 INSERT 到 SQLite
- 旧的 JSON 文件迁移后保留（不删除），作为备份
- `/status` 命令中的 `len(session.messages)` 改为 `store.get_message_count(key)`

### 第七步：测试更新

| 测试文件 | 改动 |
|----------|------|
| test_session.py | 全部重写，测 SQLite 版的 SessionStore |
| test_context.py | 全部重写，测新的 tool 消息丢弃策略 |
| test_agent.py | 更新 on_message 回调相关的测试 |
| test_gateway.py | 更新消息路由中的持久化验证 |
| test_db.py（新） | DB 层的 CRUD、并发、WAL、迁移 |

### 第八步：清理

- 删除 `SessionStore` 中不再使用的方法（`prune_old_sessions`、`_load_all`）
- 删除旧 JSON 文件相关逻辑（`_sanitize_session_key`、`_save`、`_load_all`）
- 更新 PROGRESS.md

---

## 不做的事

- **二级压缩**：如果丢弃 tool 消息后仍超 context window，暂不处理（日志警告）
- **长期记忆提取**：这是更高层的功能，不在本次范围内
- **压缩链 / parent session**：SQLite 天然保留完整历史，不需要像 Hermes 那样做 session 链
- **消息搜索 API**：DB 架构支持但本次不暴露给上层

---

## 与 Hermes 的对比

| 维度 | Hermes | tyagent（重构后） |
|------|--------|---------------------|
| 存储引擎 | SQLite (state.db) | SQLite (sessions.db) |
| 压缩方式 | 辅助 LLM 做中间摘要 | 丢弃旧 tool 消息，保留 assistant 文本 |
| 消息完整性 | 压缩链追溯 | 单表完整保留，无压缩 |
| 长期记忆 | MEMORY.md + 每日日志 | 暂不实现 |
| 辅助模型 | 需要（压缩/记忆提取） | 不需要 |

---

## 风险

1. **并发写入**：多个消息同时到达同一 session。SQLite WAL 模式 + 写串行化可解决。
2. **DB 文件损坏**：极端情况下（进程被 kill）WAL 可能残留。SQLite 本身有恢复机制，
   但应在启动时加一个 integrity check。
3. **迁移数据丢失**：迁移脚本必须验证导入前后的消息数量一致。
