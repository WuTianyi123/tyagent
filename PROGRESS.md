# tyagent 开发进展追踪

> 最后更新: 2026-05-04
> 下次会话开始时先读此文件了解当前状态

---

## 测试状态

**508 tests passed**（全部通过）

## 已完成功能

### Codex 风格上下文压缩（2026-04-28 ~ 05-04）
- **`tyagent/compaction.py`**（新建，525 行）：Codex CLI `compact.rs` 完整对齐
  - `COMPACT_USER_MESSAGE_MAX_TOKENS = 20,000`（固定尾保预算）
  - `SUMMARY_PREFIX` + `COMPACTION_PROMPT` 来自 Codex 模板原文
  - `collect_user_messages()` / `select_tail_messages()` / `build_compacted_history()`
  - 超预算消息截断而非丢弃（匹配 Codex `truncate_middle_with_token_budget`）
  - `ContextWindowExceeded` 渐进收缩恢复（移除最旧消息 + 重置重试预算）
  - 预轮 + 轮中触发，用供应商精确 `prompt_tokens`
  - 摘要不设 `max_tokens`，温度 0.0
  - 38 个单元测试

### 记忆系统（2026-05-04）
- **`tyagent/tools/memory_tool.py`**：MemoryStore + 冻结快照
  - `load_from_disk()` 刷新 MEMORY.md / USER.md
  - `_rebuild_snapshot()` 冻结 system prompt 快照（prefix cache 稳定）
  - `format_for_system_prompt()` 供 prompt_builder 注入
  - 压缩后 `_refresh_memory_and_prompt()` 刷新快照

### System Prompt 重构（2026-05-04）
- **`tyagent/prompt_builder.py`**（新建）：7 层 system prompt
  - identity.md → 用户自定义 system_prompt → 模型信息 → 记忆快照
  - 会话中缓存复用，压缩后重建
  - 记忆从 MemoryStore 快照注入，不再从 gateway 拼接

### Profile 目录重构（2026-05-03）
- 配置从 `~/.tyagent/config.yaml` → `~/.tyagent/<profile>/config.yaml`
- 默认 profile：`tyagent`
- `tyagent/migrate.py`：v1→v2 自动迁移

### Actor 模型 + 异步子代理（2026-04-28）
- Gateway actor 模型：`select!` over inbox/collector/stop
- 子代理工具：`spawn`/`wait`/`close`/`list_tasks`
- `session_key` 隔离（`/new` = archive 旧 + 新 key 不删数据）
- `EventCollector` 子代理完成事件桥接
- 442 个测试（重构前）

### 子代理系统（delegate_task）（2026-04-27）
- **`tyagent/tools/delegate_tool.py`**：spawn 子代理执行隔离任务
  - 子代理获得独立对话上下文、受限工具集
  - `wants_parent` 机制透传 `parent_agent`
  - 21 个单元测试，2 轮盲审通过

### 工具进度展示
- **`tyagent/gateway/progress.py`**：飞书实时进度条
  - 工具 emoji 注册表 + 预览生成
  - `ProgressSender`：队列式异步消息编辑

### 命令注册表 + 优雅重启
- **`tyagent/gateway/commands.py`**：`CommandRegistry` 类
- **`tyagent/gateway/lifecycle.py`**：`GatewaySupervisor` 类
  - 优雅重启：通知→排水→标记→重启
  - 崩溃恢复：检查 marker→挂起重启通知

### Gateway 子包架构
- 从单体 `gateway.py`（819 行）拆分为 6 个模块
- 2 轮盲审通过

### 流式回复
- `StreamConsumer` 队列式 sync→async 桥接
- 指数退避 + 洪水控制

### 工具系统
- 6 个内置文件/代码工具
- 10 个浏览器自动化工具（复用 Hermes agent-browser CLI）

---

## 架构总览

```
tyagent/
├── agent.py           # TyAgent（常驻 loop）
├── types.py           # actor 消息类型
├── events.py          # EventCollector
├── config.py          # 配置管理
├── compaction.py      # Codex 风格压缩
├── prompt_builder.py  # system prompt 构建
├── model_metadata.py  # 模型元数据
├── session.py         # SQLite 会话
├── db.py              # 数据库层
├── migrate.py         # 版本迁移
├── gateway/           # 网关（commands + lifecycle + consumer + progress）
├── tools/             # 工具（core + delegate + browser + memory + search）
├── platforms/         # 平台适配（feishu）
└── tests/             # 508 tests
```

## 技术栈

- Python 3.11 + uv
- SQLite（WAL 模式，线程安全）
- lark-oapi（飞书 SDK）
- agent-browser CLI（Playwright）
- systemd user service
