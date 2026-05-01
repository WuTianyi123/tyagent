# tyagent 开发进展追踪

> 记录：2026-04-22 → 2026-04-28
> 下次会话开始时先读此文件了解当前状态

---

## 已完成功能

### 子代理系统（delegate_task）
- **`tyagent/tools/delegate_tool.py`**（新建，230 行）：spawn 子代理执行隔离任务
  - 子代理获得独立对话上下文、受限工具集（禁止 delegate_task/memory）
  - 摘要式返回（中间工具调用不进父上下文）
  - `_run_child_sync`：新建独立 event loop 运行子代理，600s 超时
  - 父代理通过 `parent_agent` kwarg 透传 model/api_key/system_prompt/compression
  - registry 新增 `wants_parent` 标志支持 handler 签名差异
  - 21 个单元测试，2 轮盲审通过

### 工具进度展示
- **`tyagent/gateway/progress.py`**（新建，299 行）：飞书实时进度条
  - 工具 emoji 注册表 + 预览生成（`📖 read_file: "config.py"`）
  - `ProgressSender`：队列式异步消息编辑，创建/编辑飞书消息显示进度
  - agent.chat() 新增 `tool_progress_callback` 参数

### 命令注册表 + 优雅重启
- **`tyagent/gateway/commands.py`**（新建，197 行）：`CommandRegistry` 类
  - 可插拔命令注册，`/help` 自动从注册表生成
  - `/status` `/restart` `/new` 统一注册和分发
- **`tyagent/gateway/lifecycle.py`**（新建，322 行）：`GatewaySupervisor` 类
  - 信号处理（SIGINT/SIGTERM/SIGUSR1）
  - 优雅重启：通知活跃会话 → 排水 → 标记 → systemctl 重启
  - `.clean_shutdown` JSON marker（含 requestor 信息 + 时间戳）
  - 崩溃恢复：检查 marker → 挂起重启通知 → adapter 连接后自动发送
  - `schedule_restart_notification`：轮询 adapter 状态，连接后发送"已重启（耗时 Xs）"

### 飞书处理状态反应
- **`tyagent/platforms/feishu.py`**：Typing badge ⌨️
  - `_add_reaction` / `_remove_reaction`：添加/移除消息反应
  - LRU 缓存（100 条），`FEISHU_REACTIONS` 环境变量开关

### Gateway 子包架构
- 从单体 `gateway.py`（819 行）拆分为 6 个模块：
  - `gateway.py`（601 行）— 纯编排
  - `commands.py`（197 行）— 命令注册分发
  - `lifecycle.py`（322 行）— 信号/重启/排水/恢复
  - `consumer.py`（309 行）— 流式消息
  - `progress.py`（299 行）— 工具进度
  - `__init__.py`（5 行）— 导出
- 2 轮盲审通过，406 测试全绿

### 工具系统（Tools）
- 6 个内置文件/代码工具：read_file, write_file, patch, search_files, terminal, execute_code
- 10 个浏览器自动化工具：browser_navigate, browser_snapshot, browser_click, browser_type, browser_scroll, browser_back, browser_press, browser_get_images, browser_vision, browser_console
  - 复用 Hermes 的 agent-browser CLI（Playwright），零 API key
  - 支持 per-task session 隔离
  - 24 个测试全部通过

### CLI
- configure — 交互式配置 LLM
- setup-feishu — 扫码绑定飞书机器人
- gateway run/install/start/stop/restart/status — 网关管理
- config / set-model / test-llm

### 平台适配
- Feishu WebSocket 消息收发
- 扫码注册流程
- Markdown → Feishu post 自动转换（代码块、粗体、斜体、链接等）
- 群聊 @mention 门控
- 自身消息过滤
- 消息去重持久化（24h TTL，保存到 ~/.tyagent/cache/feishu/seen_message_ids.json）
- 入站媒体下载：支持 image/file/audio/media，自动推断扩展名
- 出站媒体发送：send_photo / send_document 通过 Feishu 上传 API 实现

---

## 架构重构（2026-04-24）

### SQLite 会话存储

将 session 存储从 JSON 文件迁移到 SQLite：

- **`tyagent/db.py`**（新建）：SQLite 存储层，WAL 模式 + 线程锁，消息永不删除
  - `sessions` 表和 `messages` 表，外键约束保障引用完整性
  - `INSERT OR IGNORE` 原子 create-if-not-exists，消除 TOCTOU 竞态
  - `add_message()` 自动创建 session（无需显式 create）
  - `delete_sessions_older_than()` 单条 SQL 批量清理
  - `import_messages()` 支持 JSON → SQLite 迁移
  - 29 个测试全部通过，经过 7 轮盲审-修复循环后连续 2 轮无实质性问题

- **`tyagent/session.py`**（重写）：精简 Session + SessionStore 封装
  - `Session.messages` 属性按需从 DB 惰性加载
  - `Session.add_message()` 委托给 store 实时持久化
  - `SessionStore` 带 `close()` 和 context manager，自动清理临时目录
  - `save()` 为 no-op（消息实时写入，无需显式保存）
  - 26 个测试全部通过，经过 7 轮盲审-修复循环后连续 2 轮无实质性问题

### 上下文压缩

- **`tyagent/context.py`**：LLM 单次摘要方案
  - 当 token 估算超过上下文窗口 50% 时触发
  - 将最近一条 user 消息之前的全部对话发给 LLM 做一次摘要
  - 保留当前轮次的完整 tool 调用链
  - 利用 KV-cache 优化（对话内容在前、指令在后）

### Agent 层适配

- **`tyagent/agent.py`**：新增 `on_message`、`tool_progress_callback` 回调参数
  - tool loop 中每条 assistant/tool 消息持久化回调
  - `_compression_config` 存储（供 delegate_task 子代理克隆）
  - `parent_agent=self` 透传给 registry.dispatch（供 delegate_task 使用）
  - 14 个测试全部通过

---

## 测试状态

**406 tests passed**（核心模块 + 工具 + 平台）

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|----------|
| test_db.py | 29 | SQLite CRUD、并发安全、迁移、边界情况 |
| test_session.py | 26 | Session/SessionStore、惰性加载、归档、context manager |
| test_context.py | 26 | 上下文压缩、字符预算、边界情况、常量导出 |
| test_agent.py | 14 | 初始化、chat 流程、tool loop、错误处理、回调 |
| test_gateway.py | 14 | 消息路由、命令、drain、重启、恢复、状态 |
| test_delegate_tool.py | 21 | 注册、验证、工具过滤、子代理执行、异常路径 |
| test_migrate.py | 12 | 迁移脚本、归档文件、损坏文件、验证 |
| test_browser_tools.py | 24 | 浏览器工具注册、CLI 发现、snapshot 解析、handler |
| test_config.py | 54 | PlatformConfig、AgentConfig、TyAgentConfig、load/save |
| test_feishu_media.py | 8 | 媒体扩展名推断 |
| test_feishu_message_building.py | 13 | Markdown→飞书 post 构建 |
| test_memory_tool.py | 20+ | 记忆 CRUD、扩展、注入扫描 |
| test_search_tool.py | 5+ | 会话搜索 |
| 其他 | 140 | stream、adapter、context |

---

## 已知问题

- 3 个浏览器集成测试需要真实 `agent-browser` CLI 环境（不在 CI 中运行）
- delegate_task 缺少并发和超时场景的集成测试（21 个单元测试覆盖关键路径，盲审认定为优化性而非阻塞性）

---

## 技术栈

- Python 3.11 + uv
- SQLite（WAL 模式，线程安全）
- lark-oapi（飞书 SDK）
- agent-browser CLI（Playwright，v1217 chromium）
- systemd user service
