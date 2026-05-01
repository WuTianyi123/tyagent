# tyagent 开发进展追踪

> 记录：2026-04-22 → 2026-04-24
> 下次会话开始时先读此文件了解当前状态

---

## 已完成功能

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

### 确定性上下文压缩

替换旧的头-中-尾摘要方案：

- **`tyagent/context.py`**（重写）：基于 tool 消息丢弃的确定性压缩
  - 保留最后一个 user 消息及之后的所有 tool 链
  - 丢弃之前的 tool 消息，剥离之前 assistant 的 tool_calls
  - 零额外 API 成本，不依赖辅助 LLM
  - `_content_chars()` 统一字符预算计算（含 tool_calls 元数据）
  - `DEFAULT_MAX_CHARS = 280_000`（~70k tokens，适合 128k+ 模型）
  - 26 个测试全部通过，经过 6 轮盲审-修复循环后连续 2 轮无实质性问题

### Agent 层适配

- **`tyagent/agent.py`**：新增 `on_message` 回调参数
  - tool loop 中每条 assistant/tool 消息持久化回调
  - 使用 `build_api_messages()` + `should_compress(messages, max_chars=...)` 正确触发压缩
  - 14 个测试全部通过

### Gateway 层适配

- **`tyagent/gateway.py`**：使用新的 session store API
  - `build_api_messages(session.messages)` 构建压缩视图
  - `on_message` 回调持久化 tool loop 消息
  - `get_message_count()` 替代 `len(session.messages)`
  - 移除 `session_store.save()` 调用
  - 14 个测试全部通过

### 迁移工具

- **`tyagent/migrate.py`**（新建）：JSON → SQLite 迁移脚本
  - 支持活跃 session 和归档 session 的迁移
  - `verify_migration()` 验证 JSON 文件数与 DB session 数一致
  - 旧 JSON 文件保留作为备份
  - 12 个测试全部通过

---

## 测试状态

**196 tests passed**（核心模块 119 + 其他 77）

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|----------|
| test_db.py | 29 | SQLite CRUD、并发安全、迁移、边界情况 |
| test_session.py | 26 | Session/SessionStore、惰性加载、归档、context manager |
| test_context.py | 26 | 确定性压缩、字符预算、边界情况、常量导出 |
| test_agent.py | 14 | 初始化、chat 流程、tool loop、错误处理、on_message 回调 |
| test_gateway.py | 14 | 消息路由、归档、status、错误降级、on_message |
| test_migrate.py | 12 | 迁移脚本、归档文件、损坏文件、验证 |
| test_browser_tools.py | 24 | 浏览器工具注册、CLI 发现、snapshot 解析、handler |
| test_config.py | 54 | PlatformConfig、AgentConfig、TyAgentConfig、load/save |
| test_feishu_media.py | 8 | 媒体扩展名推断 |
| test_feishu_message_building.py | 13 | Markdown→飞书 post 构建 |

---

## 已知问题

- 3 个浏览器集成测试需要真实 `agent-browser` CLI 环境（不在 CI 中运行）

---

## 技术栈

- Python 3.11 + uv
- SQLite（WAL 模式，线程安全）
- lark-oapi（飞书 SDK）
- agent-browser CLI（Playwright，v1217 chromium）
- systemd user service
