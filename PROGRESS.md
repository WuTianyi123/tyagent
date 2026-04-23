# ty-agent 开发进展追踪

> 记录：2026-04-22 → 2026-04-23
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
- 消息去重持久化（24h TTL，保存到 ~/.ty_agent/cache/feishu/seen_message_ids.json）
- 入站媒体下载：支持 image/file/audio/media，自动推断扩展名
- 出站媒体发送：send_photo / send_document 通过 Feishu 上传 API 实现

### 稳定性改进（2026-04-23）
- Tool dispatch 异步化：`registry.dispatch()` 通过 `run_in_executor` 在线程池执行，不再阻塞 event loop
- Agent 添加 `import asyncio`（之前缺失，tool calling loop 中需要）
- 修复 agent-browser Chromium 版本不匹配（v1217）

### 上下文管理（2026-04-23 晚）
- **移除了自动裁剪**：不再在启动/定期删除旧 session 文件
- **新增 `ty_agent/context.py`**：上下文压缩模块
  - `estimate_tokens()`：字符数 / 3.5 粗估 token 数
  - `should_compress()`：检查是否超出预算（默认 100k 字符 / 28k token）
  - `compress_messages()`：头保护 + 中间摘要 + 尾保护，返回临时副本不动原始数据
  - `_summarize_middle()`：将中间消息压缩为结构化文本摘要（话题 + 关键事实 + 工具操作）
- **agent.chat() 集成**：发送 API 前自动检查并压缩；tool loop 每轮刷新压缩视图
- **`/new` 命令改为归档**：旧 session 文件重命名为 `<key>__archived_<timestamp>.json` 保留，新建空 session

---

## 已核实问题状态

### 已修复（13个）

1. WebSocket 回调线程安全 — 使用 run_coroutine_threadsafe，不是 create_task
2. 图片下载 API 参数 — message_id 和 file_key 分开传
3. 群聊 @mention 门控 — 已实现，检查 raw_content 中是否含 @bot
4. 自身消息识别 — 检查 sender_type == "bot" 和 sender_id
5. Markdown/富文本发送 — _build_outbound_payload 检测 Markdown 语法自动发 post
6. gateway.py 路由 — 按 event.platform 查找适配器
7. 消息去重持久化 — 保存到磁盘 JSON，启动时加载，线程安全锁保护
8. 媒体下载拓展名 — 根据 Content-Type 和 filename 推断，支持 image/file/audio/media
9. Post 消息媒体标签 — 支持 img/media/file/audio/video 标签解析
10. send_photo — 上传图片到 Feishu 后发送 image 类型消息
11. send_document — 上传文件到 Feishu 后发送 file 类型消息
12. Agent `import asyncio` 缺失 — tool calling loop 中 `asyncio.get_running_loop()` 需要
13. agent-browser Chromium 版本不匹配 — 安装 v1217 匹配 agent-browser 0.13.0

### 仍存在（0个）

所有已知问题已修复。

---

## 测试状态

**169 tests passed**

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|----------|
| test_browser_tools.py | 24 | 注册、CLI 发现、snapshot 解析、命令执行、handler、集成 |
| test_feishu_media.py | 8 | 媒体扩展名推断 |
| test_feishu_message_building.py | 13 | Markdown→飞书 post 构建 |
| test_config.py | 54 | PlatformConfig、AgentConfig、TyAgentConfig、load/save |
| test_session.py | 26 | Session CRUD、持久化、sanitize、archive |
| test_agent.py | 14 | 初始化、chat 基本流程、tool loop、错误处理、max turns |
| test_gateway.py | 12 | 消息路由、reset/status 命令、archive、media 附件、错误降级 |
| test_context.py | 18 | token 估算、压缩触发、压缩结构、摘要生成、原始数据保护 |

---

## 近期计划

1. 消息处理超时保护 — LLM 长时间无响应时给用户反馈
2. WebSocket 断连日志降级 — SDK 重连日志从 ERROR 改为 WARNING
3. 持续测试和稳定性改进
4. 考虑添加更多平台适配器（Telegram、Discord 等）

---

## 技术栈

- Python 3.11 + uv
- lark-oapi（飞书 SDK）
- agent-browser CLI（Playwright，v1217 chromium）
- systemd user service
