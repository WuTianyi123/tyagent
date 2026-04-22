# ty-agent 开发进展追踪

> 记录：2026-04-22
> 下次会话开始时先读此文件了解当前状态

---

## 已完成功能

### 工具系统（Tools）
- [x] 6 个内置文件/代码工具：`read_file`, `write_file`, `patch`, `search_files`, `terminal`, `execute_code`
- [x] 10 个浏览器自动化工具：`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`, `browser_get_images`, `browser_vision`, `browser_console`
  - 复用 Hermes 的 agent-browser CLI（Playwright），零 API key
  - 支持 per-task session 隔离
  - 24 个测试全部通过

### CLI
- [x] `configure` — 交互式配置 LLM
- [x] `setup-feishu` — 扫码绑定飞书机器人
- [x] `gateway run/install/start/stop/restart/status` — 网关管理
- [x] `config` / `set-model` / `test-llm`

### 平台适配
- [x] Feishu WebSocket 消息收发（基础版本）
- [x] 扫码注册流程

---

## 已知严重问题（来自 feishu_review.md）

### 必须修复（P0）
1. **WebSocket 回调线程安全** — `asyncio.create_task()` 在后台线程会抛 RuntimeError，应改为 `run_coroutine_threadsafe`
2. **图片下载 API 参数错误** — `GetMessageResourceRequest` 把 `image_key` 同时传给 `message_id` 和 `file_key`，应分开传 `message_id` 和 `file_key`
3. **群聊缺少 @mention 门控** — 不 @bot 的群聊消息也会回复，导致消息洪水
4. **自身消息识别不完整** — 仅检查 `sender_id`，未检查 `sender_type`，有回声循环风险

### 重要修复（P1）
5. 消息去重不可持久化（内存实现，重启失效）
6. 媒体下载严重不足（扩展名硬编码、仅支持图片、同步阻塞、无 MIME 解析）
7. 消息解析不完整（缺少 img/media/file/audio/video/code_block 等标签）
8. 发送消息不支持 Markdown/富文本（永远发纯 text）
9. gateway.py `_find_adapter_for_event` 永远返回第一个适配器
10. 缺少 WebSocket 自动重连
11. `send_photo` / `send_document` 未实现（只发文本路径）

---

## 近期计划

1. 修复 Feishu 适配器的 4 个 P0 严重问题
2. 实现 Markdown 到 Feishu post 的自动转换
3. 添加 WebSocket 自动重连
4. 完善媒体下载（图片/file/audio/video）
5. 消息去重持久化到磁盘

---

## 测试状态

```
37 tests passed (24 browser + 13 feishu)
```

---

## 技术栈

- Python 3.11 + uv
- lark-oapi（飞书 SDK）
- agent-browser CLI（Playwright，已安装在 Hermes env）
- systemd user service
