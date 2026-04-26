# tyagent Feishu 适配器专项审查报告

> 本报告基于较早版本代码写成，已在标题中添加【核实】标注。当前实际状态以源码为准。

审查文件：
- `tyagent/platforms/feishu.py`
- `tyagent/gateway.py`
- `tyagent/platforms/base.py`

参考实现：
- `hermes-agent/gateway/platforms/feishu.py`

---

## 一严重问题（已修复 4/4）

### 1. WebSocket 回调线程安全问题（崩溃级）【已修复】

**位置**：`tyagent/platforms/feishu.py:563-565`

实际代码使用了 `asyncio.run_coroutine_threadsafe()`，不是 review 中提到的 `asyncio.create_task()`。且注释明确写着 "We must not call asyncio.create_task() here directly."

```python
future = asyncio.run_coroutine_threadsafe(
    self._handle_message(event), self._loop
)
```

---

### 2. 图片下载 API 参数错误（功能不可用）【已修复】

**位置**：`tyagent/platforms/feishu.py:700-703`

实际代码中 `message_id` 和 `file_key` 是分开传的，不是 review 中提到的都传 `image_key`：

```python
req = GetMessageResourceRequest.builder() \
    .message_id(message_id) \
    .file_key(image_key) \
    .build()
```

---

### 3. 群聊缺少 @mention 门控（洪水级）【已修复】

**位置**：`tyagent/platforms/feishu.py:649-654`

实际代码已实现群聊门控：检查 raw_content 中是否含有 `user_id="{bot_open_id}"`，没有则返回 None 忽略消息。

```python
if chat_type == "group" and self.group_policy != "open":
    raw_content = content_str if isinstance(content_str, str) else json.dumps(content_str)
    if self._bot_open_id and f'user_id="{self._bot_open_id}"' not in raw_content:
        logger.debug("Ignoring group message without @mention: %s", message_id)
        return None
```

---

### 4. 没有识别自身发送的消息（回声/循环风险）【已修复】

**位置**：`tyagent/platforms/feishu.py:625`

实际代码同时检查了 `sender_type` 和 `sender_id`：

```python
if sender_type == "bot" or sender_id == self._bot_open_id:
```

---

## 二、重要缺陷（5 个已修复，6 个部分存在）

### 5. 消息去重不可持久化，且清理效率低 【部分存在】

**位置**：`tyagent/platforms/feishu.py:681-690`

确实是内存字典 `self._dedup`，重启后失效。但已有 24h TTL 和清理逻辑：

```python
self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}
```

---

### 6. 媒体下载实现严重不足 【部分存在】

**位置**：`tyagent/platforms/feishu.py:692-715`

- 扩展名硬编码为 `.png` — 仍然存在
- 仅支持图片 — 仍然存在（只有 _download_image 方法）
- 同步阻塞 — 【已修复】已用 `loop.run_in_executor` 包装
- 缺少 Content-Type 解析 — 仍然存在

---

### 7. 消息解析不完善 【部分存在】

**位置**：`tyagent/platforms/feishu.py:893-936`

实际代码已处理：text, a, at, img, code, code_block/pre, br, hr, divider — 比 review 中说的“只处理 text/a/at”完善很多。但缺少 media/file/audio/video 标签解析。

---

### 8. 发送消息不支持 Markdown/富文本 【已修复】

**位置**：`tyagent/platforms/feishu.py:149-163`

实际代码已实现 `_build_outbound_payload`，检测 Markdown 语法自动发 `post` 类型，并有 fallback 到纯文本的机制。

---

### 9. gateway.py 的 `_find_adapter_for_event` 逻辑错误 【已修复】

**位置**：`tyagent/gateway.py:156-158`

实际代码按 `event.platform` 查找适配器，不是 review 中提到的“永远返回第一个”：

```python
def _find_adapter_for_event(self, event: MessageEvent) -> Optional[BasePlatformAdapter]:
    return self.adapters.get(event.platform)
```

---

### 10. 缺少 WebSocket 自动重连机制 【部分存在】

**位置**：`tyagent/gateway.py:221-246`

实际代码有 `_run_adapter_with_retry`，支持指数退订重试（最多 10 次）。但这是 gateway 层面重启整个适配器，不是 WS client 内部自动重连。

---

### 11. 没有 `send_photo` / `send_document` 的真正实现 【部分存在】

**位置**：`tyagent/platforms/base.py:142-168`

基类中的 `send_photo` 和 `send_document` 默认实现只发送文本路径（如 `[Photo: /path/to/img.png]`）。FeishuAdapter 未覆盖这些方法，未使用 Feishu 图片/文件上传 API。

---

## 三、轻微建议（可选）

### 12. 扫码注册流程可以优化
- `_poll_registration` 使用固定间隔轮询，建议参考实现加入指数退订
- `probe_bot` 使用 `urlopen` 同步请求，可考虑改为 `asyncio.to_thread`
- 注册成功后的 `app_name` 字段没有保存，可以存入配置

### 13. 会话密钥（session key）粒度
`build_session_key` 在群聊中使用了 `sender_id`，这是正确的（与参考实现一致）。但如果未来需要支持“群聊中所有人共享一个会话”，建议增加 `group_sessions_per_user` 配置项。

### 14. 日志和监控
- 缺少对发送失败的分类日志（参考实现区分了 `retryable` 错误和非重试错误）
- 没有 webhook 异常检测计数器
- 没有处理状态反馈（如 Typing 反应）

### 15. 其他遗漏的参考功能（非必需，但值得了解）
- Webhook 模式（HTTP 事件推送）
- 消息已读事件（`im.message.message_read_v1`）
- 消息撤回事件（`im.message.recalled_v1`）
- 表情反应事件路由为合成文本事件
- 卡片按钮点击事件（审批/交互）
- 执行审批卡片（exec approval）
- 文本/媒体消息批处理（batching）
- 发送消息时的分片/长度限制处理
- 速率限制和重试退订

---

## 四、总体评价

**本报告基于较早版本代码，实际代码已修复大部分 P0 问题。当前状态：**

- 4 个 P0 严重问题均已修复（线程安全、API 参数、群聊门控、自身消息过滤）
- 5 个 P1 问题已修复（Markdown 发送、路由逻辑、同步阻塞包装、消息解析完善度大幅提升）
- 仍存在的主要问题：去重不持久化、媒体扩展名硬编码、媒体类型单一、send_photo/send_document 未真正实现
