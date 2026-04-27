# Streaming Response Optimization Plan

## Problem

用户发出消息后，tyagent 直到完整回复生成完毕才一次性发送，中间没有任何反馈。对于思考时间长（如 DeepSeek）、工具调用多的情况，用户会感到"没反应"。

**根因：**
1. `agent.chat()` 使用 `stream=False` 调用 API，直到完整响应才返回
2. 最终内容通过单次 `adapter.send_message()` 一次性发送
3. Agent 实例每次 _on_message 重建，无 session 级缓存

## 实现方案：三层渐进

借鉴 Hermes 的设计，用三层架构解决：

```
用户 ←飞书WS→ FeishuAdapter ←asyncio→ Gateway.StreamConsumer ←queue→ TyAgent.chat() (stream=True)
                                      ↑ sync callback          ↑ API stream
                                   gateway.py                agent.py
```

### 第一层：agent.py — Streaming API 支持

**现状：** `self._client.post(url, json=payload_base)` 单次 POST，`resp = resp.json()` 等待完整响应。

**改动：** 加 `stream` 参数 + `stream_delta_callback` 回调。

```python
async def chat(
    self,
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    stream_delta_callback: Optional[Callable[[str], None]] = None,
    on_message: Optional[OnMessageCallback] = None,
) -> str:
```

**流式调用路径：**
1. 设置 `"stream": True`，使用 httpx 的 `event-stream` 或 OpenAI Streaming API
2. 逐 chunk 接收 SSE delta：
   - 文本 delta → 调用 `stream_delta_callback(text)`（同步回调）
   - tool_calls delta → 累加
   - reasoning_content delta → 调用 `stream_delta_callback(reasoning_text)`（通过区分前缀标记）
3. 文本流结束后（`finish_reason="stop"` 且无 tool_calls），用累加的完整内容发 `on_message("assistant", content)`
4. 如果有 tool_calls，先发 `on_message("assistant", content, tool_calls=...)`，然后清空流状态执行工具调用，继续循环

**关于 tool 循环中的流式行为：**
- 第一轮（初始回答+工具调用）：文本 delta 推送给 consumer，tool_calls 累加但**不推送**（工具调用时的 "思考中" 文本意义不大）
- 工具执行结果返回后：新的 assistant 回复文本也推送给 consumer
- 工具边界：通过回调 `stream_delta_callback(None)` 通知 consumer —— 这触发的 `_NEW_SEGMENT` 标记让 consumer 知道"前面的内容发完了，下面会开始新的一段"（对应 Hermes 的 `on_segment_break`）

**设计理由（与 Hermes 对比）：**
- Hermes 的 `_interruptible_streaming_api_call` 同样在 tool_calls 出现时抑制文本 delta 推送给显示层（第 6201-6204 行）
- Hermes 对 reasoning_content 也通过 `_fire_reasoning_delta` 独立通道推送（第 6192-6196 行），但是否推送给平台 adapter 取决于平台是否支持

TODO：tyagent 的飞书暂时不支持 reasoning delta 显示，第一阶段只做文本 streaming。

### 第二层：gateway.py — StreamConsumer

**现状：** `_on_message` 拿到消息 → `agent.chat(api_messages, ...)` → 一次性 `adapter.send_message(response)`。

**改动：** 新增 `StreamConsumer` 类，桥接 sync → async。

```python
class StreamConsumer:
    """Bridges sync stream_delta from agent thread to async platform message edit.
    
    Usage:
        consumer = StreamConsumer(adapter, chat_id)
        consumer_task = asyncio.create_task(consumer.run())
        await agent.chat(messages, stream=True, stream_delta_callback=consumer.on_delta)
        consumer.finish()  # Signal stream complete
        await consumer_task  # Wait for final edit
    """

    def __init__(self, adapter: BasePlatformAdapter, chat_id: str):
        self.adapter = adapter
        self.chat_id = chat_id
        self._queue: queue.Queue = queue.Queue()  # thread-safe sync queue
        self._message_id: Optional[str] = None  # ID of the in-progress platform message
        self._accumulated: str = ""
        self._last_edit_time: float = 0.0
        self._edit_interval: float = 0.8  # seconds between edits
        self._buffer_threshold: int = 30  # chars to send before timer
        self._final_content: str = ""  # full accumulated response
    
    def on_delta(self, text: Optional[str]) -> None:
        """Called from agent's thread (sync). None = tool boundary."""
        self._queue.put(_NEW_SEGMENT if text is None else text)
    
    def finish(self) -> None:
        """Signal stream end from main async context."""
        self._queue.put(_DONE)
    
    async def run(self) -> str:
        """Drain queue, edit platform message, return final content."""
        while True:
            # Drain all available items
            got_done = False
            got_segment_break = False
            while True:
                try:
                    item = self._queue.get_nowait()
                    if item is _DONE:
                        got_done = True
                        break
                    if item is _NEW_SEGMENT:
                        got_segment_break = True
                        break
                    self._accumulated += item
                except queue.Empty:
                    break
            
            # Decide whether to flush
            elapsed = time.monotonic() - self._last_edit_time
            should_edit = (got_done or got_segment_break
                        or (elapsed >= self._edit_interval and self._accumulated)
                        or len(self._accumulated) >= self._buffer_threshold)
            
            # If first flush: send initial message (need message_id for subsequent edits)
            if should_edit and self._accumulated:
                if self._message_id is None:
                    result = await self.adapter.send_message(self.chat_id, self._accumulated + " ▉")
                    if result.success:
                        self._message_id = result.message_id
                    else:
                        # If first send fails, just accumulate and try final send
                        pass
                else:
                    # Edit existing message
                    display = self._accumulated
                    if not got_done:
                        display += " ▉"
                    await self._try_edit(display)
                self._last_edit_time = time.monotonic()
            
            # Tool boundary: finalize current message, prepare for new one
            if got_segment_break:
                self._message_id = None
                self._accumulated = ""  # Content already in the platform message
            
            if got_done:
                # Final send without cursor
                if self._accumulated:
                    if self._message_id:
                        await self._try_edit(self._accumulated)
                    else:
                        result = await self.adapter.send_message(self.chat_id, self._accumulated)
                        if result.success:
                            self._message_id = result.message_id
                return self._accumulated
    
    async def _try_edit(self, text: str) -> None:
        """Try to edit the platform message; fallback to new message on failure."""
        if self._message_id and self._message_id != "__sentinel__":
            result = await self.adapter.edit_message(self.chat_id, self._message_id, text)
            if result.success:
                return
        # Edit failed, send new message
        result = await self.adapter.send_message(self.chat_id, text)
        if result.success:
            self._message_id = result.message_id
```

**设计理由（与 Hermes 对比）：**
- Hermes 的 `GatewayStreamConsumer` 同样使用 `queue.Queue` 桥接 sync→async（第 91 行 `self._queue: queue.Queue`）
- Hermes 的编辑策略相同：定时编辑（第 309-313 行 `_edit_interval` / `_buffer_threshold`）+ 工具边界 `_NEW_SEGMENT`（第 33 行）
- Hermes 有 `_send_fallback_final` 和 `_edit_supported` 降级机制——当编辑连续失败时回退到发新消息
- 简化：Hermes 还有 `_filter_and_accumulate` 过滤 think block 的逻辑，tyagent 第一阶段不实现（DeepSeek 的 think block 自然会由 reasoning_content 字段承载）

### 第三层：agent 缓存

**现状：** 每次 `_on_message` 都从 `TyAgent.from_config(config.agent)` 新创建 agent 实例（gateway.py:102）。

**改动：** Gateway 对 session_key 缓存 TyAgent 实例。

```python
# In Gateway.__init__
self._agent_cache: Dict[str, TyAgent] = {}
self._agent_cache_max_size = 100

# In _on_message:
def _get_or_create_agent(self, session_key: str) -> TyAgent:
    """Get cached agent for session, or create new one."""
    agent = self._agent_cache.get(session_key)
    if agent is None:
        agent = TyAgent.from_config(self.config.agent)
        self._agent_cache[session_key] = agent
        # Evict LRU if over cap
        if len(self._agent_cache) > self._agent_cache_max_size:
            lru_key = next(iter(self._agent_cache))
            del self._agent_cache[lru_key]
    return agent
```

**设计理由（与 Hermes 对比）：**
- Hermes 的 `_agent_cache` 是 `OrderedDict[str, tuple]`（第 699 行），每次 cache hit 执行 `move_to_end()` 实现 LRU
- Hermes 的 eviction 有硬上限 `_AGENT_CACHE_MAX_SIZE=100`（第 698 行附近），由 `_enforce_agent_cache_cap()` 在每次插入后触发
- Hermes 还 cache 了 `config_signature_str` 用于检测配置变更时重建 agent（第 692-693 行注释）
- 简化：tyagent 先做基础 LRU cache，暂不做 config signature 检测

### 第四层：feishu.py — 编辑消息支持

**现状：** 只有 `send_message()`（创建新消息），无编辑现有消息的能力。

**改动：** 新增 `edit_message()` 方法。

飞书 API 支持编辑消息：`PATCH /im/v1/messages/{message_id}`

```python
# In FeishuAdapter:
async def edit_message(
    self,
    chat_id: str,
    message_id: str,
    text: str,
    **kwargs: Any,
) -> SendResult:
    """Edit an existing message (progressive update)."""
    if not self._client:
        return SendResult(success=False, error="Client not initialized")
    
    msg_type, payload = _build_outbound_payload(text)
    
    loop = asyncio.get_running_loop()
    def _do_edit() -> SendResult:
        try:
            from lark_oapi.api.im.v1.model import PatchMessageRequest, PatchMessageRequestBody
            req = (
                PatchMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(payload)
                    .msg_type(msg_type)
                    .build()
                )
                .build()
            )
            resp = self._client.im.v1.message.patch(req)
            if resp.code == 0:
                return SendResult(success=True, message_id=message_id)
            return SendResult(success=False, error=f"{resp.code}: {resp.msg}")
        except Exception as exc:
            logger.exception("Failed to edit Feishu message")
            return SendResult(success=False, error=str(exc), retryable=True)
    return await loop.run_in_executor(None, _do_edit)
```

同时在 `BasePlatformAdapter` 添加 `edit_message` 的默认实现（返回 `SendResult(success=False, error="Not supported")`），这样不支持的平台不会崩。

## 方案总览

### 修改文件清单

| 文件 | 改动量 | 内容 |
|---|---|---|
| `tyagent/platforms/base.py` | ~10 行 | 加 `edit_message()` 默认实现 |
| `tyagent/platforms/feishu.py` | ~50 行 | 加 `edit_message()` 方法 |
| `tyagent/agent.py` | ~80 行 | 加 stream 支持 + `stream_delta_callback` 参数 |
| `tyagent/gateway.py` | ~120 行 | 加 `StreamConsumer` 类 + agent 缓存 |
| **总计** | **~260 行** | |

### 运行时流程

```
用户发消息
  → FeishuAdapter 收到 WS 事件
  → Gateway._on_message(event)
      → SessionStore 写入 user message
      → Gateway._get_or_create_agent(session_key)
      → 创建 StreamConsumer
      → asyncio.create_task(consumer.run())
      → agent.chat(messages, stream=True, stream_delta_callback=consumer.on_delta)
          → POST /chat/completions (stream=True)
          → for each SSE chunk:
              delta.content → consumer.on_delta(text)
              tool_calls delta → 累加，不推送
          → finish_reason="tool_calls"
              → consumer.on_delta(None)  # tool boundary
              → 执行工具
              → for each tool result → 追加到 messages
              → 继续下一轮 streaming（重复）
          → finish_reason="stop" (no tool_calls)
              → consumer.finish()
      → await consumer_task  # 确保最后一次编辑完成
      → 返回 consumer 的最终完整内容
```

## 未纳入此计划（可能的后续）

1. **Reasoning delta（thinking 内容）**：当前 DeepSeek 的 `reasoning_content` 不会推送到平台。后续可以在 `StreamConsumer` 中加 `reasoning_callback`，在飞书上发一条"正在思考..."的临时消息，思考完成后删除/替换。
2. **StreamConsumer 的 flood control 降级机制**：Hermes 在连续编辑失败三次后自动停用 progressive edit，回退到单次发送。目前 tyagent 只在 `_try_edit` 中有简单降级，无自适应 backoff。
3. **Config signature 检查**：如果用户改了 config 中的 model/base_url/api_key，当前缓存的 agent 实例不会自动重建。
