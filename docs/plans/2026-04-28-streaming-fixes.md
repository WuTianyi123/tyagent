# 修复 streaming 回复的两个遗留问题

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 修复 streaming 回复路径的两个中严重度问题，完成后按盲审循环标准审查直到通过。

**原则:** 只改需要改的地方，不改无关代码。

---

## 问题 1（中严重度）：`reply_to_message_id` 未传递

**现象:** 非流式路径传了 `reply_to_message_id=event.message_id`，流式路径没传 → streaming 回复不显示为"回复引用"。

**根因:** `StreamConsumer.__init__` 没接收 `reply_to_message_id`，两次 `send_message` 调用（L120 首次发送、L156 最终发送）都没传。

**修复方案:**
1. `StreamConsumer.__init__` 新增 `reply_to_message_id: Optional[str] = None` 参数，存入 `self._reply_to_message_id`
2. 两个 `self.adapter.send_message` 调用处都传 `reply_to_message_id=self._reply_to_message_id`
3. `Gateway._on_message` 中构造 `StreamConsumer(adapter, event.chat_id, reply_to_message_id=event.message_id)`

### 受影响的文件

| 文件 | 修改 |
|------|------|
| `tyagent/gateway/consumer.py` | `__init__` 加参数 + 2 处 `send_message` 传参 |
| `tyagent/gateway/gateway.py` | `StreamConsumer(adapter, event.chat_id)` → 加 `reply_to_message_id=event.message_id` |

---

## 问题 2（中严重度）：`consumer_task` 异常可能被静默吞噬

**现象:** 如果 `StreamConsumer.run()` 内部出了非预期异常（非 `CancelledError` 也非常规可预期错误），这个异常被 `finally` 块中 `except Exception: logger.exception(...)` 捕获后只打日志不往上抛。用户端看不到任何错误信号。

**根因:** 当前代码（`gateway.py:278-283`）：
```python
finally:
    consumer.finish()
    try:
        await consumer_task
    except Exception:
        logger.exception("StreamConsumer task failed during cleanup")
```

如果 `consumer_task` 的 `run()` 方法抛出了 `consumer.py:173` 之外的异常，这个异常会在这里被 catch 并只记录日志——没重新抛。

**修复方案:**
`finally` 块中判断异常类型：如果是 `CancelledError` → 正常关闭不抛；如果是其他非预期异常 → 重新抛出，让外层 `except AgentError` / `except Exception` 处理并回复错误消息给用户。

```python
consumer.finish()
try:
    await consumer_task
except asyncio.CancelledError:
    pass  # Normal cancellation, consumer already handled it
except Exception:
    logger.exception("StreamConsumer task failed during cleanup")
    raise  # Re-raise so outer handler sends error to user
```

### 受影响的文件

| 文件 | 修改 |
|------|------|
| `tyagent/gateway/gateway.py` | `finally` 块的异常处理区分 CancelledError |

---

## 验证

```bash
cd ~/project/tyagent
python3 -m pytest tests/ -q
# 预期: 全部通过（371 passed）
```
