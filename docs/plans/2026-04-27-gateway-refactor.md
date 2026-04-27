# Gateway 重构计划：拆分 `gateway.py` 为子包

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 将 691 行的单体 `gateway.py` 拆分为 `gateway/` 子包，分离 Gateway 编排逻辑和 StreamConsumer 流式编辑引擎。

**架构:** `gateway/` 子包包含 `__init__.py`（导出 Gateway 保证 import 兼容）+ `gateway.py`（Gateway 编排类）+ `consumer.py`（StreamConsumer 流式编辑引擎）。

**原则:** 零逻辑改动，只搬代码不改一行行为。搬完后 import 链不变（`from tyagent.gateway import Gateway` 照常工作）。

---

## 依赖分析

| 组件 | 所在文件 | 依赖 | 被依赖 |
|------|----------|------|--------|
| `_DONE` / `_NEW_SEGMENT` | `gateway.py:30-31` | 无 | `StreamConsumer`, `Gateway._on_message` |
| `_sanitize_message_chain()` | `gateway.py:34-70` | 无 | `Gateway._on_message` |
| `StreamConsumer` | `gateway.py:95-286` | `_DONE`, `_NEW_SEGMENT`, `queue.Queue`, `BasePlatformAdapter`, `SendResult` | `Gateway._on_message`, test |
| `Gateway` | `gateway.py:289-629` | `StreamConsumer`, `_sanitize_message_chain`, `_DONE`, `_NEW_SEGMENT`, 及所有 tyagent 模块 | 所有外部引用 |

**外部引用：**
- `tests/test_gateway.py:14` — `from tyagent.gateway import Gateway, StreamConsumer, _sanitize_message_chain`
- `tyagent/tools/__init__.py` — 不直接引用 gateway
- `tyagent/service_manager.py` — 不直接引用 gateway

---

## 最终文件结构

```
tyagent/
├── gateway/
│   ├── __init__.py          # from .gateway import Gateway (保持 import 兼容)
│   └── consumer.py          # StreamConsumer + sentinel 常量
└── gateway.py               # 删除（内容已迁移）
```

> 注意：不把 Gateway 类也拆出 `__init__.py`，而是新建一个 `gateway/gateway.py` 文件通过 `__init__.py` 再导出。这是因为：
> 1. `Gateway` 类本身也超过 300 行，以后可能继续拆；如果放 `__init__.py` 会变成新的单体文件
> 2. 这样 `gateway/consumer.py` 的 import 自洽：`from .gateway import ...` vs `from .consumer import StreamConsumer`

### 拆后文件对照

| 文件 | 原来行数 | 拆后行数 | 内容 |
|------|----------|----------|------|
| `tyagent/gateway.py` | 691 | — | 删除 |
| `tyagent/gateway/__init__.py` | — | ~3 | `from .gateway import Gateway` |
| `tyagent/gateway/gateway.py` | — | ~440 | `_sanitize_message_chain` + `Gateway` 类 |
| `tyagent/gateway/consumer.py` | — | ~240 | sentinel + `StreamConsumer` 类 |

---

## Task 1: 创建目录和 `consumer.py`

**Objective:** 创建 `tyagent/gateway/` 目录，从 `gateway.py` 提取 `StreamConsumer` 类和相关常量到 `consumer.py`

**Files:**
- Create: `tyagent/gateway/consumer.py`
- Create: `tyagent/gateway/__init__.py`
- Delete: `tyagent/gateway.py`（仅在迁移完成后）

**Step 1: 提取 StreamConsumer 到新文件**

从 `tyagent/gateway.py` 中提取以下内容到 `tyagent/gateway/consumer.py`：

1. 三行 sentinel 常量（L29-31）：
   ```python
   _DONE = object()
   _NEW_SEGMENT = object()
   ```

2. `StreamConsumer` 类完整代码（L95-L286），包括所有方法：
   - `__init__`（L109）
   - `on_delta`（L127）
   - `on_segment_break`（L131）
   - `finish`（L135）
   - `run`（异步方法——这是核心，包含整个消息编辑循环）
   - `_try_edit`（异步方法）
   - `_is_flood_error`（静态方法 L283）

3. `StreamConsumer.__init__` 当前签名：
   ```python
   def __init__(self, adapter: BasePlatformAdapter, chat_id: str):
   ```

**Step 2: consumer.py 创建要点**

```python
"""Streaming message consumer for tyagent gateway.

Manages progressive message delivery by accumulating streamed text
and sending/editing platform messages with flood control.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any, Dict, Optional

from tyagent.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)

# Sentinels for StreamConsumer queue
_DONE = object()
_NEW_SEGMENT = object()

class StreamConsumer:
    # ... 完整类代码
```

注意模块级 docstring 和 import 精简——consumer.py 只依赖 `queue`, `asyncio`, `time`, `logging` + `BasePlatformAdapter`, `SendResult`。

**Step 3: 创建 `tyagent/gateway/__init__.py`**

```python
"""Gateway subsystem — message routing, streaming consumer."""
from tyagent.gateway.gateway import Gateway

__all__ = ["Gateway"]
```

**Step 4: 更新 `tyagent/gateway.py` 为旧版兼容桩**

暂不改——等 `gateway/gateway.py` 建好后再统一操作。

**Verify:**
- `python3 -c "from tyagent.gateway.consumer import StreamConsumer, _DONE, _NEW_SEGMENT"` 不报错

---

## Task 2: 创建 `gateway/gateway.py`（精简版 Gateway）

**Objective:** 从旧 `gateway.py` 中去除 StreamConsumer 相关代码，只保留 `_sanitize_message_chain` + `Gateway` 类

**Files:**
- Create: `tyagent/gateway/gateway.py`
- Keep: `tyagent/gateway/__init__.py`（需更新 import 为相对导入）

**Step 1: 精简 import**

从原 `gateway.py:1-27` 中删掉：
- `import queue`（只在 StreamConsumer 用）
- 删掉 sentinel 常量

新增 import：
- `from tyagent.gateway.consumer import StreamConsumer, _DONE, _NEW_SEGMENT`

**Step 2: 去除 `_DONE`、`_NEW_SEGMENT` 常量**

在 `gateway/gateway.py` 中不再定义，而是 import 自 `consumer`

**Step 3: 精简模块级函数**

只保留：
- `_sanitize_message_chain`（在 `_on_message` 中使用）
- 删掉 `_DONE`/`_NEW_SEGMENT`

**Step 4: 更新 `__init__.py` 为相对导入**

```python
"""Gateway subsystem — message routing, streaming consumer."""
from .gateway import Gateway

__all__ = ["Gateway"]
```

**Verify:**
- `from tyagent.gateway import Gateway` 不报错

---

## Task 3: 更新测试文件 import

**Files:**
- Modify: `tests/test_gateway.py:14`

**Step 1: 改 import 行**

```python
# 原来
from tyagent.gateway import Gateway, StreamConsumer, _sanitize_message_chain

# 改为
from tyagent.gateway import Gateway
from tyagent.gateway.consumer import StreamConsumer, _DONE, _NEW_SEGMENT
from tyagent.gateway.gateway import _sanitize_message_chain
```

注意：`_sanitize_message_chain` 是模块级函数，不在 `__init__.py` 的公共导出中，所以直接 `from .gateway import _sanitize_message_chain`。

**Verify:**
- `pytest tests/test_gateway.py -q` 全部通过
- `pytest tests/ -q` 全部通过

---

## Task 4: 删除旧 `gateway.py` 并最终确认

**Objective:** 删掉旧的单体 `tyagent/gateway.py`，确认所有 import 和测试正常

**Files:**
- Delete: `tyagent/gateway.py`
- Keep: `tyagent/gateway/` 整个子包

**Step 1: 删除旧文件**

```bash
git rm tyagent/gateway.py
```

**Step 2: 验证**

```bash
pytest tests/ -q
# 预期: 全部通过（371 passed）
```

**Step 3: 提交**

```bash
git add -A
git commit -m "refactor: 将 gateway.py 拆分为 gateway/ 子包

- 新建 tyagent/gateway/ 子包（__init__.py + gateway.py + consumer.py）
- StreamConsumer 及相关 sentinel 常量移至 consumer.py
- _sanitize_message_chain 和 Gateway 类保留在 gateway/gateway.py
- 更新测试文件 import 路径
- 零逻辑改动，测试全部通过"
```

---

## 最终验证

```bash
# 运行全部测试
pytest tests/ -q

# 确认 import 链正常
python3 -c "from tyagent.gateway import Gateway; print('OK')"
python3 -c "from tyagent.gateway.consumer import StreamConsumer; print('OK')"
python3 -c "from tyagent.gateway.gateway import _sanitize_message_chain; print('OK')"

# 确认旧文件已删除
ls tyagent/gateway.py 2>&1 || echo "old file gone"
```
