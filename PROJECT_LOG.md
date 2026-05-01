# TyAgent 项目日志

> 最后更新: 2026-04-28

## 项目概览

- **位置**: `~/project/tyagent/`
- **描述**: 全功能、可扩展的 AI Agent 框架，优先支持飞书/Lark 消息网关
- **版本**: 0.1.0
- **语言**: Python 3.11+
- **测试**: 371 个 pytest 测试全部通过

## 当前代码架构

### 核心系统
```
tyagent/
├── __init__.py (3 行)      # 包标识 + 版本号
├── gateway/                  # 消息路由 + 流式消费者（由 gateway.py 拆分而来）
│   ├── __init__.py (5 行)    # 导出 Gateway
│   ├── gateway.py (495 行)   # Gateway 编排 + _sanitize_message_chain
│   └── consumer.py (223 行)  # StreamConsumer 流式编辑引擎
├── agent.py (388 行)         # TyAgent — LLM 交互核心（含流式/非流式 chat）
├── config.py (196 行)        # 配置管理
├── context.py (221 行)       # 上下文构建（build_api_messages + 压缩）
├── db.py (648 行)            # SQLite 数据库操作层（WAL 模式）
├── migrate.py (172 行)       # 数据库迁移
├── session.py (255 行)       # 会话管理（Session + SessionStore）
├── service_manager.py (243 行) # systemd 服务管理
├── platforms/
│   ├── __init__.py (5 行)
│   ├── base.py (195 行)      # 平台适配器基类（SendResult, edit_message）
│   └── feishu.py (1418 行)   # 飞书适配器（最大文件）
└── tools/
    ├── __init__.py (18 行)
    ├── core.py (887 行)      # 核心工具集：文件/终端/浏览器/图片/语音/委托
    ├── browser_tools.py (741 行) # 浏览器自动化
    ├── memory_tool.py (699 行)   # 持久记忆管理
    ├── search_tool.py (112 行)   # 搜索工具
    └── registry.py (153 行)      # 工具注册中心
```

### CLI 入口
`tyagent_cli.py` (501 行) — 命令行工具，支持：
- `gateway run/install/start/stop/restart/status`
- `configure`、`set-model`、`config`、`test-llm`
- `setup-feishu`

### 入口点
`main.py` (10 行) — 简单启动脚本

## 关键设计决策

### 会话管理
- SQLite 存储（WAL 模式），`Session` 对象不再持有 messages 列表
- `SessionStore` 底层通过 `db.py` 的 `Database` 类操作
- 消息按 session_key 归档存储

### 上下文压缩
- `context.py` → `build_api_messages()` 构建给 LLM 的消息列表
- 压缩策略：找到最后一条 user 消息，丢弃其之前的 tool 消息，保留 assistant 文本（去掉 tool_calls）
- 预设 `~100K` 字符压缩阈值，无需额外 LLM 调用

### 流式回复（Streaming）
2026-04-27 新实现：
- `agent.chat(stream=True)` 通过生成器 yield delta，`Gateway` 通过 `stream_delta_callback` 连接 `StreamConsumer` 做平台编辑
- `StreamConsumer` 通过 `queue.Queue` 做 sync→async 桥接
- 首次用 `send_message` 发送，后续用 `edit_message` 原地编辑
- 指数退避 + 洪水控制（最多 6 次高峰自动降级到发新消息）
- 截断保护：单条消息不超 `MAX_MESSAGE_LENGTH - 100`
- 已传递 `reply_to_message_id`（修复后）
- 非流式路径完全不变

### Gateway 架构
- 适配器模式：每个平台一个 `BasePlatformAdapter` 子类
- 当前只有 `FeishuAdapter`（1418 行）
- 消息路由：非命令消息走流式，命令消息（new/reset 等）走非流式
- 工具注册：`tools/registry.py` 统一管理

### 工具系统
- `core.py` (887 行) — 最大的单个工具文件，包含：
  - 文件读写、搜索、终端执行
  - 浏览器导航/交互
  - 图片生成、语音合成
  - 任务委托、会话搜索
- `memory_tool.py` (699 行) — 持久记忆管理
- `browser_tools.py` (741 行) — 浏览器自动化

## 近期重要改动（2026-04-27/28）

<!-- 新增条目请追加到此表末尾 -->

| 日期 | 提交 | 改动 |
|------|------|------|
| 04-28 | b573424 | fix: streaming 回复的两个中严重度问题（reply_to_message_id、异常传播） |
| 04-28 | 4e051e8 | cleanup: 移除未使用的 import |
| 04-28 | 578b12a | refactor: gateway.py 拆分为 gateway/ 子包（零逻辑） |
| 04-27 | 1528a20~9ba47f3 | streaming 回复 3 轮盲审修复（截断策略、BUG 修复等） |
| 04-27 | 3fd7551 | [verified] 完整实现 streaming 回复 |
| 04-26~27 | b29c607~f61dd78 | 设计方案 3 轮迭代 + 方案初稿 |
| 更早 | 199ea1a | rename: ty-agent → tyagent（包名统一） |
| 更早 | 5907907 | feat: context compression + archive-based /new |

## 已知状态

- 全部 371 测试通过
- 流式回复功能完整（含引用回复、异常传播）
- 平台支持：仅飞书（一个适配器其他平台可加）
- gateway.py 已拆为子包
- session 重构方案见 `docs/plan-sqlite-session.md`（SQLite 迁移已完成，session 抽象层待优化）

## 待办/未解决的问题

1. **`reply_to_message_id`** 已修复 ✅（04-28）
2. **`consumer_task` 异常传播** 已修复 ✅（04-28）
3. **代码审查偏好**：严格执行 SDD 子代理驱动开发 + 连续两轮盲审通过
