# TyAgent 项目日志

> 最后更新: 2026-04-28

## 项目概览

- **位置**: `~/project/tyagent/`
- **描述**: 全功能、可扩展的 AI Agent 框架，native 飞书/Lark 消息网关
- **版本**: 0.1.0
- **语言**: Python 3.11+
- **测试**: 406 个 pytest 测试全部通过

## 当前代码架构

```
tyagent/
├── __init__.py (3 行)        # 包标识 + 版本号
├── gateway/
│   ├── __init__.py (5 行)    # 导出 Gateway
│   ├── gateway.py (601 行)   # Gateway 编排（消息路由、agent/adapter）
│   ├── commands.py (197 行)  # CommandRegistry：/help /status /restart /new
│   ├── lifecycle.py (322 行) # GatewaySupervisor：信号、重启、排水、恢复
│   ├── consumer.py (309 行)  # StreamConsumer 流式编辑引擎
│   └── progress.py (299 行)  # ProgressSender 工具进度实时显示
├── agent.py (~420 行)        # TyAgent — LLM 交互核心
├── config.py (196 行)        # 配置管理
├── context.py (221 行)       # LLM 单次摘要压缩
├── db.py (648 行)            # SQLite 数据库（WAL 模式）
├── session.py (255 行)       # 会话管理
├── platforms/
│   ├── base.py (195 行)      # 平台适配器基类
│   └── feishu.py (1585 行)   # 飞书适配器（含 Typing 反应）
└── tools/
    ├── registry.py (153 行)  # 工具注册中心（wants_parent 支持）
    ├── delegate_tool.py (227 行) # delegate_task 子代理
    ├── core.py (887 行)      # 核心工具集
    ├── browser_tools.py (741 行) # 浏览器自动化
    ├── memory_tool.py (699 行)   # 持久记忆管理
    └── search_tool.py (112 行)   # 搜索工具
```

## 关键设计决策

### 子代理系统（delegate_task）
- 独立 event loop 运行子代理，600s 超时
- `wants_parent` 机制：registry.dispatch 透传 `parent_agent` kwarg
- 子代理获得独立上下文 + 受限工具集（禁止 delegate_task/memory）
- 摘要式返回，中间工具调用不进父上下文

### 工具进度展示
- `ProgressSender` 异步队列式消息编辑
- 创建单条飞书消息，每次 tool call 后原地更新
- 1.5s 编辑节流，fallback 到新消息发送

### 优雅重启
- `.clean_shutdown` JSON marker（含 requestor + 时间戳）
- 新进程启动后挂起通知，adapter 连接后自动发送
- `systemd-run` 避免 RestartSec 延迟

### Gateway 子包架构
- `gateway.py` 只保留编排逻辑（601 行，从 819 行减少 27%）
- `CommandRegistry` 可插拔命令模式
- `GatewaySupervisor` 生命周期关注点完全分离

### 流式回复
- `StreamConsumer` 队列式 sync→async 桥接
- 指数退避 + 洪水控制（6 次高峰自动降级）
- 截断保护 + `reply_to_message_id` 透传

## 近期重要改动

| 日期 | 提交 | 改动 |
|------|------|------|
| 04-28 | 5d1eb93 | fix: Gateway.set_restart_requestor() 封装 |
| 04-28 | 240bc72 | refactor: gateway 拆分为 commands + lifecycle 子包 |
| 04-28 | 94e76d5~cd36cca | feat: Feishu 反应 + 命令注册表 + 进度展示 + delegate_task |
| 04-28 | f4de13d~26af994 | feat: delegate_task 子代理（含 21 测试 + 盲审通过） |
| 04-27 | 1528a20~9ba47f3 | streaming 回复 3 轮盲审修复 |

## 已知状态

- 全部 406 测试通过
- 子代理功能可用（2 轮盲审通过）
- Gateway 子包架构完成（2 轮盲审通过）
- 平台支持：飞书（FeishuAdapter，含 Typing 反应）
