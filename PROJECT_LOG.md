# TyAgent 项目日志

> 最后更新: 2026-05-04

## 项目概览

- **位置**: `~/project/tyagent/`
- **描述**: 全功能 AI Agent 框架，native 飞书/Lark 消息网关
- **语言**: Python 3.11+
- **测试**: 508 个 pytest 测试全部通过
- **远程**: `github-hermes:WuTianyi123/tyagent.git`

## 当前架构（2026-05-04）

```
tyagent/
├── agent.py (~740 行)        # TyAgent — 常驻 agent loop + tool loop + 压缩
├── types.py                   # actor 消息类型
├── events.py                  # EventCollector
├── config.py                  # TyAgentConfig, CompressionConfig
├── compaction.py (~525 行)    # Codex CLI 风格压缩
├── prompt_builder.py          # system prompt 构建
├── model_metadata.py          # 模型上下文长度
├── session.py                 # SQLite 会话
├── db.py                      # 数据库层
├── migrate.py                 # 版本迁移
├── gateway/
│   ├── gateway.py             # Gateway 编排 + actor 模型
│   ├── commands.py            # CommandRegistry
│   ├── lifecycle.py           # GatewaySupervisor
│   ├── consumer.py            # StreamConsumer
│   └── progress.py            # ProgressSender
├── tools/
│   ├── registry.py            # ToolRegistry
│   ├── delegate_tool.py       # spawn/wait/close/list
│   ├── core.py                # 文件/代码/终端
│   ├── browser_tools.py       # Playwright
│   ├── memory_tool.py         # MemoryStore + 快照
│   └── search_tool.py         # 会话搜索
├── platforms/
│   ├── base.py                # BasePlatformAdapter
│   └── feishu.py              # FeishuAdapter
└── tests/                     # 508 测试
```

## 近期重要改动

| 日期 | 提交范围 | 改动 |
|------|----------|------|
| 05-04 | 065b9e2 | compression: Codex 风格压缩 + 消息截断 + 前缀匹配 + 空摘要兜底 |
| 05-04 | d46c0d9 | compression: JSON 解析/空摘要永久错误处理 |
| 05-04 | fffcd79 | compression: ContextWindowExceeded 渐进收缩恢复 |
| 05-04 | 1c5b2cf | compaction: _approx_token_count ceil 对齐 Codex |
| 05-03 | ~ | profile 目录重构 + migrate v1→v2 |
| 05-01 | ~ | memory + prompt_builder + system_prompt 重构 |
| 04-28 | ~ | async actor 模型 + 子代理工具 + 442 测试 |
| 04-27 | ~ | delegate_task + ProgressSender + 优雅重启 + streaming 修复 |
| 04-22~27 | ~ | SQLite session + 上下文压缩 + gateway 子包 + 工具系统 |

## 已知状态

- 508 测试全部通过
- Codex 风格压缩完成，4 轮盲审通过
- Actor 模型重构完成
- Profile 目录重构完成
- 平台支持：飞书（FeishuAdapter）
