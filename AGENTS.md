# AGENTS.md — tyagent

AI agent 框架，native Feishu/Lark gateway。Python 3.11 + asyncio + SQLite。

## 项目结构

```
tyagent/
├── agent.py              # TyAgent: LLM 对话，tool loop，压缩
├── config.py             # TyAgentConfig, PlatformConfig
├── context.py            # LLM 单次摘要压缩
├── session.py            # Session + SessionStore（SQLite 后端）
├── db.py                 # SQLite 存储层（WAL，线程安全）
├── gateway/
│   ├── gateway.py        # Gateway：编排（消息路由、agent/adapter 管理）
│   ├── commands.py       # CommandRegistry：/help /status /restart /new
│   ├── lifecycle.py      # GatewaySupervisor：信号、重启、排水、恢复
│   ├── consumer.py       # StreamConsumer：流式消息编辑发送
│   └── progress.py       # ProgressSender：工具进度实时显示
├── tools/
│   ├── registry.py       # ToolRegistry：wants_parent 支持
│   ├── delegate_tool.py  # delegate_task：spawn 子代理
│   ├── core.py           # 文件/代码工具
│   ├── browser_tools.py  # Playwright 浏览器工具
│   ├── memory_tool.py    # 持久记忆
│   └── search_tool.py    # 会话搜索
├── skills/
│   └── neat-freak/       # 会话收尾知识库同步（洁癖）
├── platforms/
│   ├── base.py           # BasePlatformAdapter
│   └── feishu.py         # FeishuAdapter
└── tests/                # 406 tests，pytest
```

## 关键约定

- **测试先行**：改动前跑 `python3 -m pytest tests/ -q`，确保 406 全绿
- **盲审循环**：重要改动走 iter-review-optimize skill（连续 2 轮无高/中问题即通过）
- **Git 工作流**：feat/fix 分开 commit，push 到 `github-hermes:WuTianyi123/tyagent.git`
- **文件命名**：描述性命名，按含义归入子文件夹
- **禁止 `replace_all`**：patch 只用精确匹配

## 核心工具列表

| 工具 | 文件 |
|------|------|
| read_file, write_file, patch, search_files | `tools/core.py` |
| terminal, execute_code | `tools/core.py` |
| browser_navigate, browser_click, ... | `tools/browser_tools.py` |
| delegate_task | `tools/delegate_tool.py` |
| memory | `tools/memory_tool.py` |
| session_search | `tools/search_tool.py` |

## 飞书命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示所有命令 |
| `/status` | 会话状态（模型、消息数） |
| `/new` | 归档当前会话，开始新对话 |
| `/restart` | 优雅重启 gateway |

## 配置

- 默认路径：`~/.tyagent/config.yaml`
- 环境变量覆盖：`OPENAI_API_KEY`、`DEEPSEEK_API_KEY` 等
- systemd 服务：`tyagent-gateway`（user scope）
- 重启标记：`~/.tyagent/.clean_shutdown`（JSON，含 requestor 和时间戳）

## 已知陷阱

- `tools/registry.py` 的 `wants_parent=True` 仅 delegate_task 使用，dispatch 内部判断
- Gateway 通过 `self.commands` 和 `self.supervisor` 委托——不要直接在 Gateway 上找 `_cmd_*` / `_do_graceful_restart` 方法
- `_compression_config` 存在 agent 上供 delegate_task 子代理克隆
- 测试中 `_format_status` 从 `commands.py` 导入，不是 Gateway 方法
- feishu.py 的 `_add_reaction` 受 `FEISHU_REACTIONS` 环境变量控制（默认 true）

## 预装 Skill

### neat-freak（洁癖）
会话结束后对项目文档和记忆进行审查与同步。触发词：`/neat`、`整理一下`、`同步一下`、`收尾`。
参见 `skills/neat-freak/SKILL.md`。
