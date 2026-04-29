# tyagent

可扩展的 AI Agent 框架，支持飞书/Lark 消息网关。

## 快速开始

```bash
# 安装依赖
uv sync

# 交互式配置 LLM（模型、API Key、Base URL）
uv run python tyagent_cli.py configure

# 配置飞书机器人（扫码绑定）
uv run python tyagent_cli.py setup-feishu

# 前台运行网关
uv run python tyagent_cli.py gateway run

# 安装为 systemd 用户服务（推荐）
uv run python tyagent_cli.py gateway install

# 之后用 systemctl 管理服务
systemctl --user start tyagent-gateway
systemctl --user restart tyagent-gateway
systemctl --user stop tyagent-gateway
systemctl --user status tyagent-gateway
journalctl --user -u tyagent-gateway -f
```

## CLI 命令总览

| 命令 | 说明 |
|------|------|
| `gateway run` | 前台运行网关（阻塞，Ctrl+C 停止） |
| `gateway install` | 安装为 systemd 用户服务 |
| `gateway uninstall` | 卸载 systemd 服务 |
| `gateway start` | 启动 systemd 服务 |
| `gateway stop` | 停止 systemd 服务 |
| `gateway restart` | 重启 systemd 服务 |
| `gateway status` | 查看服务状态 |
| `setup-feishu` | 扫码配置飞书/Lark 机器人 |
| `configure` | 交互式配置向导（选择模型、填 API Key） |
| `set-model` | 直接设置模型参数 |
| `config` | 查看当前配置（敏感值脱敏） |
| `test-llm` | 直接测试 LLM API 连接 |

## 全局参数

- `-c, --config <path>` — 指定配置文件路径
- `-l, --log-level <level>` — 日志级别（DEBUG/INFO/WARNING/ERROR）

## 配置

配置默认保存在 `~/.tyagent/config.yaml`，包括：

- **LLM 设置**：model、base_url、api_key、system_prompt
- **平台设置**：飞书 app_id、app_secret、domain

可用环境变量覆盖：

- `OPENAI_API_KEY`、`KIMI_API_KEY`、`ANTHROPIC_API_KEY`
- `DEEPSEEK_API_KEY`、`OPENROUTER_API_KEY`

## 查看日志

```bash
# systemd 服务日志
journalctl --user -u tyagent-gateway -f

# 前台运行时直接看终端输出
uv run python tyagent_cli.py gateway run
```
