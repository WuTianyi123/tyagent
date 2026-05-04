# tyagent

全功能 AI Agent 框架，native 飞书/Lark 消息网关。支持子代理委派、流式输出、工具进度显示、优雅重启、Codex 风格上下文压缩。

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

# 管理服务
systemctl --user start tyagent-gateway
systemctl --user restart tyagent-gateway
systemctl --user stop tyagent-gateway
systemctl --user status tyagent-gateway
journalctl --user -u tyagent-gateway -f
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `gateway run` \| `install` \| `start` \| `stop` \| `restart` \| `status` \| `uninstall` | 网关管理 |
| `setup-feishu` | 扫码配置飞书机器人 |
| `configure` | 交互式配置向导 |
| `set-model` | 直接设置模型参数 |
| `config` | 查看当前配置（脱敏） |
| `test-llm` | 测试 LLM API 连接 |

全局参数：`-c, --config <path>` 指定配置文件路径

## 配置

Profile 目录：`~/.tyagent/<profile>/config.yaml`

- 默认 profile：`tyagent`（即 `~/.tyagent/tyagent/config.yaml`）
- 使用 `--profile <name>` 指定其他 profile

环境变量覆盖：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `KIMI_API_KEY` | Kimi API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `FEISHU_APP_ID` | 飞书 App ID |
| `FEISHU_APP_SECRET` | 飞书 App Secret |

## 飞书命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示所有命令 |
| `/status` | 会话状态（模型、消息数） |
| `/new` | 归档当前会话，开始新对话 |
| `/restart` | 优雅重启 gateway |

## 查看日志

```bash
journalctl --user -u tyagent-gateway -f
```
