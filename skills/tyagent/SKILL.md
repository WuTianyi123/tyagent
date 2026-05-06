---
name: tyagent
description: "Use, configure, or extend tyagent — the AI agent framework with native Feishu/Lark gateway."
---

# tyagent

tyagent is an AI agent framework with a native Feishu/Lark messaging gateway. It runs on Linux/macOS and supports any OpenAI-compatible LLM provider (DeepSeek, Anthropic, OpenAI, OpenRouter, etc.).

## Architecture

```
Feishu/Lark ←→ Gateway ←→ TyAgent (actor model loop) ←→ LLM API
                         ↕
                    Tools / Skills / Sub-agents
```

- **Gateway** (`gateway/gateway.py`) — message routing, session management, platform adapters, graceful restart
- **TyAgent** (`agent.py`) — permanent event loop, LLM+tool-call loop, Codex-style context compaction
- **Tools** (`tools/`) — file ops, terminal, browser (Playwright), sub-agent delegation, memory, session search
- **Skills** (`skills/`) — pluggable procedures activated by user intent (e.g. `neat-freak` for knowledge sync)
- **Sub-agents** (`subagent/`) — hierarchical task tree, mailbox-based inter-agent communication

## CLI Commands

```
uv run python tyagent_cli.py configure     # Interactive LLM config (model, key, base URL)
uv run python tyagent_cli.py setup-feishu  # Scan QR code to bind Feishu bot
uv run python tyagent_cli.py gateway run   # Run gateway in foreground
uv run python tyagent_cli.py gateway install  # Install as systemd user service
uv run python tyagent_cli.py config        # View current config (redacted)
uv run python tyagent_cli.py test-llm      # Test LLM API connection
uv run python tyagent_cli.py set-model     # Set model parameters directly
```

Global flags: `-c, --config <path>` for custom config, `--profile <name>` for named profiles.

## Feishu Commands

| Command | Function |
|---------|----------|
| `/help` | Show all commands |
| `/status` | Session status (model, message count) |
| `/new` | Archive current session, start fresh |
| `/restart` | Graceful restart gateway |

## Service Management

```bash
# Start / Stop / Restart (use `tyagent gateway restart` for graceful restart)
systemctl --user start tyagent-gateway
systemctl --user stop tyagent-gateway
tyagent gateway restart
systemctl --user status tyagent-gateway

# View logs
journalctl --user -u tyagent-gateway -f
```

The restart is graceful: active requests get up to 5s to finish before the process exits. systemd automatically starts the new process.

## Config

Profiles live at `~/.tyagent/<profile>/config.yaml`. Default profile: `tyagent`.

Environment variable overrides:

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENAI_API_KEY` | OpenAI / compatible key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `FEISHU_APP_ID` | Feishu app ID |
| `FEISHU_APP_SECRET` | Feishu app secret |

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run python -m pytest tests/ -q

# Run specific tests
uv run python -m pytest tests/test_gateway.py tests/test_feishu_adapter.py -q
```

Key conventions (see `AGENTS.md`):
- Test first: run `tests/ -q` before making changes
- Git workflow: separate feat/fix commits, push to `github-hermes:WuTianyi123/tyagent.git`
- Patch: use exact match only (no `replace_all`)
- Blind review loop: major changes go through iter-review-optimize (2 rounds clean)

## Project Structure

```
tyagent/
├── agent.py              # TyAgent: permanent agent loop, tool loop, compaction
├── config.py             # TyAgentConfig, PlatformConfig, CompressionConfig
├── compaction.py         # Codex-style context compression
├── prompt_builder.py     # System prompt assembly (4 layers)
├── types.py              # Message types (InboxMessage, AgentOutput, ReplyTarget)
├── gateway/              # Gateway, commands, lifecycle, progress
├── tools/                # Core tools, browser, delegate, memory, search
├── platforms/            # Base + Feishu adapter
├── skills/               # Pluggable skills (neat-freak, etc.)
├── subagent/             # Mailbox, TaskTree for sub-agent management
├── prompts/              # File-based prompt templates
└── tests/                # pytest test suite
```
