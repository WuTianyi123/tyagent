"""CLI entry point for tyagent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from tyagent.config import load_config, save_config, TyAgentConfig


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_gateway(args: argparse.Namespace) -> int:
    """Run the gateway."""
    from tyagent.gateway import run_gateway

    setup_logging(args.log_level or "INFO")
    asyncio.run(run_gateway(config_path=args.config))
    return 0


def cmd_gateway_install(args: argparse.Namespace) -> int:
    """Install gateway as a systemd user service."""
    from tyagent.service_manager import install_service
    return install_service(force=args.force)


def cmd_gateway_uninstall(args: argparse.Namespace) -> int:
    """Uninstall the gateway systemd service."""
    from tyagent.service_manager import uninstall_service
    return uninstall_service()


def cmd_gateway_start(args: argparse.Namespace) -> int:
    """Start the gateway systemd service."""
    from tyagent.service_manager import start_service
    return start_service()


def cmd_gateway_stop(args: argparse.Namespace) -> int:
    """Stop the gateway systemd service."""
    from tyagent.service_manager import stop_service
    return stop_service()


def cmd_gateway_restart(args: argparse.Namespace) -> int:
    """Restart the gateway via SIGUSR1 (graceful restart)."""
    from tyagent.service_manager import get_pid

    pid = get_pid()
    if pid is None:
        print("Gateway is not running. Start it with 'tyagent gateway start'.")
        return 1

    print(f"Sending SIGUSR1 to gateway (PID {pid}) for graceful restart...")
    os.kill(pid, signal.SIGUSR1)

    # Wait briefly for the new process to start
    time.sleep(2)

    new_pid = get_pid()
    if new_pid and new_pid != pid:
        print(f"✓ Gateway restarted successfully (new PID {new_pid})")
    else:
        print("Gateway restart signal sent. Check status with 'tyagent gateway status'.")

    return 0


def cmd_gateway_status(args: argparse.Namespace) -> int:
    """Show gateway service status."""
    from tyagent.service_manager import status_service
    return status_service()


def cmd_setup_feishu(args: argparse.Namespace) -> int:
    """Interactive setup for Feishu/Lark."""
    from tyagent.platforms.feishu import qr_register
    from tyagent.config import TyAgentConfig, PlatformConfig

    print("=" * 50)
    print("  tyagent Feishu / Lark Setup")
    print("=" * 50)
    print()

    domain = "feishu"
    if args.lark:
        domain = "lark"
        print("  Using Lark (international) domain.")
    else:
        print("  Using Feishu (China) domain.")

    result = qr_register(initial_domain=domain, timeout_seconds=args.timeout)
    if not result:
        print("\n  Setup failed. Please try again.")
        return 1

    print()
    print("  Configuration:")
    print(f"    app_id:     {result['app_id']}")
    print(f"    app_secret: {'*' * 20}")
    print(f"    domain:     {result['domain']}")
    if result.get("bot_name"):
        print(f"    bot_name:   {result['bot_name']}")

    # Save to config
    config = load_config(Path(args.config) if args.config else None)
    config.platforms["feishu"] = PlatformConfig(
        enabled=True,
        extra={
            "app_id": result["app_id"],
            "app_secret": result["app_secret"],
            "domain": result["domain"],
        },
    )
    save_config(config, Path(args.config) if args.config else None)
    print()
    print(f"  Config saved to {config.home_dir / 'config.yaml'}")
    print("  Run 'python -m tyagent gateway' to start.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show current configuration (sensitive values redacted)."""
    config = load_config(Path(args.config) if args.config else None)
    import copy

    data = copy.deepcopy(config.to_dict())

    # Redact sensitive fields
    def _redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if any(s in k.lower() for s in ("secret", "token", "api_key", "password", "key")):
                    result[k] = "***" if v else v
                else:
                    result[k] = _redact(v)
            return result
        if isinstance(obj, list):
            return [_redact(i) for i in obj]
        return obj

    redacted = _redact(data)
    print(yaml.dump(redacted, default_flow_style=False, allow_unicode=True))
    return 0


def cmd_set_model(args: argparse.Namespace) -> int:
    """Set AI model configuration."""
    config = load_config(Path(args.config) if args.config else None)

    if args.model:
        config.agent.model = args.model
        print(f"  Model set to: {args.model}")
    if args.api_key:
        config.agent.api_key = args.api_key
        print("  API key set.")
    if args.base_url:
        config.agent.base_url = args.base_url
        print(f"  Base URL set to: {args.base_url}")
    if args.system_prompt:
        config.agent.system_prompt = args.system_prompt
        print(f"  System prompt set to: {args.system_prompt}")

    save_config(config, Path(args.config) if args.config else None)
    print(f"  Config saved to {config.home_dir / 'config.yaml'}")
    return 0


def cmd_test_llm(args: argparse.Namespace) -> int:
    """Test LLM API connection directly (debug helper)."""
    import asyncio
    from tyagent.config import load_config
    from tyagent.agent import TyAgent

    config = load_config(Path(args.config) if args.config else None)
    agent = TyAgent.from_config(config.agent)

    messages = [{"role": "user", "content": args.message or "你好，请用一句话介绍自己"}]

    print(f"Model:    {agent.model}")
    print(f"Base URL: {agent.base_url}")
    print(f"API Key:  {'*' * 10 if agent.api_key else '(none)'}")
    print()
    print(f"Sending: {messages[0]['content']!r}")
    print("-" * 40)

    async def _run() -> str:
        try:
            resp = await agent.chat(messages)
            return resp
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        finally:
            await agent.close()

    resp = asyncio.run(_run())
    print(f"Response: {resp}")
    return 0


def cmd_configure(args: argparse.Namespace) -> int:
    """Interactive configuration wizard for tyagent."""
    import os

    config = load_config(Path(args.config) if args.config else None)

    print("=" * 50)
    print("  tyagent Configuration Wizard")
    print("=" * 50)
    print()

    # Show current config
    current_model = config.agent.model or "(not set)"
    current_base_url = config.agent.base_url or "(not set)"
    has_key = "Yes" if config.agent.api_key else "No"
    print(f"  Current model:      {current_model}")
    print(f"  Current base URL:   {current_base_url}")
    print(f"  API key configured: {has_key}")
    print()

    # Provider presets
    _PROVIDERS = [
        {
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "env_var": "OPENAI_API_KEY",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        },
        {
            "name": "Anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "env_var": "ANTHROPIC_API_KEY",
            "models": ["claude-sonnet-4", "claude-opus-4", "claude-haiku-4"],
        },
        {
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "env_var": "OPENROUTER_API_KEY",
            "models": ["anthropic/claude-sonnet-4", "openai/gpt-4o", "deepseek/deepseek-chat"],
        },
        {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "env_var": "DEEPSEEK_API_KEY",
            "models": ["deepseek-chat", "deepseek-reasoner"],
        },
        {
            "name": "Moonshot (Kimi)",
            "base_url": "https://api.moonshot.cn/v1",
            "env_var": "MOONSHOT_API_KEY",
            "models": ["kimi-latest", "kimi-k2", "moonshot-v1-128k"],
        },
        {
            "name": "SiliconFlow",
            "base_url": "https://api.siliconflow.cn/v1",
            "env_var": "SILICONFLOW_API_KEY",
            "models": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-72B-Instruct"],
        },
        {
            "name": "Local / Custom",
            "base_url": "",
            "env_var": "",
            "models": [],
        },
    ]

    print("  Select your LLM provider:")
    for i, p in enumerate(_PROVIDERS, 1):
        marker = "  "
        print(f"    {i}. {marker}{p['name']}")
    print(f"    0.  Leave unchanged")
    print()

    try:
        choice = input("  Enter number: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return 0

    if choice == "0":
        print("  No change.")
        return 0

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(_PROVIDERS):
            print("  Invalid choice.")
            return 1
    except ValueError:
        print("  Invalid input.")
        return 1

    provider = _PROVIDERS[idx]
    print()
    print(f"  → {provider['name']}")

    # Base URL
    if provider["name"] == "Local / Custom":
        try:
            base_url = input("  Base URL (e.g. http://localhost:8000/v1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 0
        if not base_url:
            print("  Base URL is required for custom providers.")
            return 1
        config.agent.base_url = base_url
    else:
        config.agent.base_url = provider["base_url"]
        print(f"  Base URL: {provider['base_url']}")

    # Model selection
    if provider["models"]:
        print()
        print("  Select a model (or type a custom name):")
        for i, m in enumerate(provider["models"], 1):
            print(f"    {i}. {m}")
        print(f"    0.  Other (type manually)")
        print()

        try:
            model_choice = input("  Enter number or model name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 0

        if model_choice == "0":
            try:
                model_name = input("  Model name: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return 0
            if not model_name:
                print("  Model name is required.")
                return 1
            config.agent.model = model_name
        else:
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(provider["models"]):
                    config.agent.model = provider["models"][model_idx]
                else:
                    config.agent.model = model_choice
            except ValueError:
                config.agent.model = model_choice
    else:
        try:
            model_name = input("  Model name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 0
        if not model_name:
            print("  Model name is required.")
            return 1
        config.agent.model = model_name

    # API Key
    print()
    env_key = None
    if provider["env_var"]:
        env_key = os.getenv(provider["env_var"])
    if env_key:
        print(f"  {provider['env_var']} found in environment.")
        try:
            use_env = input(f"  Use it? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 0
        if use_env not in ("n", "no"):
            config.agent.api_key = env_key
        else:
            try:
                api_key = input("  API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return 0
            config.agent.api_key = api_key
    else:
        try:
            api_key = input("  API key (leave blank for none): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 0
        config.agent.api_key = api_key or None

    # System prompt
    print()
    try:
        system_prompt = input("  System prompt (leave blank for default): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return 0
    if system_prompt:
        config.agent.system_prompt = system_prompt

    # Save
    save_config(config, Path(args.config) if args.config else None)
    print()
    print("  Configuration saved!")
    print()
    print(f"    Model:        {config.agent.model}")
    print(f"    Base URL:     {config.agent.base_url}")
    print(f"    API key:      {'*' * 10 if config.agent.api_key else '(none)'}")
    print(f"    System:       {config.agent.system_prompt[:40]}...")
    print()
    print(f"  Config file: {config.home_dir / 'config.yaml'}")
    print()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tyagent",
        description="tyagent: A lightweight, extensible agent framework.",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config file",
    )
    parser.add_argument(
        "-l", "--log-level",
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # gateway
    gateway_parser = subparsers.add_parser("gateway", help="Gateway commands")
    gateway_sub = gateway_parser.add_subparsers(dest="gateway_cmd", help="Gateway subcommands")
    
    # gateway run (default - foreground)
    run_parser = gateway_sub.add_parser("run", help="Run gateway in foreground")
    run_parser.set_defaults(func=cmd_gateway)
    
    # gateway install
    install_parser = gateway_sub.add_parser("install", help="Install gateway as systemd user service")
    install_parser.add_argument("--force", action="store_true", help="Force reinstall")
    install_parser.set_defaults(func=cmd_gateway_install)
    
    # gateway uninstall
    uninstall_parser = gateway_sub.add_parser("uninstall", help="Uninstall gateway systemd service")
    uninstall_parser.set_defaults(func=cmd_gateway_uninstall)
    
    # gateway start
    start_parser = gateway_sub.add_parser("start", help="Start gateway systemd service")
    start_parser.set_defaults(func=cmd_gateway_start)
    
    # gateway stop
    stop_parser = gateway_sub.add_parser("stop", help="Stop gateway systemd service")
    stop_parser.set_defaults(func=cmd_gateway_stop)
    
    # gateway restart
    restart_parser = gateway_sub.add_parser("restart", help="Restart gateway systemd service")
    restart_parser.set_defaults(func=cmd_gateway_restart)
    
    # gateway status
    status_parser = gateway_sub.add_parser("status", help="Show gateway service status")
    status_parser.set_defaults(func=cmd_gateway_status)
    
    # For backward compatibility: "tyagent gateway" without subcommand runs in foreground
    gateway_parser.set_defaults(func=cmd_gateway)

    # setup-feishu
    setup_parser = subparsers.add_parser(
        "setup-feishu", help="Set up Feishu / Lark bot via QR code"
    )
    setup_parser.add_argument(
        "--lark", action="store_true", help="Use Lark instead of Feishu"
    )
    setup_parser.add_argument(
        "--timeout", type=int, default=600, help="QR code timeout in seconds"
    )
    setup_parser.set_defaults(func=cmd_setup_feishu)

    # config
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.set_defaults(func=cmd_config)

    # set-model
    set_model_parser = subparsers.add_parser("set-model", help="Set AI model configuration")
    set_model_parser.add_argument("--model", help="Model name (e.g. gpt-4o, claude-sonnet-4)")
    set_model_parser.add_argument("--api-key", help="API key for the LLM provider")
    set_model_parser.add_argument("--base-url", help="Base URL for the LLM API (e.g. https://api.openai.com/v1)")
    set_model_parser.add_argument("--system-prompt", help="System prompt for the agent")
    set_model_parser.set_defaults(func=cmd_set_model)

    # configure
    configure_parser = subparsers.add_parser("configure", help="Interactive configuration wizard")
    configure_parser.set_defaults(func=cmd_configure)

    # test-llm
    test_parser = subparsers.add_parser("test-llm", help="Test LLM API connection directly")
    test_parser.add_argument("--message", "-m", help="Test message to send")
    test_parser.set_defaults(func=cmd_test_llm)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
