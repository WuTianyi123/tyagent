"""CLI entry point for ty-agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from ty_agent.config import load_config, save_config, TyAgentConfig


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_gateway(args: argparse.Namespace) -> int:
    """Run the gateway."""
    from ty_agent.gateway import run_gateway

    setup_logging(args.log_level or "INFO")
    asyncio.run(run_gateway(config_path=args.config))
    return 0


def cmd_setup_feishu(args: argparse.Namespace) -> int:
    """Interactive setup for Feishu/Lark."""
    from ty_agent.platforms.feishu import qr_register
    from ty_agent.config import TyAgentConfig, PlatformConfig

    print("=" * 50)
    print("  ty-agent Feishu / Lark Setup")
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
    print("  Run 'python -m ty_agent gateway' to start.")
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


def cmd_configure(args: argparse.Namespace) -> int:
    """Interactive configuration wizard for ty-agent."""
    import os

    config = load_config(Path(args.config) if args.config else None)

    print("=" * 50)
    print("  ty-agent Configuration Wizard")
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
        prog="ty-agent",
        description="ty-agent: A lightweight, extensible agent framework.",
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
    gateway_parser = subparsers.add_parser("gateway", help="Start the messaging gateway")
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

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
