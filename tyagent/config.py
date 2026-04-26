"""Configuration management for tyagent."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config directory — use passwd entry so we aren't fooled by $HOME overrides
_usr_home = Path(os.path.expanduser("~"))
try:
    import pwd
    _usr_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
except (ImportError, KeyError):
    pass
default_home = _usr_home / ".tyagent"

# Default workspace — user's real home, not the agent profile home
default_workspace = _usr_home


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""

    enabled: bool = False
    token: Optional[str] = None
    api_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"enabled": self.enabled, "extra": self.extra}
        if self.token:
            result["token"] = self.token
        if self.api_key:
            result["api_key"] = self.api_key
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlatformConfig:
        return cls(
            enabled=data.get("enabled", False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            extra=data.get("extra", {}),
        )


@dataclass
class AgentConfig:
    """Configuration for the AI agent."""

    model: str = "anthropic/claude-sonnet-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_turns: int = 50
    max_tool_turns: int = 30
    system_prompt: str = "You are a helpful assistant."
    context_max_chars: int = 280_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_turns": self.max_turns,
            "max_tool_turns": self.max_tool_turns,
            "system_prompt": self.system_prompt,
            "context_max_chars": self.context_max_chars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        return cls(
            model=data.get("model", "anthropic/claude-sonnet-4"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            max_turns=data.get("max_turns", 50),
            max_tool_turns=data.get("max_tool_turns", 30),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            context_max_chars=data.get("context_max_chars", 280_000),
        )


@dataclass
class TyAgentConfig:
    """Main configuration for tyagent."""

    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)
    agent: AgentConfig = field(default_factory=AgentConfig)
    home_dir: Path = field(default_factory=lambda: default_home)
    workspace_dir: Path = field(default_factory=lambda: default_workspace)
    sessions_dir: Path = field(default_factory=lambda: default_home / "sessions")
    log_level: str = "INFO"
    reset_triggers: List[str] = field(default_factory=lambda: ["new", "reset"])

    def get_platform(self, name: str) -> Optional[PlatformConfig]:
        return self.platforms.get(name)

    def get_connected_platforms(self) -> List[str]:
        """Return list of enabled and configured platform names."""
        connected = []
        for name, cfg in self.platforms.items():
            if not cfg.enabled:
                continue
            # Feishu needs app_id
            if name == "feishu" and cfg.extra.get("app_id"):
                connected.append(name)
            # Generic token-based platforms
            elif cfg.token or cfg.api_key:
                connected.append(name)
        return connected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
            "agent": self.agent.to_dict(),
            "home_dir": str(self.home_dir),
            "workspace_dir": str(self.workspace_dir),
            "sessions_dir": str(self.sessions_dir),
            "log_level": self.log_level,
            "reset_triggers": self.reset_triggers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TyAgentConfig:
        platforms = {}
        for name, pdata in data.get("platforms", {}).items():
            platforms[name] = PlatformConfig.from_dict(pdata)
        home = Path(data.get("home_dir", str(default_home)))
        workspace = Path(data.get("workspace_dir", str(default_workspace)))
        return cls(
            platforms=platforms,
            agent=AgentConfig.from_dict(data.get("agent", {})),
            home_dir=home,
            workspace_dir=workspace,
            sessions_dir=Path(data.get("sessions_dir", str(home / "sessions"))),
            log_level=data.get("log_level", "INFO"),
            reset_triggers=data.get("reset_triggers", ["new", "reset"]),
        )


def load_config(config_path: Optional[Path] = None) -> TyAgentConfig:
    """Load configuration from file or environment.

    Priority:
    1. Explicit config_path
    2. ~/.tyagent/config.yaml
    3. ~/.tyagent/config.json
    4. Built-in defaults
    """
    if config_path:
        return _load_from_path(config_path)

    home = default_home
    yaml_path = home / "config.yaml"
    json_path = home / "config.json"

    if yaml_path.exists():
        return _load_from_path(yaml_path)
    if json_path.exists():
        return _load_from_path(json_path)

    logger.info("No config file found, using defaults.")
    return TyAgentConfig()


def _load_from_path(path: Path) -> TyAgentConfig:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    return TyAgentConfig.from_dict(data)


def save_config(config: TyAgentConfig, path: Optional[Path] = None) -> None:
    """Save configuration to file."""
    if path is None:
        path = config.home_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
    # Restrict permissions so only owner can read
    try:
        import os
        os.chmod(path, 0o600)
    except OSError:
        pass
