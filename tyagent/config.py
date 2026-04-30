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

_usr_home = Path(os.path.expanduser("~"))
try:
    import pwd
    _usr_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
except (ImportError, KeyError):
    pass
default_home = _usr_home / ".tyagent"
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
class CompressionConfig:
    """Configuration for context compression (LLM summarization on overflow).

    All fields optional — None falls back to the agent model (model/api_key/base_url).
    context_window and cut_ratio control the single-pass compression cut point.
    """
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    context_window: int = 128000
    cut_ratio: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.model:
            d["model"] = self.model
        if self.api_key:
            d["api_key"] = self.api_key
        if self.base_url:
            d["base_url"] = self.base_url
        d["context_window"] = self.context_window
        d["cut_ratio"] = self.cut_ratio
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompressionConfig:
        return cls(
            model=data.get("model"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            context_window=int(_v) if (_v := data.get("context_window")) is not None else 128000,
            cut_ratio=float(_v) if (_v := data.get("cut_ratio")) is not None else 0.5,
        )


@dataclass
class AgentConfig:
    """Configuration for the AI agent."""
    model: str = "anthropic/claude-sonnet-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tool_turns: Optional[int] = 200  # None = no limit
    system_prompt: str = "You are a helpful assistant."
    reasoning_effort: Optional[str] = "high"  # None/"" = don't send

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
        }
        if self.max_tool_turns is not None:
            d["max_tool_turns"] = self.max_tool_turns
        if self.reasoning_effort:
            d["reasoning_effort"] = self.reasoning_effort
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        return cls(
            model=data.get("model", "anthropic/claude-sonnet-4"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            max_tool_turns=data.get("max_tool_turns", 200),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            reasoning_effort=data.get("reasoning_effort", "high"),
        )


@dataclass
class TyAgentConfig:
    """Main configuration for tyagent."""
    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)
    agent: AgentConfig = field(default_factory=AgentConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    home_dir: Path = field(default_factory=lambda: default_home)
    workspace_dir: Path = field(default_factory=lambda: default_workspace)
    sessions_dir: Path = field(default_factory=lambda: default_home / "sessions")
    log_level: str = "INFO"
    reset_triggers: List[str] = field(default_factory=lambda: ["new", "reset"])

    def get_platform(self, name: str) -> Optional[PlatformConfig]:
        return self.platforms.get(name)

    def get_connected_platforms(self) -> List[str]:
        connected = []
        for name, cfg in self.platforms.items():
            if not cfg.enabled:
                continue
            if name == "feishu" and cfg.extra.get("app_id"):
                connected.append(name)
            elif cfg.token or cfg.api_key:
                connected.append(name)
        return connected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
            "agent": self.agent.to_dict(),
            "compression": self.compression.to_dict(),
            "home_dir": str(self.home_dir),
            "workspace_dir": str(self.workspace_dir),
            "sessions_dir": str(self.sessions_dir),
            "log_level": self.log_level,
            "reset_triggers": self.reset_triggers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TyAgentConfig:
        platforms = {}
        platform_data = data.get("platforms") or {}
        for name, pdata in platform_data.items():
            platforms[name] = PlatformConfig.from_dict(pdata)
        home = Path(data.get("home_dir", str(default_home)))
        workspace = Path(data.get("workspace_dir", str(default_workspace)))
        return cls(
            platforms=platforms,
            agent=AgentConfig.from_dict(data.get("agent") or {}),
            compression=CompressionConfig.from_dict(data.get("compression") or {}),
            home_dir=home,
            workspace_dir=workspace,
            sessions_dir=Path(data.get("sessions_dir", str(home / "sessions"))),
            log_level=data.get("log_level", "INFO"),
            reset_triggers=data.get("reset_triggers", ["new", "reset"]),
        )


def load_config(config_path: Optional[Path] = None) -> TyAgentConfig:
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
    if path is None:
        path = config.home_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
