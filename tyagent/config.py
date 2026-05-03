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
DEFAULT_PROFILE = "tyagent"
default_home = _usr_home / ".tyagent" / DEFAULT_PROFILE

# Canonical config schema — every key that should exist in config.yaml.
# Used at startup to auto-fill missing fields (user values are never overwritten).
# Only user-configurable leaf fields are listed; runtime-derived values (like
# context_window from the model) stay in dataclass constructors only.
DEFAULT_CONFIG: Dict[str, Any] = {
    "platforms": {},
    "agent": {
        "provider": "deepseek",
        "model": "anthropic/claude-sonnet-4",
        "max_tool_turns": 200,
        "reasoning_effort": "high",
        "system_prompt": "You are a helpful assistant.",
    },
    "compression": {
        "cut_ratio": 0.5,
    },
    "workspace": {
        "lock": "off",
    },
    "log_level": "INFO",
    "reset_triggers": ["new"],
}


def _deep_merge_defaults(user: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
    """Recursively add missing keys from *defaults* into *user* dict in-place.

    Returns True if any key was added (caller should re-save config.yaml).
    Never overwrites an existing user value — only fills absent keys.
    """
    changed = False
    for key, default_val in defaults.items():
        if key not in user:
            user[key] = default_val
            changed = True
        elif isinstance(default_val, dict) and isinstance(user[key], dict):
            if _deep_merge_defaults(user[key], default_val):
                changed = True
    return changed


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
    cut_ratio controls the single-pass compression cut point: the cut lands at
    roughly ``context_length * cut_ratio`` tokens into the conversation.
    """
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cut_ratio: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.model:
            d["model"] = self.model
        if self.api_key:
            d["api_key"] = self.api_key
        if self.base_url:
            d["base_url"] = self.base_url
        d["cut_ratio"] = self.cut_ratio
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompressionConfig:
        return cls(
            model=data.get("model"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            cut_ratio=float(_v) if (_v := data.get("cut_ratio")) is not None else 0.5,
        )


@dataclass
class WorkspaceConfig:
    """Workspace (working directory) configuration.

    lock: "on"  → always use locked_directory (error if missing)
          "off" → follow session state; restore last cwd from state file
    locked_directory: only meaningful when lock is "on"
    """

    lock: str = "off"
    locked_directory: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lock not in ("on", "off"):
            logger.warning(
                "workspace.lock must be 'on' or 'off', got %r — treating as 'off'",
                self.lock,
            )
            object.__setattr__(self, "lock", "off")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"lock": self.lock}
        if self.locked_directory:
            d["locked_directory"] = self.locked_directory
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkspaceConfig:
        lock = data.get("lock", "off")
        if lock not in ("on", "off"):
            logger.warning(
                "workspace.lock must be 'on' or 'off', got %r — treating as 'off'",
                lock,
            )
            lock = "off"
        return cls(
            lock=lock,
            locked_directory=data.get("locked_directory"),
        )


@dataclass
class AgentConfig:
    """Configuration for the AI agent.

    api_key lives in ``home_dir/.env`` as TYAGENT_API_KEY, not in config.yaml.
    context_length defaults to None (auto-detect from model when possible).
    """
    provider: str = "deepseek"
    model: str = "anthropic/claude-sonnet-4"
    base_url: Optional[str] = None
    max_tool_turns: Optional[int] = 200  # None = no limit
    system_prompt: str = "You are a helpful assistant."
    reasoning_effort: Optional[str] = "high"  # None/"" = don't send
    context_length: Optional[int] = None

    # ―― runtime only, set by CLI / gateway after .env loading ―――
    api_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
        }
        if self.max_tool_turns is not None:
            d["max_tool_turns"] = self.max_tool_turns
        if self.reasoning_effort:
            d["reasoning_effort"] = self.reasoning_effort
        if self.context_length is not None:
            d["context_length"] = self.context_length
        # api_key deliberately excluded — lives in .env
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        cl = data.get("context_length")
        return cls(
            provider=data.get("provider", "deepseek"),
            model=data.get("model", "anthropic/claude-sonnet-4"),
            base_url=data.get("base_url"),
            max_tool_turns=data.get("max_tool_turns", 200),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            reasoning_effort=data.get("reasoning_effort", "high"),
            context_length=int(cl) if cl is not None else None,
            # api_key is loaded from .env, not config.yaml
        )


@dataclass
class TyAgentConfig:
    """Main configuration for tyagent."""
    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)
    agent: AgentConfig = field(default_factory=AgentConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    home_dir: Path = field(default_factory=lambda: default_home)
    sessions_dir: Path = field(default_factory=lambda: default_home / "sessions")
    log_level: str = "INFO"
    reset_triggers: List[str] = field(default_factory=lambda: ["new"])

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
            "workspace": self.workspace.to_dict(),
            "home_dir": str(self.home_dir),
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
        ws_data = data.get("workspace") or {}
        return cls(
            platforms=platforms,
            agent=AgentConfig.from_dict(data.get("agent") or {}),
            compression=CompressionConfig.from_dict(data.get("compression") or {}),
            workspace=WorkspaceConfig.from_dict(ws_data),
            home_dir=home,
            sessions_dir=Path(data.get("sessions_dir", str(home / "sessions"))),
            log_level=data.get("log_level", "INFO"),
            reset_triggers=data.get("reset_triggers", ["new"]),
        )


def _load_profile_env(profile_dir: Path) -> None:
    """Load ``profile_dir/.env`` into os.environ (if it exists)."""
    env_path = profile_dir / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        # Fallback: simple KEY=VALUE parser
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key:
                os.environ[key] = value


def _migrate_api_key_to_env(raw: Dict[str, Any], profile_dir: Path) -> bool:
    """One-time: move ``agent.api_key`` from config.yaml → .env.

    Returns True if migration was performed.
    """
    agent_section = raw.get("agent") or {}
    if not isinstance(agent_section, dict):
        return False
    api_key = agent_section.pop("api_key", None)
    if not api_key:
        return False
    env_path = profile_dir / ".env"
    existing_lines: List[str] = []
    has_key = False
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TYAGENT_API_KEY="):
                has_key = True
            existing_lines.append(line)
    if has_key:
        # .env already has the key — just remove from config
        return True
    existing_lines.append(f"TYAGENT_API_KEY={api_key}")
    env_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    return True


def load_config(config_path: Optional[Path] = None, profile: Optional[str] = None) -> TyAgentConfig:
    """Load tyagent configuration.

    Precedence:
    1. *config_path* — explicit file path
    2. *profile* — ~/.tyagent/<profile>/config.yaml
    3. default — ~/.tyagent/tyagent/config.yaml

    Side-effects: loads .env, auto-fills missing schema keys, migrates
    api_key from config.yaml → .env on first encounter.
    """
    if config_path:
        return _load_from_path(config_path)

    if profile:
        profile_dir = _usr_home / ".tyagent" / profile
    else:
        profile_dir = default_home

    # ── Load .env (API keys) ───────────────────────────────────
    _load_profile_env(profile_dir)

    yaml_path = profile_dir / "config.yaml"
    json_path = profile_dir / "config.json"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # ── Migrate api_key from config.yaml → .env ────────────
        if _migrate_api_key_to_env(raw, profile_dir):
            _yaml_dump(raw, yaml_path)
            logger.info("Migrated api_key to .env")
        if _deep_merge_defaults(raw, DEFAULT_CONFIG):
            _yaml_dump(raw, yaml_path)
            logger.info("Config schema updated with new defaults.")
        cfg = TyAgentConfig.from_dict(raw)
        # Override home_dir / sessions_dir to use the actual profile
        # directory, not whatever is stored in the config file.
        cfg.home_dir = profile_dir
        cfg.sessions_dir = profile_dir / "sessions"
        migrate_legacy_home(cfg.home_dir)
        return cfg
    if json_path.exists():
        cfg = _load_from_path(json_path)
        cfg.home_dir = profile_dir
        cfg.sessions_dir = profile_dir / "sessions"
        migrate_legacy_home(cfg.home_dir)
        return cfg
    logger.info("No config file found for profile %s, using defaults.", profile or DEFAULT_PROFILE)
    cfg = TyAgentConfig()
    cfg.home_dir = profile_dir
    cfg.sessions_dir = profile_dir / "sessions"
    # Check for legacy migration on first use
    migrate_legacy_home(cfg.home_dir)
    return cfg


def _load_from_path(path: Path) -> TyAgentConfig:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    return TyAgentConfig.from_dict(data)


def migrate_legacy_home(home_dir: Path) -> None:
    """One-time migration: old flat ~/.tyagent/ → new profile directory.

    Only triggers when:
      - Legacy ~/.tyagent/config.yaml exists
      - Target home_dir/config.yaml does NOT exist
      - home_dir is the default profile (not a custom path or other profile)

    Safe to call from any CLI entry point; no-op if conditions not met.
    """
    import shutil

    if home_dir != default_home:
        return  # not the default profile

    legacy_home = default_home.parent  # ~/.tyagent/
    legacy_config = legacy_home / "config.yaml"
    if not legacy_config.exists():
        return

    if (home_dir / "config.yaml").exists():
        return  # target already has config — migration done or not needed

    logger.info("Migrating legacy ~/.tyagent/ → %s/", home_dir)
    home_dir.mkdir(parents=True, exist_ok=True)

    # Move data dirs first, config.yaml last — its presence signals migration done.
    for item in ["memories", "sessions", "cache", "home", ".clean_shutdown", "config.yaml"]:
        src = legacy_home / item
        dst = home_dir / item
        if src.exists() and not dst.exists():
            try:
                shutil.move(str(src), str(dst))
                logger.info("  Moved %s/ → %s/", item, dst)
            except OSError as exc:
                logger.warning("  Failed to move %s: %s", item, exc)


def _yaml_dump(data: Dict[str, Any], path: Path) -> None:
    """Write *data* to *path* as YAML, using '~' for None."""
    class _TyDumper(yaml.Dumper):
        pass

    def _repr_none(self, _data):
        return self.represent_scalar("tag:yaml.org,2002:null", "~")

    _TyDumper.add_representer(type(None), _repr_none)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=_TyDumper, default_flow_style=False, allow_unicode=True)


def save_config(config: TyAgentConfig, path: Optional[Path] = None) -> None:
    if path is None:
        path = config.home_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    _yaml_dump(config.to_dict(), path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
