"""Comprehensive unit tests for tyagent.config."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from tyagent.config import (
    AgentConfig,
    CompressionConfig,
    PlatformConfig,
    TyAgentConfig,
    default_home,
    default_workspace,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# PlatformConfig
# ---------------------------------------------------------------------------

class TestPlatformConfigDefaults:
    """Test PlatformConfig default values."""

    def test_defaults(self):
        cfg = PlatformConfig()
        assert cfg.enabled is False
        assert cfg.token is None
        assert cfg.api_key is None
        assert cfg.extra == {}


class TestPlatformConfigToDict:
    """Test PlatformConfig.to_dict."""

    def test_minimal_defaults(self):
        cfg = PlatformConfig()
        d = cfg.to_dict()
        assert d == {"enabled": False, "extra": {}}

    def test_with_token(self):
        cfg = PlatformConfig(token="tok-123")
        d = cfg.to_dict()
        assert d == {"enabled": False, "extra": {}, "token": "tok-123"}

    def test_with_api_key(self):
        cfg = PlatformConfig(api_key="key-456")
        d = cfg.to_dict()
        assert d == {"enabled": False, "extra": {}, "api_key": "key-456"}

    def test_with_all_fields(self):
        cfg = PlatformConfig(
            enabled=True, token="tok", api_key="key", extra={"app_id": "xyz"}
        )
        d = cfg.to_dict()
        assert d == {
            "enabled": True,
            "token": "tok",
            "api_key": "key",
            "extra": {"app_id": "xyz"},
        }

    def test_extra_with_nested_data(self):
        cfg = PlatformConfig(extra={"nested": {"a": 1, "b": [2, 3]}})
        d = cfg.to_dict()
        assert d["extra"] == {"nested": {"a": 1, "b": [2, 3]}}


class TestPlatformConfigFromDict:
    """Test PlatformConfig.from_dict."""

    def test_empty_dict(self):
        cfg = PlatformConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.token is None
        assert cfg.api_key is None
        assert cfg.extra == {}

    def test_full_dict(self):
        data = {
            "enabled": True,
            "token": "tok",
            "api_key": "key",
            "extra": {"app_id": "abc"},
        }
        cfg = PlatformConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.token == "tok"
        assert cfg.api_key == "key"
        assert cfg.extra == {"app_id": "abc"}

    def test_partial_dict(self):
        data = {"enabled": True}
        cfg = PlatformConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.token is None
        assert cfg.api_key is None
        assert cfg.extra == {}

    def test_roundtrip(self):
        original = PlatformConfig(enabled=True, token="tok", api_key="key", extra={"x": 1})
        restored = PlatformConfig.from_dict(original.to_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

class TestAgentConfigDefaults:
    """Test AgentConfig default values."""

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model == "anthropic/claude-sonnet-4"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.max_tool_turns == 200
        assert cfg.system_prompt == "You are a helpful assistant."


class TestAgentConfigToDict:
    """Test AgentConfig.to_dict."""

    def test_defaults_to_dict(self):
        cfg = AgentConfig()
        d = cfg.to_dict()
        assert d == {
            "model": "anthropic/claude-sonnet-4",
            "api_key": None,
            "base_url": None,
            "max_tool_turns": 200,
            "system_prompt": "You are a helpful assistant.",
            "reasoning_effort": "high",
        }

    def test_custom_to_dict(self):
        cfg = AgentConfig(
            model="openai/gpt-4o",
            api_key="sk-test",
            base_url="https://api.example.com",
            max_tool_turns=100,
            system_prompt="Be concise.",
        )
        d = cfg.to_dict()
        assert d["model"] == "openai/gpt-4o"
        assert d["api_key"] == "sk-test"
        assert d["base_url"] == "https://api.example.com"
        assert d["max_tool_turns"] == 100
        assert d["system_prompt"] == "Be concise."


class TestAgentConfigFromDict:
    """Test AgentConfig.from_dict."""

    def test_empty_dict(self):
        cfg = AgentConfig.from_dict({})
        assert cfg.model == "anthropic/claude-sonnet-4"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.max_tool_turns == 200
        assert cfg.system_prompt == "You are a helpful assistant."

    def test_full_dict(self):
        data = {
            "model": "openai/gpt-4o",
            "api_key": "sk-test",
            "base_url": "https://api.example.com",
            "max_tool_turns": 100,
            "system_prompt": "Be concise.",
        }
        cfg = AgentConfig.from_dict(data)
        assert cfg.model == "openai/gpt-4o"
        assert cfg.api_key == "sk-test"
        assert cfg.base_url == "https://api.example.com"
        assert cfg.max_tool_turns == 100
        assert cfg.system_prompt == "Be concise."

    def test_partial_dict(self):
        data = {"model": "custom-model", "max_tool_turns": 20}
        cfg = AgentConfig.from_dict(data)
        assert cfg.model == "custom-model"
        assert cfg.api_key is None
        assert cfg.max_tool_turns == 20
        assert cfg.system_prompt == "You are a helpful assistant."

    def test_roundtrip(self):
        original = AgentConfig(
            model="openai/gpt-4o",
            api_key="sk-test",
            base_url="https://api.example.com",
            max_tool_turns=75,
            system_prompt="Custom prompt",
        )
        restored = AgentConfig.from_dict(original.to_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# TyAgentConfig
# ---------------------------------------------------------------------------

class TestTyAgentConfigDefaults:
    """Test TyAgentConfig default values."""

    def test_defaults(self):
        cfg = TyAgentConfig()
        assert cfg.platforms == {}
        assert isinstance(cfg.agent, AgentConfig)
        assert cfg.home_dir == default_home
        assert cfg.workspace_dir == default_workspace
        assert cfg.sessions_dir == default_home / "sessions"
        assert cfg.log_level == "INFO"
        assert cfg.reset_triggers == ["new", "reset"]


class TestTyAgentConfigToDict:
    """Test TyAgentConfig.to_dict."""

    def test_defaults_to_dict(self):
        cfg = TyAgentConfig()
        d = cfg.to_dict()
        assert d["platforms"] == {}
        assert d["agent"] == AgentConfig().to_dict()
        assert d["compression"] == CompressionConfig().to_dict()
        assert d["home_dir"] == str(default_home)
        assert d["workspace_dir"] == str(default_workspace)
        assert d["sessions_dir"] == str(default_home / "sessions")
        assert d["log_level"] == "INFO"
        assert d["reset_triggers"] == ["new", "reset"]

    def test_with_platforms(self):
        pc = PlatformConfig(enabled=True, token="tok")
        cfg = TyAgentConfig(platforms={"slack": pc})
        d = cfg.to_dict()
        assert d["platforms"]["slack"] == pc.to_dict()

    def test_custom_agent(self):
        ac = AgentConfig(model="test-model", max_tool_turns=10)
        cfg = TyAgentConfig(agent=ac)
        d = cfg.to_dict()
        assert d["agent"]["model"] == "test-model"
        assert d["agent"]["max_tool_turns"] == 10

    def test_custom_dirs(self):
        cfg = TyAgentConfig(
            home_dir=Path("/tmp/tyhome"),
            workspace_dir=Path("/tmp/tywork"),
            sessions_dir=Path("/tmp/tyhome/sessions"),
        )
        d = cfg.to_dict()
        assert d["home_dir"] == "/tmp/tyhome"
        assert d["workspace_dir"] == "/tmp/tywork"
        assert d["sessions_dir"] == "/tmp/tyhome/sessions"


class TestTyAgentConfigFromDict:
    """Test TyAgentConfig.from_dict."""

    def test_empty_dict(self):
        cfg = TyAgentConfig.from_dict({})
        assert cfg.platforms == {}
        assert cfg.agent == AgentConfig()
        assert cfg.log_level == "INFO"
        assert cfg.reset_triggers == ["new", "reset"]

    def test_full_dict(self):
        data = {
            "platforms": {
                "slack": {"enabled": True, "token": "slack-tok", "extra": {}},
                "discord": {"enabled": False, "api_key": "dk", "extra": {}},
            },
            "agent": {"model": "gpt-4o", "max_tool_turns": 30, "system_prompt": "Hi"},
            "home_dir": "/opt/ty",
            "workspace_dir": "/opt/ty/work",
            "sessions_dir": "/opt/ty/sessions",
            "log_level": "DEBUG",
            "reset_triggers": ["restart"],
        }
        cfg = TyAgentConfig.from_dict(data)
        assert "slack" in cfg.platforms
        assert cfg.platforms["slack"].enabled is True
        assert cfg.platforms["slack"].token == "slack-tok"
        assert cfg.platforms["discord"].api_key == "dk"
        assert cfg.agent.model == "gpt-4o"
        assert cfg.agent.max_tool_turns == 30
        assert cfg.home_dir == Path("/opt/ty")
        assert cfg.workspace_dir == Path("/opt/ty/work")
        assert cfg.sessions_dir == Path("/opt/ty/sessions")
        assert cfg.log_level == "DEBUG"
        assert cfg.reset_triggers == ["restart"]

    def test_partial_dict(self):
        data = {"log_level": "WARNING"}
        cfg = TyAgentConfig.from_dict(data)
        assert cfg.log_level == "WARNING"
        assert cfg.agent == AgentConfig()

    def test_roundtrip(self):
        original = TyAgentConfig(
            platforms={"slack": PlatformConfig(enabled=True, token="tok")},
            agent=AgentConfig(model="test-model"),
            home_dir=Path("/tmp/tyhome"),
            workspace_dir=Path("/tmp/tywork"),
            sessions_dir=Path("/tmp/tyhome/sessions"),
            log_level="DEBUG",
            reset_triggers=["reset"],
        )
        restored = TyAgentConfig.from_dict(original.to_dict())
        assert restored == original


class TestTyAgentConfigGetPlatform:
    """Test TyAgentConfig.get_platform."""

    def test_existing_platform(self):
        pc = PlatformConfig(enabled=True, token="tok")
        cfg = TyAgentConfig(platforms={"slack": pc})
        result = cfg.get_platform("slack")
        assert result is pc

    def test_missing_platform(self):
        cfg = TyAgentConfig(platforms={"slack": PlatformConfig()})
        result = cfg.get_platform("discord")
        assert result is None

    def test_empty_platforms(self):
        cfg = TyAgentConfig()
        assert cfg.get_platform("anything") is None


class TestTyAgentConfigGetConnectedPlatforms:
    """Test TyAgentConfig.get_connected_platforms."""

    def test_empty_platforms(self):
        cfg = TyAgentConfig()
        assert cfg.get_connected_platforms() == []

    def test_disabled_platform_excluded(self):
        cfg = TyAgentConfig(
            platforms={"slack": PlatformConfig(enabled=False, token="tok")}
        )
        assert cfg.get_connected_platforms() == []

    def test_enabled_with_token(self):
        cfg = TyAgentConfig(
            platforms={"slack": PlatformConfig(enabled=True, token="tok")}
        )
        assert cfg.get_connected_platforms() == ["slack"]

    def test_enabled_with_api_key(self):
        cfg = TyAgentConfig(
            platforms={"slack": PlatformConfig(enabled=True, api_key="key")}
        )
        assert cfg.get_connected_platforms() == ["slack"]

    def test_enabled_no_token_no_key_excluded(self):
        cfg = TyAgentConfig(
            platforms={"slack": PlatformConfig(enabled=True)}
        )
        assert cfg.get_connected_platforms() == []

    def test_feishu_with_app_id(self):
        cfg = TyAgentConfig(
            platforms={
                "feishu": PlatformConfig(
                    enabled=True, extra={"app_id": "abc123"}
                )
            }
        )
        assert cfg.get_connected_platforms() == ["feishu"]

    def test_feishu_without_app_id_excluded(self):
        cfg = TyAgentConfig(
            platforms={
                "feishu": PlatformConfig(enabled=True, extra={})
            }
        )
        assert cfg.get_connected_platforms() == []

    def test_feishu_with_token_not_app_id(self):
        """Feishu with a token but no app_id should be connected via the token path."""
        cfg = TyAgentConfig(
            platforms={
                "feishu": PlatformConfig(enabled=True, token="tok")
            }
        )
        assert cfg.get_connected_platforms() == ["feishu"]

    def test_multiple_platforms_mixed(self):
        cfg = TyAgentConfig(
            platforms={
                "slack": PlatformConfig(enabled=True, token="tok"),
                "discord": PlatformConfig(enabled=False, token="tok"),
                "telegram": PlatformConfig(enabled=True),
                "wechat": PlatformConfig(enabled=True, api_key="key"),
            }
        )
        result = cfg.get_connected_platforms()
        assert "slack" in result
        assert "wechat" in result
        assert "discord" not in result
        assert "telegram" not in result


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Test load_config with various file formats and edge cases."""

    def test_load_yaml_file(self, tmp_path):
        config_data = {
            "platforms": {
                "slack": {"enabled": True, "token": "slack-tok", "extra": {}},
            },
            "agent": {"model": "test-model", "max_tool_turns": 10},
            "log_level": "DEBUG",
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data), encoding="utf-8")

        cfg = load_config(yaml_path)
        assert isinstance(cfg, TyAgentConfig)
        assert "slack" in cfg.platforms
        assert cfg.platforms["slack"].token == "slack-tok"
        assert cfg.agent.model == "test-model"
        assert cfg.log_level == "DEBUG"

    def test_load_yml_extension(self, tmp_path):
        config_data = {"log_level": "WARNING"}
        yml_path = tmp_path / "config.yml"
        yml_path.write_text(yaml.dump(config_data), encoding="utf-8")

        cfg = load_config(yml_path)
        assert cfg.log_level == "WARNING"

    def test_load_json_file(self, tmp_path):
        config_data = {
            "platforms": {
                "discord": {"enabled": True, "api_key": "dk", "extra": {}},
            },
            "agent": {"model": "json-model"},
            "log_level": "ERROR",
        }
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps(config_data), encoding="utf-8")

        cfg = load_config(json_path)
        assert isinstance(cfg, TyAgentConfig)
        assert "discord" in cfg.platforms
        assert cfg.platforms["discord"].api_key == "dk"
        assert cfg.agent.model == "json-model"
        assert cfg.log_level == "ERROR"

    def test_load_missing_file_raises(self, tmp_path):
        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(missing_path)

    def test_load_no_path_no_default_files(self, tmp_path, monkeypatch):
        """When no config_path given and no default files exist, return defaults."""
        monkeypatch.setattr("tyagent.config.default_home", tmp_path)
        cfg = load_config()
        assert isinstance(cfg, TyAgentConfig)
        assert cfg.agent == AgentConfig()
        assert cfg.platforms == {}

    def test_load_defaults_from_yaml_in_home(self, tmp_path, monkeypatch):
        """When no config_path given, should find config.yaml in default_home."""
        monkeypatch.setattr("tyagent.config.default_home", tmp_path)
        config_data = {"log_level": "CUSTOM"}
        (tmp_path / "config.yaml").write_text(
            yaml.dump(config_data), encoding="utf-8"
        )
        cfg = load_config()
        assert cfg.log_level == "CUSTOM"

    def test_load_defaults_from_json_in_home(self, tmp_path, monkeypatch):
        """When no config_path given, should find config.json in default_home."""
        monkeypatch.setattr("tyagent.config.default_home", tmp_path)
        config_data = {"log_level": "CUSTOM_JSON"}
        (tmp_path / "config.json").write_text(
            json.dumps(config_data), encoding="utf-8"
        )
        cfg = load_config()
        assert cfg.log_level == "CUSTOM_JSON"

    def test_yaml_takes_priority_over_json(self, tmp_path, monkeypatch):
        """When both config.yaml and config.json exist, yaml should win."""
        monkeypatch.setattr("tyagent.config.default_home", tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"log_level": "FROM_YAML"}), encoding="utf-8"
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"log_level": "FROM_JSON"}), encoding="utf-8"
        )
        cfg = load_config()
        assert cfg.log_level == "FROM_YAML"

    def test_load_empty_yaml(self, tmp_path):
        """An empty YAML file should produce default config."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("", encoding="utf-8")
        cfg = load_config(yaml_path)
        assert cfg.agent == AgentConfig()


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

class TestSaveConfig:
    """Test save_config output format and permissions."""

    def test_save_creates_yaml_file(self, tmp_path):
        cfg = TyAgentConfig(log_level="DEBUG")
        out_path = tmp_path / "output" / "config.yaml"
        save_config(cfg, out_path)

        assert out_path.exists()
        with open(out_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["log_level"] == "DEBUG"

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = TyAgentConfig()
        out_path = tmp_path / "deep" / "nested" / "config.yaml"
        save_config(cfg, out_path)
        assert out_path.exists()

    def test_save_roundtrip(self, tmp_path):
        original = TyAgentConfig(
            platforms={"slack": PlatformConfig(enabled=True, token="tok")},
            agent=AgentConfig(model="roundtrip-model", max_tool_turns=5),
            home_dir=Path("/tmp/tyhome"),
            workspace_dir=Path("/tmp/tywork"),
            sessions_dir=Path("/tmp/tyhome/sessions"),
            log_level="WARNING",
            reset_triggers=["go"],
        )
        out_path = tmp_path / "config.yaml"
        save_config(original, out_path)

        loaded = load_config(out_path)
        assert loaded == original

    def test_save_permissions_0600(self, tmp_path):
        cfg = TyAgentConfig()
        out_path = tmp_path / "config.yaml"
        save_config(cfg, out_path)

        file_stat = os.stat(out_path)
        file_mode = stat.S_IMODE(file_stat.st_mode)
        assert file_mode == 0o600

    def test_save_default_path_uses_home_dir(self, tmp_path, monkeypatch):
        """When no path is given, save to home_dir/config.yaml."""
        home = tmp_path / "tyhome"
        cfg = TyAgentConfig(home_dir=home)
        save_config(cfg)

        expected = home / "config.yaml"
        assert expected.exists()

    def test_save_valid_yaml_content(self, tmp_path):
        cfg = TyAgentConfig(
            platforms={
                "slack": PlatformConfig(enabled=True, token="tok", extra={"channel": "#general"})
            },
            agent=AgentConfig(system_prompt="Hello world"),
        )
        out_path = tmp_path / "config.yaml"
        save_config(cfg, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["platforms"]["slack"]["enabled"] is True
        assert data["platforms"]["slack"]["token"] == "tok"
        assert data["platforms"]["slack"]["extra"]["channel"] == "#general"
        assert data["agent"]["system_prompt"] == "Hello world"

    def test_save_unicode_content(self, tmp_path):
        cfg = TyAgentConfig(
            agent=AgentConfig(system_prompt="你好世界 🌍"),
        )
        out_path = tmp_path / "config.yaml"
        save_config(cfg, out_path)

        loaded = load_config(out_path)
        assert loaded.agent.system_prompt == "你好世界 🌍"
