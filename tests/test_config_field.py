"""Tests for tyagent.config_field — schema descriptors and utilities."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tyagent.config_field import (
    ConfigField,
    SchemaDict,
    collect_secrets,
    format_schema_as_yaml_comment,
    schema_from_config,
    schema_to_defaults,
    validate_config,
)


# ── ConfigField basics ────────────────────────────────────────────────────────


class TestConfigField:
    """ConfigField dataclass construction."""

    def test_default_values(self) -> None:
        """Default constructor produces sensible defaults."""
        f = ConfigField()
        assert f.type is None
        assert f.default is None
        assert f.required is False
        assert f.secret is False
        assert f.choices is None
        assert f.doc == ""

    def test_all_fields(self) -> None:
        """Constructor accepts all keyword args."""
        f = ConfigField(str, default="hello", required=True, secret=True,
                        choices=["a", "b"], doc="A test field")
        assert f.type is str
        assert f.default == "hello"
        assert f.required is True
        assert f.secret is True
        assert f.choices == ["a", "b"]
        assert f.doc == "A test field"


# ── schema_to_defaults ────────────────────────────────────────────────────────


class TestSchemaToDefaults:
    """Converting schemas to default-values dicts."""

    def test_flat_schema(self) -> None:
        """Flat schema with simple fields — str None defaults become ''."""
        schema: SchemaDict = {
            "enabled": ConfigField(bool, default=False),
            "name": ConfigField(str, default="bot"),
            "count": ConfigField(int, default=42),
            "nullable": ConfigField(),  # type=None → stays None
            "unset_str": ConfigField(str),  # type=str, default=None → ''
        }
        result = schema_to_defaults(schema)
        assert result == {
            "enabled": False, "name": "bot", "count": 42,
            "nullable": None, "unset_str": "",
        }

    def test_nested_schema(self) -> None:
        """Nested schema mirrors config structure."""
        schema: SchemaDict = {
            "enabled": ConfigField(bool, default=False),
            "extra": {
                "key1": ConfigField(str, default="v1"),
                "key2": ConfigField(int, default=0),
            },
        }
        result = schema_to_defaults(schema)
        assert result == {"enabled": False, "extra": {"key1": "v1", "key2": 0}}

    def test_str_none_default_becomes_empty(self) -> None:
        """Explicit None default for str-typed field becomes '' for YAML readability."""
        schema: SchemaDict = {"field": ConfigField(str, default=None)}
        assert schema_to_defaults(schema) == {"field": ""}

    def test_empty_schema(self) -> None:
        assert schema_to_defaults({}) == {}

    def test_deep_nesting(self) -> None:
        """Three levels of nesting."""
        schema: SchemaDict = {
            "a": {
                "b": {
                    "c": ConfigField(int, default=1),
                },
            },
        }
        assert schema_to_defaults(schema) == {"a": {"b": {"c": 1}}}


# ── schema_from_config ────────────────────────────────────────────────────────


class TestSchemaFromConfig:
    """Inferring schemas from existing config dicts."""

    def test_flat(self) -> None:
        config = {"enabled": True, "name": "bot", "count": 42}
        result = schema_from_config(config)
        assert isinstance(result["enabled"], ConfigField)
        assert result["enabled"].type is bool
        assert result["enabled"].default is True
        assert result["name"].type is str
        assert result["name"].default == "bot"
        assert result["count"].type is int
        assert result["count"].default == 42

    def test_nested(self) -> None:
        config = {"enabled": False, "extra": {"key1": "val1", "key2": 99}}
        result = schema_from_config(config)
        assert result["enabled"].type is bool
        assert isinstance(result["extra"], dict)
        assert result["extra"]["key1"].type is str
        assert result["extra"]["key2"].type is int

    def test_none_values(self) -> None:
        """Config with None leaf values — type is None."""
        config = {"key": None}
        result = schema_from_config(config)
        assert result["key"].type is None
        assert result["key"].default is None

    def test_empty(self) -> None:
        assert schema_from_config({}) == {}


# ── validate_config ───────────────────────────────────────────────────────────


class TestValidateConfig:
    """Walking a config dict against a schema."""

    def test_valid_config(self) -> None:
        """No errors for a fully correct config."""
        schema: SchemaDict = {
            "enabled": ConfigField(bool, default=False),
            "extra": {
                "name": ConfigField(str, default="bot"),
            },
        }
        config = {"enabled": True, "extra": {"name": "alice"}}
        assert validate_config(schema, config) == []

    def test_missing_optional_field(self) -> None:
        """Optional fields may be absent."""
        schema: SchemaDict = {"name": ConfigField(str, default="bot")}
        assert validate_config(schema, {}) == []

    def test_missing_required_field(self) -> None:
        """Required but missing field produces an error."""
        schema: SchemaDict = {"key": ConfigField(str, required=True)}
        errors = validate_config(schema, {})
        assert len(errors) == 1
        assert "required but missing" in errors[0]

    def test_missing_required_nested(self) -> None:
        """Required field nested inside a dict."""
        schema: SchemaDict = {
            "extra": {
                "id": ConfigField(str, required=True),
            },
        }
        errors = validate_config(schema, {"extra": {}})
        assert len(errors) == 1
        assert "extra.id" in errors[0]

    def test_type_mismatch(self) -> None:
        schema: SchemaDict = {"count": ConfigField(int, default=0)}
        errors = validate_config(schema, {"count": "not_an_int"})
        assert len(errors) == 1
        assert "expected int" in errors[0]

    def test_choices_violation(self) -> None:
        schema: SchemaDict = {
            "mode": ConfigField(str, choices=["a", "b", "c"]),
        }
        errors = validate_config(schema, {"mode": "z"})
        assert len(errors) == 1
        assert "must be one of" in errors[0]
        assert "z" in errors[0]

    def test_valid_choice(self) -> None:
        schema: SchemaDict = {
            "mode": ConfigField(str, choices=["a", "b"]),
        }
        assert validate_config(schema, {"mode": "a"}) == []

    def test_nested_path_in_errors(self) -> None:
        """Error messages include full dotted path."""
        schema: SchemaDict = {
            "outer": {
                "inner": ConfigField(str, required=True),
            },
        }
        errors = validate_config(schema, {"outer": {}})
        assert any("outer.inner" in e for e in errors)

    def test_extra_keys_ignored(self) -> None:
        """Config keys not in the schema are silently ignored."""
        schema: SchemaDict = {"key": ConfigField(str)}
        assert validate_config(schema, {"key": "v", "extra_key": 123}) == []

    def test_repeated_required(self) -> None:
        """Multiple required fields all generate errors."""
        schema: SchemaDict = {
            "a": ConfigField(str, required=True),
            "b": ConfigField(int, required=True),
        }
        errors = validate_config(schema, {"a": "ok"})
        # Only 'b' should be missing; 'a' is present
        assert len(errors) == 1
        assert "b" in errors[0]

    def test_empty_string_as_missing(self) -> None:
        """Empty string is equivalent to missing for required checks."""
        schema: SchemaDict = {"key": ConfigField(str, required=True)}
        errors = validate_config(schema, {"key": ""})
        assert len(errors) == 1

    def test_explicit_none_not_desired(self) -> None:
        """Explicit None is also treated as missing for required."""
        schema: SchemaDict = {"key": ConfigField(str, required=True)}
        errors = validate_config(schema, {"key": None})
        assert len(errors) == 1

    def test_wrong_nested_type(self) -> None:
        """When a nested key should be a dict but is something else."""
        schema: SchemaDict = {
            "extra": {
                "x": ConfigField(int),
            },
        }
        errors = validate_config(schema, {"extra": "not_a_dict"})
        assert len(errors) == 1
        assert "expected dict" in errors[0]


# ── collect_secrets ───────────────────────────────────────────────────────────


class TestCollectSecrets:
    """Extracting secret-marked field values."""

    def test_no_secrets(self) -> None:
        schema: SchemaDict = {"name": ConfigField(str)}
        assert collect_secrets(schema, {"name": "bot"}) == {}

    def test_secret_field_collected(self) -> None:
        schema: SchemaDict = {
            "token": ConfigField(str, secret=True),
        }
        secrets = collect_secrets(schema, {"token": "abc123"})
        assert secrets == {"TOKEN": "abc123"}

    def test_secret_field_missing(self) -> None:
        """Missing secret field does not appear in output."""
        schema: SchemaDict = {"token": ConfigField(str, secret=True)}
        assert collect_secrets(schema, {}) == {}

    def test_nested_secret(self) -> None:
        schema: SchemaDict = {
            "extra": {
                "api_key": ConfigField(str, secret=True),
            },
        }
        secrets = collect_secrets(schema, {"extra": {"api_key": "sk-123"}})
        assert secrets == {"EXTRA_API_KEY": "sk-123"}

    def test_mixed_secret_public(self) -> None:
        """Only secrets are returned, not public fields."""
        schema: SchemaDict = {
            "public": ConfigField(str),
            "secret_key": ConfigField(str, secret=True),
        }
        secrets = collect_secrets(schema, {"public": "x", "secret_key": "s3cr3t"})
        assert secrets == {"SECRET_KEY": "s3cr3t"}
        assert "PUBLIC" not in secrets

    def test_string_conversion(self) -> None:
        """Non-string secrets are converted to str."""
        schema: SchemaDict = {
            "port": ConfigField(int, secret=True),
        }
        secrets = collect_secrets(schema, {"port": 8080})
        assert secrets == {"PORT": "8080"}


# ── format_schema_as_yaml_comment ──────────────────────────────────────────────


class TestFormatSchemaAsYamlComment:
    """Human-readable schema rendering."""

    def test_simple_field(self) -> None:
        schema: SchemaDict = {
            "key": ConfigField(str, default="val", doc="A test key"),
        }
        text = format_schema_as_yaml_comment(schema)
        assert "A test key" in text
        assert "key:" in text

    def test_nested(self) -> None:
        schema: SchemaDict = {
            "extra": {
                "id": ConfigField(str, doc="Identifier"),
            },
        }
        text = format_schema_as_yaml_comment(schema, indent=1)
        assert "extra:" in text
        assert "Identifier" in text
        # Indented lines
        assert "  " in text
