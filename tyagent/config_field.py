"""Schema descriptors for tyagent configuration.

Provides ``ConfigField`` dataclass and utilities for converting between
schema declarations (declared by each platform adapter) and runtime config
values (loaded from ``config.yaml``).

Architecture
------------
Each ``BasePlatformAdapter`` subclass declares a ``config_schema`` dict whose
structure mirrors the platform's section in ``config.yaml``:

.. code-block:: python

    class FeishuAdapter(BasePlatformAdapter):
        config_schema = {
            "enabled": ConfigField(bool, default=False, doc="启用此平台"),
            "extra": {
                "app_id": ConfigField(str, required=True, doc="App ID"),
                "app_secret": ConfigField(str, secret=True, doc="App Secret"),
            },
        }

``schema_to_defaults()`` converts these declarations into a default-values
dict used by ``_deep_merge_defaults()`` to auto-fill ``config.yaml``.

``validate_config()`` walks a loaded config section against its schema and
returns human-readable errors, caught at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union


# ── Schema descriptor ───────────────────────────────────────────────────────


@dataclass
class ConfigField:
    """Descriptor for a single configuration leaf field.

    Each field in a platform adapter's ``config_schema`` is described by
    one of these, providing default, validation rules, documentation, and
    security classification.

    Attributes:
        type: Expected Python type (for startup validation).
        default: Default value when the field is absent from user config.
        required: If True, startup logs a warning when the field is missing
            (does NOT abort — usable defaults are assumed).
        secret: If True, the value is stripped before writing config.yaml
            and should be supplied via environment variable or .env.
        choices: Restricted set of allowed values (validated on startup).
        doc: Human-readable description shown in generated config comments.
    """
    type: Optional[Type] = None
    default: Any = field(default=None)
    required: bool = False
    secret: bool = False
    choices: Optional[List[Any]] = None
    doc: str = ""


# Recursive type: schema is a dict where values are either ConfigField or
# nested dicts mirroring the config structure.
SchemaDict = Dict[str, Union[ConfigField, "SchemaDict"]]


# ── Schema → defaults dict ───────────────────────────────────────────────────


def schema_to_defaults(schema: SchemaDict) -> Dict[str, Any]:
    """Convert a schema definition into a plain defaults dict.

    ``ConfigField`` leaves become their ``.default`` value.  For str-typed
    fields with ``None`` default, ``""`` is used instead — this ensures they
    render as ``''`` in YAML rather than ``~``, which is more intuitive
    (\"this field is intentionally empty\").  Nested dicts are recursed.

    The result mirrors the config structure exactly so it can be passed to
    ``_deep_merge_defaults()``.
    """
    result: Dict[str, Any] = {}
    for key, field in schema.items():
        if isinstance(field, ConfigField):
            val = field.default if field.default is not None else None
            # Display str fields with None default as "" (→ '' in YAML)
            # rather than None (→ ~ in YAML).  Non-string fields where None
            # carries semantic meaning (auto-detect, unset) keep ~.
            if val is None and field.type is str:
                val = ""
            result[key] = val
        elif isinstance(field, dict):
            result[key] = schema_to_defaults(field)
        else:
            result[key] = None
    return result


def schema_from_config(config: Dict[str, Any]) -> SchemaDict:
    """Infer a flat ``SchemaDict`` from an existing config section.

    Every leaf value becomes ``ConfigField(type=type(value))`` with no
    default, safety, or documentation metadata.  Used for platforms that
    exist in the user's config but have no adapter — the inferred schema
    preserves the config structure during merge without losing data.
    """
    result: SchemaDict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = schema_from_config(value)
        else:
            field_type: Optional[Type] = None
            if value is not None:
                field_type = type(value)
            result[key] = ConfigField(type=field_type, default=value)
    return result


# ── Validation ───────────────────────────────────────────────────────────────


def validate_config(
    schema: SchemaDict,
    config: Dict[str, Any],
    path: str = "",
) -> List[str]:
    """Walk *config* against *schema* and return human-readable errors.

    Checks:
      - Required fields are present and non-``None`` / non-empty.
      - Value type matches the schema type when ``ConfigField.type`` is set.
      - Value is within ``ConfigField.choices`` when set.

    Returns an empty list when the config is valid.
    """
    errors: List[str] = []
    for key, field in schema.items():
        full_path = f"{path}.{key}" if path else key

        if isinstance(field, ConfigField):
            # Treat absent keys and explicit None/empty the same way
            value = config.get(key)
            no_value = value is None or value == ""

            if field.required and no_value:
                errors.append(f"{full_path}: required but missing")
                continue

            if not no_value:
                if field.type is not None and not isinstance(value, field.type):
                    errors.append(
                        f"{full_path}: expected {field.type.__name__}, "
                        f"got {type(value).__name__} ({value!r})"
                    )
                if field.choices and value not in field.choices:
                    errors.append(
                        f"{full_path}: must be one of {field.choices}, "
                        f"got {value!r}"
                    )

        elif isinstance(field, dict):
            sub = config.get(key)
            if isinstance(sub, dict):
                errors.extend(validate_config(field, sub, full_path))
            elif sub is not None:
                errors.append(
                    f"{full_path}: expected dict, got {type(sub).__name__}"
                )

    return errors


# ── Secret handling ──────────────────────────────────────────────────────────


def collect_secrets(
    schema: SchemaDict,
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, str]:
    """Extract values of all secret-marked fields from *config*.

    Returns a ``{ENV_VAR_NAME: value}`` dict.  Env-var names are
    uppercase with underscores: ``{PREFIX}_{KEY}``.
    """
    secrets: Dict[str, str] = {}
    for key, field in schema.items():
        env_key = f"{prefix}_{key}".upper() if prefix else key.upper()

        if isinstance(field, ConfigField):
            if field.secret and key in config and config[key]:
                secrets[env_key] = str(config[key])

        elif isinstance(field, dict):
            sub = config.get(key, {})
            if isinstance(sub, dict):
                secrets.update(collect_secrets(field, sub, env_key))

    return secrets


# ── Display helpers ──────────────────────────────────────────────────────────


def format_schema_as_yaml_comment(schema: SchemaDict, indent: int = 0) -> str:
    """Render a schema as human-readable YAML-format doc lines.

    Each line is prefixed with ``# `` and indented.  Useful for generating
    annotated config samples.
    """
    lines: List[str] = []
    pad = "  " * indent
    for key, field in schema.items():
        if isinstance(field, ConfigField):
            meta_parts: List[str] = []
            if field.secret:
                meta_parts.append("secret")
            if field.required:
                meta_parts.append("required")
            if field.choices:
                meta_parts.append("options: " + ", ".join(f"{v!r}" for v in field.choices))
            meta = f"  [{', '.join(meta_parts)}]" if meta_parts else ""
            doc = field.doc or ""
            if doc or meta:
                lines.append(f"{pad}# {doc}{meta}".rstrip())
            lines.append(f"{pad}#   {key}: {field.default!r}")
        elif isinstance(field, dict):
            lines.append(f"{pad}# {key}:")
            lines.append(format_schema_as_yaml_comment(field, indent + 1))
    return "\n".join(lines)
