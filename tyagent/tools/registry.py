"""Lightweight tool registry for tyagent.

Each tool self-registers its OpenAI-compatible schema and sync handler.
The registry provides schema retrieval (for LLM API calls) and dispatch
(for execution).  No async bridging — handlers are plain sync functions.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = ("name", "schema", "handler", "description", "emoji")

    def __init__(
        self,
        name: str,
        schema: dict,
        handler: Callable[[Dict[str, Any]], str],
        description: str = "",
        emoji: str = "",
    ):
        self.name = name
        self.schema = schema
        self.handler = handler
        self.description = description or schema.get("description", "")
        self.emoji = emoji


class ToolRegistry:
    """Singleton registry for tyagent tools."""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        schema: dict,
        handler: Callable[[Dict[str, Any]], str],
        description: str = "",
        emoji: str = "",
    ) -> None:
        """Register a tool.

        Args:
            name: Unique tool name (used in function calling).
            schema: OpenAI-format function schema dict (with name, description, parameters).
            handler: Sync function receiving a single dict of args, returning a JSON string.
            description: Optional human-readable description.
            emoji: Optional display emoji.
        """
        with self._lock:
            self._tools[name] = ToolEntry(
                name=name,
                schema=schema,
                handler=handler,
                description=description,
                emoji=emoji,
            )
        logger.debug("Registered tool: %s", name)

    def deregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        with self._lock:
            self._tools.pop(name, None)

    def get_definitions(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool definitions.

        Args:
            names: If given, only include these tool names.  Otherwise include all.

        Returns:
            List of {"type": "function", "function": schema} dicts.
        """
        with self._lock:
            entries = list(self._tools.values())
        result = []
        for entry in entries:
            if names is not None and entry.name not in names:
                continue
            schema = {**entry.schema, "name": entry.name}
            result.append({"type": "function", "function": schema})
        return result

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool handler by name.

        All exceptions are caught and returned as ``{"error": "..."}``.
        """
        with self._lock:
            entry = self._tools.get(name)
        if entry is None:
            return tool_error(f"Unknown tool: {name}")
        try:
            return entry.handler(args)
        except Exception as exc:
            logger.exception("Tool %s dispatch error: %s", name, exc)
            return tool_error(f"Tool execution failed: {type(exc).__name__}: {exc}")

    def get_all_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        with self._lock:
            return sorted(self._tools.keys())

    def get_schema(self, name: str) -> Optional[dict]:
        """Return raw schema dict for a tool, or None."""
        with self._lock:
            entry = self._tools.get(name)
        return entry.schema if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Return emoji for a tool, or *default*."""
        with self._lock:
            entry = self._tools.get(name)
        return entry.emoji if entry and entry.emoji else default


# Module-level singleton
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Helpers for tool response serialization
# ---------------------------------------------------------------------------


def tool_error(message: str, **extra) -> str:
    """Return a JSON error string for tool handlers."""
    result: Dict[str, Any] = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """Return a JSON result string for tool handlers.

    Accepts a dict positional arg *or* keyword arguments (not both).
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)
