"""Session Search Tool - Long-Term Conversation Recall

Searches past conversation messages stored in SQLite via FTS5 with jieba
segmentation for both Chinese and English support.

Flow:
  1. Query is jieba-segmented into terms
  2. Terms are escaped and joined as FTS5 AND query
  3. FTS5 BM25-ranked results are returned with session_key and content snippet

Uses a module-level Database singleton set by the gateway at startup.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from tyagent.tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Database singleton (set by gateway at startup)
# ---------------------------------------------------------------------------

_db: Any = None  # Database instance, typed as Any to avoid circular import


def set_search_db(db: Any) -> None:
    """Set the Database singleton for search. Called once at gateway startup."""
    global _db
    _db = db


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def _handle_session_search(args: Dict[str, Any]) -> str:
    """Search past conversations using FTS5 full-text search."""
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required for session_search")

    raw_limit = args.get("limit", 5)
    try:
        limit = max(1, min(int(raw_limit), 20))
    except (ValueError, TypeError):
        limit = 5

    if _db is None:
        return tool_error(
            "Search database is not available. "
            "The session_search tool requires a running gateway with a database."
        )

    try:
        results = _db.search_messages(query, limit=limit)
    except Exception as exc:
        logger.exception("Session search failed")
        return tool_error(f"Search failed: {type(exc).__name__}: {exc}")

    return tool_result(success=True, results=results, count=len(results))


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past conversations using full-text search. "
        "Supports both Chinese and English queries — terms are automatically "
        "segmented for Chinese text. Returns results ranked by relevance "
        "with session identifiers and content snippets.\n\n"
        "Use this when you need to recall something the user said in a "
        "previous conversation, check if a topic was discussed before, "
        "or reference past decisions or discussions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — can be Chinese, English, or mixed.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 5, max: 20).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=_handle_session_search,
    description="Search past conversations using full-text search",
    emoji="🔍",
)
