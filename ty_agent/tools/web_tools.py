"""Web search and content extraction tools for ty-agent.

Supports multiple backends via HTTP APIs (no SDK dependencies):
- Firecrawl: https://firecrawl.dev (search, scrape)
- Tavily: https://tavily.com (search, extract)
- Exa: https://exa.ai (search, contents)

Backend selection priority:
1. Config ``web.backend`` (set in ~/.ty_agent/config.yaml)
2. Environment variable presence (FIRECRAWL_API_KEY → TAVILY_API_KEY → EXA_API_KEY)
3. Default: firecrawl

Usage:
    # These are auto-registered; LLM calls them via function calling.
    web_search(query="Python async patterns", limit=5)
    web_extract(urls=["https://example.com"])
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from ty_agent.tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 60.0
_MAX_SEARCH_RESULTS = 10
_MAX_EXTRACT_URLS = 5


def _has_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip())


def _load_web_config() -> Dict[str, Any]:
    """Load the ``web:`` section from ~/.ty_agent/config.yaml."""
    try:
        from ty_agent.config import load_config

        return load_config().web.to_dict()
    except Exception:
        return {}


def _get_backend() -> str:
    """Determine which web backend to use."""
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured in ("firecrawl", "tavily", "exa"):
        return configured

    # Fallback: pick by env var presence
    if _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL"):
        return "firecrawl"
    if _has_env("TAVILY_API_KEY"):
        return "tavily"
    if _has_env("EXA_API_KEY"):
        return "exa"

    return "firecrawl"


def _get_api_key(backend: str) -> Optional[str]:
    """Get API key for the given backend."""
    config = _load_web_config()
    # Config takes priority
    if config.get("api_key"):
        return config["api_key"]

    env_map = {
        "firecrawl": "FIRECRAWL_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "exa": "EXA_API_KEY",
    }
    env_name = env_map.get(backend)
    if env_name:
        return os.getenv(env_name)
    return None


def _get_api_url(backend: str) -> Optional[str]:
    """Get custom API URL for self-hosted backends."""
    config = _load_web_config()
    if config.get("api_url"):
        return config["api_url"]

    if backend == "firecrawl":
        return os.getenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev").rstrip("/")
    if backend == "tavily":
        return "https://api.tavily.com"
    if backend == "exa":
        return "https://api.exa.ai"
    return None


def _raise_not_configured(backend: str) -> None:
    """Raise a clear configuration error."""
    env_map = {
        "firecrawl": "FIRECRAWL_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "exa": "EXA_API_KEY",
    }
    env_name = env_map.get(backend, f"{backend.upper()}_API_KEY")
    raise ValueError(
        f"Web backend '{backend}' is not configured. "
        f"Set {env_name} environment variable, or configure 'web.api_key' in ~/.ty_agent/config.yaml."
    )


# ---------------------------------------------------------------------------
# HTTP client helper
# ---------------------------------------------------------------------------


def _http_post(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make a synchronous POST request and return JSON response."""
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        raise ValueError(f"HTTP {exc.response.status_code}: {body}") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"Request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Firecrawl backend
# ---------------------------------------------------------------------------


def _firecrawl_search(query: str, limit: int) -> List[Dict[str, Any]]:
    api_key = _get_api_key("firecrawl")
    api_url = _get_api_url("firecrawl")
    if not api_key:
        _raise_not_configured("firecrawl")

    url = f"{api_url}/v1/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"query": query, "limit": min(limit, _MAX_SEARCH_RESULTS)}

    data = _http_post(url, headers, payload)
    results = data.get("data", []) if isinstance(data.get("data"), list) else []

    # Normalize to standard format
    normalized = []
    for i, item in enumerate(results):
        normalized.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", item.get("markdown", "")[:500]),
            "position": i + 1,
        })
    return normalized


def _firecrawl_extract(urls: List[str]) -> List[Dict[str, Any]]:
    api_key = _get_api_key("firecrawl")
    api_url = _get_api_url("firecrawl")
    if not api_key:
        _raise_not_configured("firecrawl")

    url = f"{api_url}/v1/scrape"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    documents = []
    for target_url in urls[:_MAX_EXTRACT_URLS]:
        payload = {"url": target_url, "formats": ["markdown"]}
        try:
            data = _http_post(url, headers, payload)
            item = data.get("data", {}) if isinstance(data, dict) else {}
            documents.append({
                "url": target_url,
                "title": item.get("metadata", {}).get("title", ""),
                "content": item.get("markdown", item.get("content", "")),
                "metadata": item.get("metadata", {}),
            })
        except Exception as exc:
            documents.append({
                "url": target_url,
                "title": "",
                "content": "",
                "error": str(exc),
            })
    return documents


# ---------------------------------------------------------------------------
# Tavily backend
# ---------------------------------------------------------------------------


def _tavily_search(query: str, limit: int) -> List[Dict[str, Any]]:
    api_key = _get_api_key("tavily")
    if not api_key:
        _raise_not_configured("tavily")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": min(limit, _MAX_SEARCH_RESULTS),
    }

    data = _http_post(url, {}, payload)
    results = data.get("results", [])

    normalized = []
    for i, item in enumerate(results):
        normalized.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("content", "")[:1000],
            "position": i + 1,
        })
    return normalized


def _tavily_extract(urls: List[str]) -> List[Dict[str, Any]]:
    api_key = _get_api_key("tavily")
    if not api_key:
        _raise_not_configured("tavily")

    url = "https://api.tavily.com/extract"
    payload = {
        "api_key": api_key,
        "urls": urls[:_MAX_EXTRACT_URLS],
        "include_images": False,
    }

    data = _http_post(url, {}, payload)
    results = data.get("results", [])
    failed = data.get("failed_results", []) + data.get("failed_urls", [])

    documents = []
    for item in results:
        documents.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "content": item.get("raw_content", item.get("content", "")),
            "metadata": {"sourceURL": item.get("url", "")},
        })
    for fail in failed:
        if isinstance(fail, dict):
            documents.append({
                "url": fail.get("url", ""),
                "title": "",
                "content": "",
                "error": fail.get("error", "extraction failed"),
            })
        elif isinstance(fail, str):
            documents.append({"url": fail, "title": "", "content": "", "error": "extraction failed"})
    return documents


# ---------------------------------------------------------------------------
# Exa backend
# ---------------------------------------------------------------------------


def _exa_search(query: str, limit: int) -> List[Dict[str, Any]]:
    api_key = _get_api_key("exa")
    if not api_key:
        _raise_not_configured("exa")

    url = "https://api.exa.ai/search"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "query": query,
        "numResults": min(limit, _MAX_SEARCH_RESULTS),
        "contents": {"text": True},
    }

    data = _http_post(url, headers, payload)
    results = data.get("results", [])

    normalized = []
    for i, item in enumerate(results):
        text = ""
        contents = item.get("contents", {})
        if isinstance(contents, dict):
            text = contents.get("text", "")[:500]
        normalized.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": text,
            "position": i + 1,
        })
    return normalized


def _exa_extract(urls: List[str]) -> List[Dict[str, Any]]:
    api_key = _get_api_key("exa")
    if not api_key:
        _raise_not_configured("exa")

    url = "https://api.exa.ai/contents"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {"ids": urls[:_MAX_EXTRACT_URLS], "text": True}

    data = _http_post(url, headers, payload)
    results = data.get("results", [])

    documents = []
    for item in results:
        documents.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "content": item.get("text", ""),
            "metadata": {"sourceURL": item.get("url", "")},
        })
    return documents


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------


def _dispatch_search(query: str, limit: int) -> List[Dict[str, Any]]:
    backend = _get_backend()
    if backend == "firecrawl":
        return _firecrawl_search(query, limit)
    if backend == "tavily":
        return _tavily_search(query, limit)
    if backend == "exa":
        return _exa_search(query, limit)
    raise ValueError(f"Unknown web backend: {backend}")


def _dispatch_extract(urls: List[str]) -> List[Dict[str, Any]]:
    backend = _get_backend()
    if backend == "firecrawl":
        return _firecrawl_extract(urls)
    if backend == "tavily":
        return _tavily_extract(urls)
    if backend == "exa":
        return _exa_extract(urls)
    raise ValueError(f"Unknown web backend: {backend}")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_web_search(args: Dict[str, Any]) -> str:
    """Search the web for information."""
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required")

    limit = max(1, min(int(args.get("limit", 5)), _MAX_SEARCH_RESULTS))

    try:
        results = _dispatch_search(query, limit)
        return tool_result(
            success=True,
            query=query,
            backend=_get_backend(),
            results=results,
            count=len(results),
        )
    except Exception as exc:
        logger.exception("web_search failed: %s", exc)
        return tool_error(f"Search failed: {exc}")


def _handle_web_extract(args: Dict[str, Any]) -> str:
    """Extract content from specific web pages."""
    urls = args.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        return tool_error("urls is required (list of URLs)")

    # Validate URLs
    valid_urls = []
    for u in urls[:_MAX_EXTRACT_URLS]:
        u = str(u).strip()
        if u.startswith(("http://", "https://")):
            valid_urls.append(u)

    if not valid_urls:
        return tool_error("No valid HTTP/HTTPS URLs provided")

    try:
        documents = _dispatch_extract(valid_urls)
        return tool_result(
            success=True,
            backend=_get_backend(),
            documents=documents,
            count=len(documents),
        )
    except Exception as exc:
        logger.exception("web_extract failed: %s", exc)
        return tool_error(f"Extraction failed: {exc}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for information. "
        "Returns a list of results with title, URL, and description. "
        "Use web_extract to get full content from specific URLs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of results (default: 5, max: {_MAX_SEARCH_RESULTS})",
                "default": 5,
                "minimum": 1,
                "maximum": _MAX_SEARCH_RESULTS,
            },
        },
        "required": ["query"],
    },
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": (
        "Extract content from specific web pages. "
        f"Accepts up to {_MAX_EXTRACT_URLS} URLs per call. "
        "Returns markdown or text content from each page."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"List of URLs to extract content from (max {_MAX_EXTRACT_URLS} URLs per call)",
                "maxItems": _MAX_EXTRACT_URLS,
            },
        },
        "required": ["urls"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="web_search",
    schema=WEB_SEARCH_SCHEMA,
    handler=_handle_web_search,
    description="Search the web for information",
    emoji="🔍",
)
registry.register(
    name="web_extract",
    schema=WEB_EXTRACT_SCHEMA,
    handler=_handle_web_extract,
    description="Extract content from web pages",
    emoji="📄",
)
