"""Tests for web search and extraction tools."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from ty_agent.tools.registry import registry
from ty_agent.tools.web_tools import (
    _dispatch_extract,
    _dispatch_search,
    _get_api_key,
    _get_backend,
    _handle_web_extract,
    _handle_web_search,
)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestWebConfig:
    def test_default_backend(self):
        """Default backend is firecrawl when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_backend() == "firecrawl"

    def test_backend_from_env_firecrawl(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc_key"}, clear=True):
            assert _get_backend() == "firecrawl"

    def test_backend_from_env_tavily(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tv_key"}, clear=True):
            with patch("ty_agent.tools.web_tools._load_web_config", return_value={}):
                assert _get_backend() == "tavily"

    def test_backend_from_env_exa(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "ex_key"}, clear=True):
            with patch("ty_agent.tools.web_tools._load_web_config", return_value={}):
                assert _get_backend() == "exa"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "secret123"}, clear=True):
            assert _get_api_key("firecrawl") == "secret123"

    def test_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _get_api_key("firecrawl") is None


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_web_search_registered(self):
        names = registry.get_all_names()
        assert "web_search" in names

    def test_web_extract_registered(self):
        names = registry.get_all_names()
        assert "web_extract" in names

    def test_web_search_schema(self):
        schema = registry.get_schema("web_search")
        assert schema is not None
        assert schema["name"] == "web_search"
        params = schema["parameters"]["properties"]
        assert "query" in params
        assert "limit" in params

    def test_web_extract_schema(self):
        schema = registry.get_schema("web_extract")
        assert schema is not None
        assert schema["name"] == "web_extract"
        params = schema["parameters"]["properties"]
        assert "urls" in params


# ---------------------------------------------------------------------------
# Handler tests (with mocked HTTP)
# ---------------------------------------------------------------------------


class TestWebSearchHandler:
    @patch("ty_agent.tools.web_tools._http_post")
    def test_firecrawl_search_success(self, mock_post):
        mock_post.return_value = {
            "data": [
                {"title": "Python Guide", "url": "https://example.com/py", "description": "A guide"},
                {"title": "Async Python", "url": "https://example.com/async", "markdown": "# Async"},
            ]
        }
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key"}, clear=True):
            result = _handle_web_search({"query": "python async", "limit": 5})

        data = json.loads(result)
        assert data["success"] is True
        assert data["backend"] == "firecrawl"
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "Python Guide"
        assert data["results"][1]["description"].startswith("# Async")

    @patch("ty_agent.tools.web_tools._http_post")
    def test_tavily_search_success(self, mock_post):
        mock_post.return_value = {
            "results": [
                {"title": "Tavily Result", "url": "https://tavily.com", "content": "Some content here"},
            ]
        }
        with patch.dict(os.environ, {"TAVILY_API_KEY": "key"}, clear=True):
            with patch("ty_agent.tools.web_tools._load_web_config", return_value={}):
                result = _handle_web_search({"query": "test", "limit": 3})

        data = json.loads(result)
        assert data["success"] is True
        assert data["backend"] == "tavily"
        assert data["count"] == 1
        assert data["results"][0]["title"] == "Tavily Result"

    def test_search_missing_query(self):
        result = _handle_web_search({"query": ""})
        data = json.loads(result)
        assert "error" in data


class TestWebExtractHandler:
    @patch("ty_agent.tools.web_tools._http_post")
    def test_firecrawl_extract_success(self, mock_post):
        mock_post.return_value = {
            "data": {
                "markdown": "# Hello World",
                "metadata": {"title": "Hello"},
            }
        }
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key"}, clear=True):
            result = _handle_web_extract({"urls": ["https://example.com"]})

        data = json.loads(result)
        assert data["success"] is True
        assert data["count"] == 1
        assert data["documents"][0]["content"] == "# Hello World"
        assert data["documents"][0]["title"] == "Hello"

    @patch("ty_agent.tools.web_tools._http_post")
    def test_tavily_extract_success(self, mock_post):
        mock_post.return_value = {
            "results": [
                {"url": "https://example.com", "title": "Ex", "raw_content": "Raw text"},
            ],
            "failed_results": [],
        }
        with patch.dict(os.environ, {"TAVILY_API_KEY": "key"}, clear=True):
            with patch("ty_agent.tools.web_tools._load_web_config", return_value={}):
                result = _handle_web_extract({"urls": ["https://example.com"]})

        data = json.loads(result)
        assert data["success"] is True
        assert data["documents"][0]["content"] == "Raw text"

    def test_extract_no_urls(self):
        result = _handle_web_extract({"urls": []})
        data = json.loads(result)
        assert "error" in data

    def test_extract_invalid_urls(self):
        result = _handle_web_extract({"urls": ["not-a-url", "ftp://bad"]})
        data = json.loads(result)
        assert "error" in data

    @patch("ty_agent.tools.web_tools._http_post")
    def test_extract_partial_failure(self, mock_post):
        """One URL succeeds, one fails."""
        # First call succeeds, second raises
        def side_effect(url, headers, payload):
            if payload.get("url") == "https://ok.com":
                return {"data": {"markdown": "OK", "metadata": {}}}
            raise ValueError("Connection error")

        mock_post.side_effect = side_effect
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key"}, clear=True):
            result = _handle_web_extract({"urls": ["https://ok.com", "https://fail.com"]})

        data = json.loads(result)
        assert data["success"] is True
        assert len(data["documents"]) == 2
        # One success, one error
        contents = [d.get("content", "") for d in data["documents"]]
        errors = [d.get("error", "") for d in data["documents"]]
        assert "OK" in contents or any("Connection error" in e for e in errors)


# ---------------------------------------------------------------------------
# Registry dispatch tests
# ---------------------------------------------------------------------------


class TestRegistryDispatch:
    @patch("ty_agent.tools.web_tools._http_post")
    def test_dispatch_web_search_via_registry(self, mock_post):
        mock_post.return_value = {
            "data": [{"title": "T", "url": "https://t.com", "description": "D"}]
        }
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key"}, clear=True):
            result = registry.dispatch("web_search", {"query": "test", "limit": 1})

        data = json.loads(result)
        assert data["success"] is True
        assert data["count"] == 1

    @patch("ty_agent.tools.web_tools._http_post")
    def test_dispatch_web_extract_via_registry(self, mock_post):
        mock_post.return_value = {"data": {"markdown": "M", "metadata": {}}}
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key"}, clear=True):
            result = registry.dispatch("web_extract", {"urls": ["https://example.com"]})

        data = json.loads(result)
        assert data["success"] is True
        assert data["documents"][0]["content"] == "M"


# ---------------------------------------------------------------------------
# Exa backend tests
# ---------------------------------------------------------------------------


class TestExaBackend:
    @patch("ty_agent.tools.web_tools._http_post")
    def test_exa_search(self, mock_post):
        mock_post.return_value = {
            "results": [
                {
                    "title": "Exa Article",
                    "url": "https://exa.com",
                    "contents": {"text": "This is the article text."},
                }
            ]
        }
        with patch.dict(os.environ, {"EXA_API_KEY": "key"}, clear=True):
            with patch("ty_agent.tools.web_tools._load_web_config", return_value={}):
                result = _dispatch_search("query", 3)

        assert len(result) == 1
        assert result[0]["title"] == "Exa Article"
        assert "article text" in result[0]["description"]

    @patch("ty_agent.tools.web_tools._http_post")
    def test_exa_extract(self, mock_post):
        mock_post.return_value = {
            "results": [
                {"url": "https://exa.com", "title": "Exa", "text": "Full content"}
            ]
        }
        with patch.dict(os.environ, {"EXA_API_KEY": "key"}, clear=True):
            with patch("ty_agent.tools.web_tools._load_web_config", return_value={}):
                result = _dispatch_extract(["https://exa.com"])

        assert len(result) == 1
        assert result[0]["content"] == "Full content"
