"""Tests for browser automation tools."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tyagent.tools.registry import registry
from tyagent.tools.browser_tools import (
    _find_agent_browser,
    _is_browser_available,
    _parse_snapshot_text,
    _run_cmd,
)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestBrowserRegistration:
    def test_all_browser_tools_registered(self):
        names = registry.get_all_names()
        expected = [
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_back",
            "browser_press",
            "browser_get_images",
            "browser_vision",
            "browser_console",
        ]
        for name in expected:
            assert name in names, f"{name} not registered"

    def test_navigate_schema(self):
        schema = registry.get_schema("browser_navigate")
        assert schema is not None
        assert schema["name"] == "browser_navigate"
        params = schema["parameters"]["properties"]
        assert "url" in params

    def test_snapshot_schema(self):
        schema = registry.get_schema("browser_snapshot")
        assert schema is not None
        params = schema["parameters"]["properties"]
        assert "full" in params

    def test_click_schema(self):
        schema = registry.get_schema("browser_click")
        assert schema is not None
        params = schema["parameters"]["properties"]
        assert "ref" in params


# ---------------------------------------------------------------------------
# CLI discovery tests
# ---------------------------------------------------------------------------


class TestCLIDiscovery:
    def test_find_agent_browser_env_override(self):
        with patch.dict(os.environ, {"AGENT_BROWSER_CMD": "/fake/agent-browser"}, clear=False):
            # Reset cache
            from tyagent.tools import browser_tools
            browser_tools._cached_browser_cmd = None
            result = _find_agent_browser(silent=True)
            assert result == "/fake/agent-browser"

    def test_is_browser_available_false(self):
        with patch.dict(os.environ, {}, clear=True):
            from tyagent.tools import browser_tools
            browser_tools._cached_browser_cmd = None
            with patch(
                "tyagent.tools.browser_tools._find_agent_browser", return_value=None
            ):
                assert _is_browser_available() is False


# ---------------------------------------------------------------------------
# Snapshot parser tests
# ---------------------------------------------------------------------------


class TestSnapshotParser:
    def test_parse_refs(self):
        text = """- document:
  - heading "Example Domain" [ref=e1] [level=1]
  - paragraph:
    - link "Learn more" [ref=e2]:
      - /url: https://iana.org/domains/example
"""
        result = _parse_snapshot_text(text)
        assert result["ref_count"] == 2
        assert "e1" in result["refs"]
        assert "e2" in result["refs"]
        assert result["refs"]["e1"]["level"] == "1"

    def test_parse_no_refs(self):
        result = _parse_snapshot_text("No refs here")
        assert result["ref_count"] == 0
        assert result["snapshot"] == "No refs here"


# ---------------------------------------------------------------------------
# Command runner tests (mocked)
# ---------------------------------------------------------------------------


class TestCommandRunner:
    @patch("tyagent.tools.browser_tools._find_agent_browser")
    @patch("tyagent.tools.browser_tools.subprocess.run")
    def test_run_cmd_success(self, mock_run, mock_find):
        mock_find.return_value = "agent-browser"
        mock_run.return_value = MagicMock(
            stdout="Success output",
            stderr="",
            returncode=0,
        )
        result = _run_cmd("test_session", "open", ["https://example.com"])
        assert result["success"] is True
        assert result["stdout"] == "Success output"

    @patch("tyagent.tools.browser_tools._find_agent_browser")
    @patch("tyagent.tools.browser_tools.subprocess.run")
    def test_run_cmd_json_output(self, mock_run, mock_find):
        mock_find.return_value = "agent-browser"
        mock_run.return_value = MagicMock(
            stdout='{"success":true,"data":{"title":"Test"}}',
            stderr="",
            returncode=0,
        )
        result = _run_cmd("test_session", "snapshot", ["--json"])
        assert result["success"] is True
        assert result["data"]["title"] == "Test"

    @patch("tyagent.tools.browser_tools._find_agent_browser")
    @patch("tyagent.tools.browser_tools.subprocess.run")
    def test_run_cmd_failure(self, mock_run, mock_find):
        mock_find.return_value = "agent-browser"
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="Page not found",
            returncode=1,
        )
        result = _run_cmd("test_session", "open", ["https://bad.url"])
        assert result["success"] is False
        assert "Page not found" in result["error"]

    @patch("tyagent.tools.browser_tools._find_agent_browser")
    def test_run_cmd_browser_not_found(self, mock_find):
        mock_find.return_value = None
        result = _run_cmd("test_session", "open", ["https://example.com"])
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# Handler tests (mocked)
# ---------------------------------------------------------------------------


class TestBrowserHandlers:
    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_navigate(self, mock_run):
        mock_run.side_effect = [
            {"success": True, "stdout": "\u2713 Example Domain\n  https://example.com/", "stderr": ""},
            {"success": True, "stdout": '- heading "Test" [ref=e1]', "stderr": ""},
        ]
        from tyagent.tools.browser_tools import _handle_browser_navigate
        result = _handle_browser_navigate({"url": "https://example.com"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["url"].startswith("https://example.com")
        assert "snapshot" in data

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_navigate_missing_url(self, mock_run):
        from tyagent.tools.browser_tools import _handle_browser_navigate
        result = _handle_browser_navigate({"url": ""})
        data = json.loads(result)
        assert "error" in data

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_click(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "\u2713 Done", "stderr": ""}
        from tyagent.tools.browser_tools import _handle_browser_click
        result = _handle_browser_click({"ref": "e5"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["ref"] == "@e5"

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_click_missing_ref(self, mock_run):
        from tyagent.tools.browser_tools import _handle_browser_click
        result = _handle_browser_click({"ref": ""})
        data = json.loads(result)
        assert "error" in data

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_type(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "\u2713 Done", "stderr": ""}
        from tyagent.tools.browser_tools import _handle_browser_type
        result = _handle_browser_type({"ref": "e3", "text": "hello"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["ref"] == "@e3"

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_scroll(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "\u2713 Done", "stderr": ""}
        from tyagent.tools.browser_tools import _handle_browser_scroll
        result = _handle_browser_scroll({"direction": "down"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["direction"] == "down"

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_press(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "\u2713 Done", "stderr": ""}
        from tyagent.tools.browser_tools import _handle_browser_press
        result = _handle_browser_press({"key": "Enter"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["key"] == "Enter"

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_console_eval(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": '"Example Domain"', "stderr": ""}
        from tyagent.tools.browser_tools import _handle_browser_console
        result = _handle_browser_console({"expression": "document.title"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["result"] == '"Example Domain"'

    @patch("tyagent.tools.browser_tools._run_cmd")
    def test_get_images(self, mock_run):
        mock_run.return_value = {
            "success": True,
            "stdout": '[{"src":"https://example.com/img.png","alt":"test","width":100,"height":50}]',
            "stderr": "",
        }
        from tyagent.tools.browser_tools import _handle_browser_get_images
        result = _handle_browser_get_images({})
        data = json.loads(result)
        assert data["success"] is True
        assert data["count"] == 1
        assert data["images"][0]["src"] == "https://example.com/img.png"


# ---------------------------------------------------------------------------
# Integration tests (require agent-browser installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_browser_available(), reason="agent-browser not installed")
class TestBrowserIntegration:
    def test_navigate_and_snapshot(self):
        """End-to-end: navigate to example.com and get snapshot."""
        result = registry.dispatch("browser_navigate", {"url": "https://example.com"})
        data = json.loads(result)
        assert data["success"] is True
        assert "snapshot" in data
        assert "refs" in data
        assert data.get("ref_count", 0) >= 1

    def test_click_and_back(self):
        """Click a link, then go back."""
        # Navigate first
        registry.dispatch("browser_navigate", {"url": "https://example.com"})
        # Click the "Learn more" link
        result = registry.dispatch("browser_click", {"ref": "e2"})
        data = json.loads(result)
        assert data["success"] is True
        # Go back
        result2 = registry.dispatch("browser_back", {})
        data2 = json.loads(result2)
        assert data2["success"] is True

    def test_eval(self):
        """Execute JavaScript in the page."""
        registry.dispatch("browser_navigate", {"url": "https://example.com"})
        result = registry.dispatch("browser_console", {"expression": "document.title"})
        data = json.loads(result)
        assert data["success"] is True
        assert "Example" in data["result"]
