"""Tests for FeishuAdapter — edit_message method and credential extraction.

Run with: python3 -m pytest tests/test_feishu_adapter.py -v

Tests use real PlatformConfig objects (not MagicMock) so format migration
issues are caught at test time. Only external dependencies are mocked:
FEISHU_AVAILABLE (import guard) and _client (Lark SDK).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tyagent.config import PlatformConfig
from tyagent.platforms.base import SendResult
from tyagent.platforms.feishu import FeishuAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def platform_config():
    """Real PlatformConfig with valid new-format credentials."""
    return PlatformConfig(
        enabled=True,
        extra={
            "connection": {
                "app_id": "cli_test_app",
                "app_secret": "test_app_secret",
            },
        },
    )


@pytest.fixture
def adapter(platform_config, tmp_path):
    """FeishuAdapter with real config, mocked external deps only."""
    with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
        a = FeishuAdapter(platform_config, home_dir=tmp_path)
        a._client = MagicMock()
        return a


# ---------------------------------------------------------------------------
# Tests: credential extraction from real config
# ---------------------------------------------------------------------------


class TestCredentialExtraction:
    """Verify FeishuAdapter correctly extracts credentials from config.

    These use real PlatformConfig objects — no MagicMock config.
    If the extraction path changes (e.g., extra.connection.app_id →
    extra.app_id), these tests will FAIL, not silently pass.
    """

    def test_extracts_app_id_from_connection_group(self, platform_config):
        with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
            adapter = FeishuAdapter(platform_config)
        assert adapter.app_id == "cli_test_app"

    def test_extracts_app_secret_from_connection_group(self, platform_config):
        with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
            adapter = FeishuAdapter(platform_config)
        assert adapter.app_secret == "test_app_secret"

    def test_extracts_domain_with_default(self, platform_config):
        with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
            adapter = FeishuAdapter(platform_config)
        assert adapter.domain == "feishu"

    def test_empty_app_id_raises_valueerror(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "connection": {
                    "app_id": "",
                    "app_secret": "secret",
                },
            },
        )
        with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
            with pytest.raises(ValueError, match="app_id"):
                FeishuAdapter(config)

    def test_missing_connection_group_raises_valueerror(self):
        """Config with no connection group should fail — no fallback to old
        format."""
        config = PlatformConfig(enabled=True, extra={"other": "data"})
        with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
            with pytest.raises(ValueError, match="app_id"):
                FeishuAdapter(config)

    def test_domain_from_config(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "connection": {
                    "app_id": "test",
                    "app_secret": "secret",
                    "domain": "lark",
                },
            },
        )
        with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
            adapter = FeishuAdapter(config)
        assert adapter.domain == "lark"


# ---------------------------------------------------------------------------
# Tests: edit_message with client not initialized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_message_returns_failure_when_client_not_initialized(adapter):
    """edit_message should return success=False when _client is None."""
    adapter._client = None
    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="edited text",
    )
    assert result.success is False
    assert "Client not initialized" in (result.error or "")


# ---------------------------------------------------------------------------
# Tests: edit_message calls UPDATE API correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_message_calls_update_api_with_text_msg_type(adapter):
    """Edit with text message should call the UPDATE API."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_update = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.update = mock_update

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="Hello world",
    )

    assert result.success is True
    assert result.message_id == "om_xxx"
    mock_update.assert_called_once()
    args, kwargs = mock_update.call_args
    req = args[0]
    assert req.message_id == "om_xxx"
    assert req.request_body.msg_type == "text"
    assert "Hello world" in req.request_body.content


@pytest.mark.asyncio
async def test_edit_message_calls_update_api_with_specified_msg_type(adapter):
    """Edit with explicit msg_type should use that type."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_update = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.update = mock_update

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="Updated content",
        msg_type="post",
    )

    assert result.success is True
    args, kwargs = mock_update.call_args
    req = args[0]
    assert req.request_body.msg_type == "post"
    assert "Updated content" in req.request_body.content


@pytest.mark.asyncio
async def test_edit_message_returns_message_id_on_success(adapter):
    """Successful edit should return the message_id."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_update = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.update = mock_update

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_edited_msg",
        text="Edited text",
    )

    assert result.success is True
    assert result.message_id == "om_edited_msg"


@pytest.mark.asyncio
async def test_edit_message_returns_error_on_api_failure(adapter):
    """API error should be returned in SendResult."""
    mock_resp = MagicMock()
    mock_resp.code = 100003
    mock_resp.msg = "invalid message_id"

    mock_update = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.update = mock_update

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_bad_id",
        text="This edit will fail",
    )

    assert result.success is False
    assert "100003" in (result.error or "")
    assert "invalid message_id" in (result.error or "")


@pytest.mark.asyncio
async def test_edit_message_with_markdown_content(adapter):
    """Edit with markdown content should detect post type."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_update = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.update = mock_update

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="**Bold text** and *italic*",
    )

    assert result.success is True
    args, kwargs = mock_update.call_args
    req = args[0]
    assert req.request_body.msg_type == "post"


@pytest.mark.asyncio
async def test_edit_message_handles_exception(adapter):
    """Runtime exception during edit should be caught and returned as error."""
    mock_update = MagicMock(
        side_effect=RuntimeError("Connection failed")
    )
    adapter._client.im.v1.message.update = mock_update

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="This will crash",
    )

    assert result.success is False
    assert "Connection failed" in (result.error or "")
    assert result.retryable is True
