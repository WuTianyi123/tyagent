"""Tests for FeishuAdapter edit_message method.

Run with: python3 -m pytest tests/test_feishu_adapter.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tyagent.platforms.base import SendResult
from tyagent.platforms.feishu import FeishuAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Create a mock config with valid app credentials."""
    config = MagicMock()
    config.extra = {
        "app_id": "cli_xxx",
        "app_secret": "secret_xxx",
        "domain": "feishu",
    }
    return config


@pytest.fixture
def adapter(mock_config):
    """Create a FeishuAdapter instance with mocked dependencies."""
    with patch("tyagent.platforms.feishu.FEISHU_AVAILABLE", True):
        with patch("tyagent.platforms.feishu.Path.home") as mock_home:
            mock_home.return_value = Path("/tmp")
            a = FeishuAdapter(mock_config)
            a._client = MagicMock()
            return a


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
# Tests: edit_message calls PATCH API correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_message_calls_patch_api_with_text_msg_type(adapter):
    """Edit with text message should call the PATCH API."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_patch = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.patch = mock_patch

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="Hello world",
    )

    assert result.success is True
    assert result.message_id == "om_xxx"
    mock_patch.assert_called_once()
    # Verify the request was built correctly (check args of the call)
    args, kwargs = mock_patch.call_args
    req = args[0]
    assert req.message_id == "om_xxx"
    assert req.request_body.msg_type == "text"
    assert "Hello world" in req.request_body.content


@pytest.mark.asyncio
async def test_edit_message_calls_patch_api_with_specified_msg_type(adapter):
    """Edit with explicit msg_type should use that type."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_patch = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.patch = mock_patch

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="Updated content",
        msg_type="post",
    )

    assert result.success is True
    args, kwargs = mock_patch.call_args
    req = args[0]
    assert req.request_body.msg_type == "post"
    assert "Updated content" in req.request_body.content


@pytest.mark.asyncio
async def test_edit_message_returns_message_id_on_success(adapter):
    """Successful edit should return the message_id."""
    mock_resp = MagicMock()
    mock_resp.code = 0
    mock_resp.msg = "success"

    mock_patch = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.patch = mock_patch

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

    mock_patch = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.patch = mock_patch

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

    mock_patch = MagicMock(return_value=mock_resp)
    adapter._client.im.v1.message.patch = mock_patch

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="**Bold text** and *italic*",
    )

    assert result.success is True
    args, kwargs = mock_patch.call_args
    req = args[0]
    assert req.request_body.msg_type == "post"


@pytest.mark.asyncio
async def test_edit_message_handles_exception(adapter):
    """Runtime exception during edit should be caught and returned as error."""
    mock_patch = MagicMock(
        side_effect=RuntimeError("Connection failed")
    )
    adapter._client.im.v1.message.patch = mock_patch

    result = await adapter.edit_message(
        chat_id="oc_xxx",
        message_id="om_xxx",
        text="This will crash",
    )

    assert result.success is False
    assert "Connection failed" in (result.error or "")
    assert result.retryable is True


if __name__ == "__main__":
    import traceback

    tests = [
        test_edit_message_returns_failure_when_client_not_initialized,
        test_edit_message_calls_patch_api_with_text_msg_type,
        test_edit_message_calls_patch_api_with_specified_msg_type,
        test_edit_message_returns_message_id_on_success,
        test_edit_message_returns_error_on_api_failure,
        test_edit_message_with_markdown_content,
        test_edit_message_handles_exception,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            import asyncio
            asyncio.run(test(FeishuAdapter.__new__(FeishuAdapter)))
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
