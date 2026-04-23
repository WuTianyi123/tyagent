"""Unit tests for ty_agent.gateway — Gateway message routing and lifecycle."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ty_agent.agent import AgentError
from ty_agent.config import AgentConfig, TyAgentConfig
from ty_agent.gateway import Gateway
from ty_agent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from ty_agent.session import Session, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    text="hello",
    message_type=MessageType.TEXT,
    platform="feishu",
    sender_id="user1",
    chat_id="chat1",
    chat_type="private",
    message_id="msg1",
    is_cmd=False,
    cmd_name=None,
):
    event = MagicMock(spec=MessageEvent)
    event.text = text
    event.message_type = message_type
    event.platform = platform
    event.sender_id = sender_id
    event.chat_id = chat_id
    event.chat_type = chat_type
    event.message_id = message_id
    event.media_urls = None
    event.media_types = None
    event.reply_to = None
    event.reply_to_text = None
    event.raw_message = None
    event.is_command.return_value = is_cmd
    event.get_command.return_value = cmd_name
    return event


def _make_adapter(platform="feishu"):
    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.platform_name = platform

    async def fake_send(chat_id, text, **kwargs):
        return SendResult(success=True, message_id="sent1")

    adapter.send_message = AsyncMock(side_effect=fake_send)
    adapter.build_session_key = MagicMock(return_value="feishu:chat1")
    adapter.start = AsyncMock()
    adapter.stop = AsyncMock()
    return adapter


def _make_config(**overrides):
    defaults = {
        "agent": AgentConfig(model="test-model", api_key="key"),
        "reset_triggers": ["new", "reset"],
    }
    defaults.update(overrides)
    return TyAgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Gateway.__init__
# ---------------------------------------------------------------------------


class TestGatewayInit:
    def test_default_init(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        assert gw.config is config
        assert isinstance(gw.session_store, SessionStore)
        assert isinstance(gw.adapters, dict)

    def test_custom_agent_and_store(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, session_store=store, agent=agent)
        assert gw.agent is agent
        assert gw.session_store is store


# ---------------------------------------------------------------------------
# Gateway._on_message — routing
# ---------------------------------------------------------------------------


class TestOnMessage:
    @pytest.mark.asyncio
    async def test_normal_message_flow(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Hi there!")
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="hello")
        result = await gw._on_message(event)

        assert result == "Hi there!"
        adapter.send_message.assert_called_once()
        # Check session has messages
        session = gw.session_store.get("feishu:chat1")
        assert len(session.messages) >= 2  # user + assistant

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_none(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, agent=MagicMock())
        event = _make_event(platform="telegram")
        result = await gw._on_message(event)
        assert result is None

    @pytest.mark.asyncio
    async def test_reset_command_archives(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, agent=MagicMock())

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        # Add some history first
        session = gw.session_store.get("feishu:chat1")
        session.add_message("user", "old message")
        gw.session_store.save("feishu:chat1")

        event = _make_event(text="/new", is_cmd=True, cmd_name="new")
        result = await gw._on_message(event)

        assert result == "Session archived"
        # Old session should be removed from active store
        # get() creates a new one with empty messages
        session = gw.session_store.get("feishu:chat1")
        assert len(session.messages) == 0
        # Archived file should exist on disk
        archive_files = list((tmp_path / "sessions").glob("*__archived_*.json"))
        assert len(archive_files) == 1

    @pytest.mark.asyncio
    async def test_status_command(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="/status", is_cmd=True, cmd_name="status")
        result = await gw._on_message(event)

        assert result == "Status sent"
        adapter.send_message.assert_called_once()
        sent_text = adapter.send_message.call_args[0][1]
        assert "ty-agent Status" in sent_text

    @pytest.mark.asyncio
    async def test_agent_error_returns_fallback(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(side_effect=AgentError("API error"))
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event()
        result = await gw._on_message(event)

        assert "error" in result.lower() or "sorry" in result.lower()

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_fallback(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(side_effect=RuntimeError("boom"))
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event()
        result = await gw._on_message(event)

        assert "sorry" in result.lower() or "wrong" in result.lower()

    @pytest.mark.asyncio
    async def test_media_attached_to_message(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Got it")
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="look at this")
        event.media_urls = ["img_key_123"]
        event.media_types = ["image"]
        result = await gw._on_message(event)

        assert result == "Got it"
        session = gw.session_store.get("feishu:chat1")
        user_msg = session.messages[0]["content"]
        assert "Attached image" in user_msg or "img_key_123" in user_msg

    @pytest.mark.asyncio
    async def test_send_failure_logged(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="reply")
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        adapter.send_message = AsyncMock(
            return_value=SendResult(success=False, error="timeout")
        )
        gw.adapters["feishu"] = adapter

        event = _make_event()
        await gw._on_message(event)
        # Should not raise, just log the error


# ---------------------------------------------------------------------------
# Gateway._format_status
# ---------------------------------------------------------------------------


class TestFormatStatus:
    def test_status_includes_key_info(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.model = "my-model"
        gw = Gateway(config, agent=agent)
        gw.adapters["feishu"] = _make_adapter()

        status = gw._format_status("test_key")
        assert "test_key" in status
        assert "my-model" in status
        assert "feishu" in status


# ---------------------------------------------------------------------------
# Gateway.stop
# ---------------------------------------------------------------------------


class TestGatewayStop:
    def test_stop_sets_flag(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, agent=MagicMock())
        assert gw._running is False
        gw._running = True
        gw.stop()
        assert gw._running is False
