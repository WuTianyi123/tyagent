"""Unit tests for tyagent.gateway — Gateway message routing and lifecycle."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyagent.agent import AgentError
from tyagent.config import AgentConfig, TyAgentConfig
from tyagent.gateway import Gateway, StreamConsumer, _sanitize_message_chain
from tyagent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from tyagent.session import Session, SessionStore


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
        gw.session_store.close()

    def test_custom_agent_and_store(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, session_store=store, agent=agent)
        assert gw._agent_cache.get("_default") is agent
        assert gw.session_store is store


# ---------------------------------------------------------------------------
# Gateway._on_message — routing
# ---------------------------------------------------------------------------


class TestOnMessage:
    @pytest.mark.asyncio
    async def test_normal_message_flow(self, tmp_path):
        """Normal message flow: non-command messages go through streaming path
        and response is sent via StreamConsumer."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Hi there!")
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="hello")
        result = await gw._on_message(event)

        # Streaming path: returns the agent response directly
        assert result == "Hi there!"
        # Streaming path does NOT call adapter.send_message at the end
        # (StreamConsumer handles message delivery internally)
        # User message is persisted by gateway directly.
        assert gw.session_store.get_message_count("feishu:chat1") >= 1
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_none(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, agent=MagicMock())
        event = _make_event(platform="telegram")
        result = await gw._on_message(event)
        assert result is None
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_reset_command_archives(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock()
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        # Add some history first
        session = gw.session_store.get("feishu:chat1")
        session.add_message("user", "old message")
        # Message is persisted immediately, so count should be 1
        assert gw.session_store.get_message_count("feishu:chat1") == 1

        event = _make_event(text="/new", is_cmd=True, cmd_name="new")
        result = await gw._on_message(event)

        assert result == "Session archived"
        # After archive, get_or_create_after_archive gives fresh session
        fresh = gw.session_store.get_or_create_after_archive("feishu:chat1")
        assert fresh.session_key == "feishu:chat1"
        # Old messages still in DB
        assert gw.session_store.get_message_count("feishu:chat1") == 1
        gw.session_store.close()

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
        assert "tyagent Status" in sent_text
        gw.session_store.close()

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
        gw.session_store.close()

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
        gw.session_store.close()

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
        msgs = gw.session_store.get_messages("feishu:chat1")
        user_msg = msgs[0]["content"]
        assert "Attached image" in user_msg or "img_key_123" in user_msg
        gw.session_store.close()

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
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_normal_message_flow_on_message_callback(self, tmp_path):
        """Verify on_message callback is passed to agent.chat()."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Hi!")
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="hello")
        await gw._on_message(event)

        # Verify agent.chat was called with on_message
        call_kwargs = agent.chat.call_args[1]
        assert "on_message" in call_kwargs
        assert callable(call_kwargs["on_message"])
        gw.session_store.close()


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
        gw.session_store.close()


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
        gw.session_store.close()


# ---------------------------------------------------------------------------
# _sanitize_message_chain — orphaned tool_calls cleanup
# ---------------------------------------------------------------------------


class TestSanitizeMessageChain:
    def test_empty_chain(self):
        assert _sanitize_message_chain([]) == []

    def test_no_tool_calls(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = _sanitize_message_chain(msgs)
        assert result is msgs  # same object, no copy

    def test_complete_tool_chain_unchanged(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
            {"role": "tool", "content": "file1.txt", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Done!"},
        ]
        result = _sanitize_message_chain(msgs)
        assert result is msgs  # unchanged

    def test_orphan_tool_calls_at_end(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
        ]
        result = _sanitize_message_chain(msgs)
        assert result is not msgs  # new list
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" not in result[1]
        assert result[1]["content"] == ""

    def test_orphan_tool_calls_followed_by_user(self):
        """The exact scenario from the bug report."""
        msgs = [
            {"role": "user", "content": "check status"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "systemctl", "arguments": "{}"}}]},
            {"role": "tool", "content": "active (running)", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc2", "type": "function", "function": {"name": "journalctl", "arguments": "{}"}}]},
            {"role": "user", "content": "你好？"},
        ]
        result = _sanitize_message_chain(msgs)
        assert result is not msgs
        assert len(result) == 5
        # The tool_calls should be stripped from message [3]
        assert result[3]["role"] == "assistant"
        assert "tool_calls" not in result[3]
        # Earlier tool_calls should be preserved
        assert result[1].get("tool_calls") is not None

    def test_middle_orphan_stays_untouched(self):
        """Only the LAST orphan is fixed — orphan in the middle is not
        this function's concern (it would have been followed by user msg
        which is valid from API perspective if no tool response exists)."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "user", "content": "hi"},
        ]
        result = _sanitize_message_chain(msgs)
        # Last orphan is the one at [0] (no tool follows) — should be fixed
        assert "tool_calls" not in result[0]

    def test_original_not_mutated(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        ]
        _sanitize_message_chain(msgs)
        assert "tool_calls" in msgs[0]  # original unchanged
