"""Unit tests for tyagent.gateway — Gateway message routing and lifecycle."""

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyagent.agent import AgentError
from tyagent.config import AgentConfig, TyAgentConfig
from tyagent.gateway import Gateway
from tyagent.gateway.commands import _format_status
from tyagent.gateway.gateway import _sanitize_message_chain
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
        "reset_triggers": ["new"],
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
        """Normal message flow: non-command messages go through actor model
        and response is sent via _consume_output background task."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Hi there!")
        agent.model = "test-model"
        agent.start = AsyncMock()
        agent.send_message = AsyncMock()
        agent._output_queue = asyncio.Queue()
        agent._tool_progress_callback = None
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="hello")
        result = await gw._on_message(event)

        # Actor model: returns None (response is sent via background consumer)
        assert result is None
        # agent.start was called with history
        agent.start.assert_called_once()
        # agent.send_message was called with the user text
        agent.send_message.assert_called_once()
        call_text = agent.send_message.call_args[0][0]
        assert "hello" in call_text
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
        # Old session preserved with its messages
        old = gw.session_store.get("feishu:chat1")
        assert old.session_key == "feishu:chat1"
        # Old messages still in DB (visible via get_message_count without filter)
        all_count = gw.session_store.get_message_count("feishu:chat1")
        assert all_count == 1
        # But session.messages (filtered by current_session_id) shows 0
        assert len(old.messages) == 0
        # A new session_id was generated
        assert "current_session_id" in old.metadata
        assert old.metadata["current_session_id"] != ""
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
        agent.start = AsyncMock()
        agent.send_message = AsyncMock(side_effect=AgentError("API error"))
        agent._tool_progress_callback = None
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event()
        result = await gw._on_message(event)

        # Error is sent via adapter.send_message, result is None
        assert result is None
        adapter.send_message.assert_called_once()
        sent_text = adapter.send_message.call_args[0][1]
        assert "error" in sent_text.lower()
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_fallback(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.start = AsyncMock()
        agent.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        agent._tool_progress_callback = None
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event()
        result = await gw._on_message(event)

        # Error is sent via adapter.send_message, result is None
        assert result is None
        adapter.send_message.assert_called_once()
        sent_text = adapter.send_message.call_args[0][1]
        assert "sorry" in sent_text.lower() or "wrong" in sent_text.lower()
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_media_attached_to_message(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Got it")
        agent.model = "test-model"
        agent.start = AsyncMock()
        agent.send_message = AsyncMock()
        agent._output_queue = asyncio.Queue()
        agent._tool_progress_callback = None
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="look at this")
        event.media_urls = ["img_key_123"]
        event.media_types = ["image"]
        result = await gw._on_message(event)

        assert result is None
        msgs = gw.session_store.get_messages("feishu:chat1")
        user_msg = msgs[0]["content"]
        assert "Attached image" in user_msg or "img_key_123" in user_msg
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_send_failure_logged(self, tmp_path):
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="reply")
        agent.start = AsyncMock()
        agent.send_message = AsyncMock()
        agent._output_queue = asyncio.Queue()
        agent._tool_progress_callback = None
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
        """Verify on_message callback is passed to agent.start()."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Hi!")
        agent.model = "test-model"
        agent.start = AsyncMock()
        agent.send_message = AsyncMock()
        agent._output_queue = asyncio.Queue()
        agent._tool_progress_callback = None
        gw = Gateway(config, agent=agent)

        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        event = _make_event(text="hello")
        await gw._on_message(event)

        # Verify agent.start was called with on_message
        call_kwargs = agent.start.call_args[1]
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

        status = _format_status(gw, "test_key")
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
        gw.supervisor.shutdown()
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
        assert result == msgs

    def test_complete_tool_chain_unchanged(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
            {"role": "tool", "content": "file1.txt", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Done!"},
        ]
        result = _sanitize_message_chain(msgs)
        assert result == msgs

    def test_orphan_tool_calls_at_end(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
        ]
        result = _sanitize_message_chain(msgs)
        assert result is not msgs  # new list
        # Assistant + synthetic tool response inserted = 3 messages
        assert len(result) == 3
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]  # preserved
        assert result[1]["content"] == ""
        # Synthetic tool response
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "tc1"
        assert "interrupted" in result[2]["content"]

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
        # 5 original + 1 synthetic tool response = 6
        assert len(result) == 6
        # The assistant with orphaned tool_calls at [3] is preserved
        assert result[3]["role"] == "assistant"
        assert "tool_calls" in result[3]
        # Synthetic tool response inserted after the orphaned assistant
        assert result[4]["role"] == "tool"
        assert result[4]["tool_call_id"] == "tc2"
        assert "interrupted" in result[4]["content"]
        # Earlier tool_calls should be preserved
        assert result[1].get("tool_calls") is not None

    def test_middle_orphan_stays_untouched(self):
        """Only the LAST orphan is fixed — orphan in the middle is not
        this function's concern (it would have been followed by user msg
        which is valid from API perspective if no tool response exists).
        Now we insert synthetic tool responses instead of stripping."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "user", "content": "hi"},
        ]
        result = _sanitize_message_chain(msgs)
        # Synthetic tool response inserted after the only orphan
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "t1"
        assert result[2]["role"] == "user"

    def test_original_not_mutated(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        ]
        _sanitize_message_chain(msgs)
        assert "tool_calls" in msgs[0]  # original unchanged


# ---------------------------------------------------------------------------
# Gateway graceful drain + SIGUSR1 + session recovery (Task 2)
# ---------------------------------------------------------------------------


class TestGatewayDrainAndRestart:
    def test_init_has_drain_attributes(self, tmp_path):
        """Gateway.__init__ sets drain/restart attributes."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        assert hasattr(gw, "_restart_requested")
        assert hasattr(gw, "_draining")
        assert hasattr(gw, "_active_sessions")
        assert isinstance(gw._active_sessions, set)
        assert hasattr(gw, "_restart_drain_timeout")
        assert gw._restart_drain_timeout == 60.0
        assert gw._restart_requested is False
        assert gw._draining is False
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_setup_signal_handlers_registers_sigusr1(self, tmp_path):
        """_setup_signal_handlers registers SIGUSR1 via loop.add_signal_handler."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        with patch.object(asyncio.get_event_loop(), "add_signal_handler") as mock_add:
            gw.supervisor.setup_signal_handlers()
            # Should be called for SIGINT, SIGTERM (from existing code), and SIGUSR1
            sigs_called = [call.args[0] for call in mock_add.call_args_list]
            assert signal.SIGUSR1 in sigs_called
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_on_message_during_drain_returns_none(self, tmp_path):
        """When draining, _on_message sends 'restarting' and returns None."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock()
        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter
        gw._draining = True

        event = _make_event(text="hello")
        result = await gw._on_message(event)

        assert result is None
        adapter.send_message.assert_called_once()
        sent_text = adapter.send_message.call_args[0][1]
        assert "restarting" in sent_text.lower()
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_on_message_suspended_session_archives_and_creates_fresh(self, tmp_path):
        """Suspended session is archived, fresh created, user notified."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="I'm back!")
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        # Create session then suspend it
        session = gw.session_store.get("feishu:chat1")
        session.add_message("user", "old msg")
        gw.session_store.suspend_session("feishu:chat1", reason="crash_recovery")

        event = _make_event(text="hello")
        result = await gw._on_message(event)

        # Should have sent a recovery message first
        recovery_calls = adapter.send_message.call_args_list
        assert len(recovery_calls) >= 1
        # Check that a recovery/notification message was sent (Chinese text about recovery)
        first_text = recovery_calls[0][0][1]
        assert "异常中断" in first_text or "恢复" in first_text

        # Session should no longer be suspended
        assert not gw.session_store.is_suspended("feishu:chat1")
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_active_session_tracking(self, tmp_path):
        """session_key is added to _active_sessions during agent processing."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()

        async def delayed_start(*args, **kwargs):
            await asyncio.sleep(0.05)

        async def delayed_send(*args, **kwargs):
            await asyncio.sleep(0.05)

        agent.start = AsyncMock(side_effect=delayed_start)
        agent.send_message = AsyncMock(side_effect=delayed_send)
        agent.chat = AsyncMock(return_value="done")
        agent.model = "test-model"
        agent._output_queue = asyncio.Queue()
        agent._tool_progress_callback = None
        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        async def check_active():
            event = _make_event(text="hello")
            async def run():
                return await gw._on_message(event)
            task = asyncio.create_task(run())
            await asyncio.sleep(0.01)
            # During _on_message(), session_key should be in active_sessions
            assert "feishu:chat1" in gw._active_sessions
            await task
            # After finish, session stays active (actor model runs in background)
            assert "feishu:chat1" in gw._active_sessions

        await check_active()
        gw.session_store.close()

    def test_check_recovery_clean_shutdown_file_exists(self, tmp_path):
        """If .clean_shutdown exists, _check_recovery_on_startup removes the file."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)

        # Create the marker file in the config's home_dir
        home_dir = config.home_dir
        home_dir.mkdir(parents=True, exist_ok=True)
        marker = home_dir / ".clean_shutdown"
        marker.write_text("clean")

        try:
            gw.supervisor.check_recovery_on_startup()
            # File should be removed
            assert not marker.exists()
        finally:
            if marker.exists():
                marker.unlink()

    def test_check_recovery_clean_shutdown_file_absent(self, tmp_path):
        """If no .clean_shutdown, sessions should be suspended."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)

        # Ensure no marker file in config home_dir
        home_dir = config.home_dir
        marker = home_dir / ".clean_shutdown"
        if marker.exists():
            marker.unlink()

        # Create a session and mark it recently active
        session = gw.session_store.get("test:key")
        session.add_message("user", "hello")

        gw.supervisor.check_recovery_on_startup()

        # Session should be suspended if it was recently active
        assert gw.session_store.is_suspended("test:key")
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_drain_notifies_and_clears_sessions(self, tmp_path):
        """_notify_active_sessions_of_restart sends message to active sessions."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        gw._active_sessions.add("feishu:sess1")
        gw._active_sessions.add("feishu:sess2")
        gw._session_to_adapter["feishu:sess1"] = "feishu"
        gw._session_to_adapter["feishu:sess2"] = "feishu"
        gw._active_chat_ids["feishu:sess1"] = "chat1"
        gw._active_chat_ids["feishu:sess2"] = "chat2"

        await gw.supervisor._notify_active_sessions()
        assert adapter.send_message.call_count == 2
        sent_text = adapter.send_message.call_args[0][1]
        assert "restarting" in sent_text.lower()
        gw.session_store.close()
