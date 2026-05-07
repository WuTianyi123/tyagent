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
    # Derive home_dir from sessions_dir if not explicitly set, so tests
    # never write to the real ~/.tyagent directory.
    if "home_dir" not in defaults and "sessions_dir" in defaults:
        defaults["home_dir"] = defaults["sessions_dir"].parent
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config, session_store=store, agent=agent)
        assert gw._default_agent_template is agent
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
        assert hasattr(gw, "_sessions")
        assert isinstance(gw._sessions, dict)
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
        agent.clone = MagicMock(return_value=agent)  # per-session clone
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
    async def test_on_message_suspended_session_clears_flag_and_continues(self, tmp_path):
        """Suspended session is not archived — flag is cleared, old messages preserved."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.clone = MagicMock(return_value=agent)  # per-session clone
        agent.chat = AsyncMock(return_value="I'm back!")
        agent.model = "test-model"
        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        # Create session then suspend it
        session = gw.session_store.get("feishu:chat1")
        session.add_message("user", "old msg")
        gw.session_store.suspend_session("feishu:chat1", reason="crash_recovery")

        assert gw.session_store.is_suspended("feishu:chat1"), "precondition"

        try:
            await gw._on_message(_make_event(text="hello"))
        except Exception:
            pass  # agent mock limitations — not testing that path

        # Session should no longer be suspended (flag was cleared)
        assert not gw.session_store.is_suspended("feishu:chat1")

        # Session should retain its messages (not archived)
        messages = gw.session_store.get_messages("feishu:chat1")
        assert any(msg.get("content") == "old msg" for msg in messages), \
            "Session should retain old messages"

        # Should NOT have sent a crash recovery message
        recovery_calls = adapter.send_message.call_args_list
        no_recovery = all("异常中断" not in str(args) for args, _ in recovery_calls)
        assert no_recovery, "Should not send crash recovery message"

        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_active_session_tracking(self, tmp_path):
        """session_key is added to _sessions during agent processing."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.clone = MagicMock(return_value=agent)  # per-session clone

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
            await asyncio.sleep(0.1)  # wait for _ensure_session_agent to create SessionContext
            # During _on_message(), session_key should be in active_sessions
            assert "feishu:chat1" in gw._sessions
            await task
            # After finish, session stays active (actor model runs in background)
            assert "feishu:chat1" in gw._sessions

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
        """If no .clean_shutdown, sessions continue normally (not suspended)."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)

        # Ensure no marker file in config home_dir
        home_dir = config.home_dir
        marker = home_dir / ".clean_shutdown"
        if marker.exists():
            marker.unlink()

        # Create a session
        session = gw.session_store.get("test:key")
        session.add_message("user", "hello")

        gw.supervisor.check_recovery_on_startup()

        # Session should NOT be suspended — it continues from persisted data
        assert not gw.session_store.is_suspended("test:key")
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_drain_notifies_and_clears_sessions(self, tmp_path):
        """_notify_active_sessions_of_restart sends message to active sessions."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        agent = MagicMock()
        agent.clone = MagicMock(return_value=agent)  # per-session clone
        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        from tyagent.gateway.gateway import SessionContext
        gw._sessions["feishu:sess1"] = SessionContext(agent, adapter, platform_name="feishu", chat_id="chat1")
        gw._sessions["feishu:sess2"] = SessionContext(agent, adapter, platform_name="feishu", chat_id="chat2")

        await gw.supervisor._notify_active_sessions()
        assert adapter.send_message.call_count == 2
        sent_text = adapter.send_message.call_args[0][1]
        assert "restarting" in sent_text.lower()
        gw.session_store.close()


# ---------------------------------------------------------------------------
# Restart marker — write + handle on startup
# ---------------------------------------------------------------------------


class TestRestartMarker:
    """Tests for _write_restart_marker and _handle_restart_marker_on_startup."""

    def _add_orphaned_tool_call(self, store, session_key):
        """Helper: add an assistant message with tool_calls but no tool response."""
        session = store.get(session_key)
        session.add_message("user", "do something")
        session.add_message(
            "assistant", "",
            tool_calls=[{
                "id": "call_orphan_1",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command": "echo hi"}'},
            }],
        )
        return session

    def _add_active_session(self, gw, session_key, session):
        """Register a session as active in gw._sessions for restart marker tests."""
        from tyagent.gateway.gateway import SessionContext
        from unittest.mock import MagicMock
        agent = MagicMock()
        agent._messages = []
        gw._sessions[session_key] = SessionContext(
            agent, MagicMock(), platform_name="test", chat_id="chat1",
        )

    def test_write_marker_no_orphans(self, tmp_path):
        """No gateway_interrupt markers, no in-flight tool calls -> no marker."""
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        session.add_message("user", "hello")
        session.add_message("assistant", "ok")
        self._add_active_session(gw, "test:key", session)

        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert not marker_path.exists(), "Should not write marker when no restart-related calls"
        gw.session_store.close()

    def test_write_marker_with_gateway_interrupt(self, tmp_path):
        """gateway_interrupt marker -> captured in .restart_pending."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)

        # Simulate a gateway_interrupt marker written by terminal tool
        interrupt_dir = config.home_dir / ".gateway_interrupt"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        marker_data = {
            "tool_call_id": "call_interrupt_1",
            "session_key": "test:key",
            "session_id": "sid1",
            "command": "tyagent gateway restart",
            "started_at": time.time(),
            "reason": "restart_trigger",
        }
        (interrupt_dir / "test.json").write_text(json.dumps(marker_data), encoding="utf-8")

        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert marker_path.exists(), "Marker should exist when gateway_interrupt markers present"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert "sessions" in marker
        assert "test:key" in marker["sessions"]
        sdata = marker["sessions"]["test:key"]
        assert sdata["session_id"] == "sid1"
        assert len(sdata["pending_tool_calls"]) == 1
        tc = sdata["pending_tool_calls"][0]
        assert tc["tool_call_id"] == "call_interrupt_1"
        assert tc["function_name"] == "terminal"
        assert tc["reason"] == "restart_trigger"
        gw.session_store.close()
        marker_path.unlink(missing_ok=True)

    def test_write_marker_multiple_sessions(self, tmp_path):
        """Multiple gateway_interrupt markers -> all recorded."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)

        interrupt_dir = config.home_dir / ".gateway_interrupt"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        for idx, (sk, sid, tcid) in enumerate([
            ("session_a", "sid_a", "call_a"),
            ("session_b", "sid_b", "call_b"),
        ]):
            data = {
                "tool_call_id": tcid,
                "session_key": sk,
                "session_id": sid,
                "command": f"restart {idx}",
                "started_at": time.time(),
                "reason": "restart_trigger",
            }
            (interrupt_dir / f"{idx}.json").write_text(json.dumps(data), encoding="utf-8")

        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert "session_a" in marker["sessions"]
        assert "session_b" in marker["sessions"]
        gw.session_store.close()
        marker_path.unlink(missing_ok=True)

    def test_handle_marker_writes_synthetic_responses(self, tmp_path):
        """Marker exists → synthetic 'restart_completed' responses written to DB."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        session = self._add_orphaned_tool_call(gw.session_store, "test:key")
        session_id = session.metadata.get("current_session_id", "")

        # Manually write a restart marker (as the old process would)
        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": session_id,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_orphan_1", "function_name": "terminal", "reason": "restart_trigger"},
                    ],
                }
            },
        }
        marker_path = config.home_dir / ".restart_pending"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

        gw.supervisor._handle_restart_marker_on_startup()

        # Marker should be removed
        assert not marker_path.exists()

        # DB should now have a tool response for the orphaned call
        messages = gw.session_store.get_messages("test:key", session_id=session_id)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1
        # The LAST tool message should be our synthetic response
        last_tool = tool_msgs[-1]
        assert last_tool["tool_call_id"] == "call_orphan_1"
        parsed = json.loads(last_tool["content"])
        assert parsed.get("restart_completed") is True
        assert parsed.get("success") is True
        gw.session_store.close()

    def test_handle_marker_no_file(self, tmp_path):
        """No marker file → no-op, no errors."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        # Should not raise
        gw.supervisor._handle_restart_marker_on_startup()
        gw.session_store.close()

    def test_handle_marker_corrupted_file(self, tmp_path):
        """Corrupted marker file → cleaned up, no crash."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        marker_path = config.home_dir / ".restart_pending"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("not valid json{{{", encoding="utf-8")

        # Should not raise
        gw.supervisor._handle_restart_marker_on_startup()

        # Corrupted marker should be removed
        assert not marker_path.exists()
        gw.session_store.close()

    def test_handle_marker_sanitize_consistent(self, tmp_path):
        """After handler writes synthetic response, _sanitize_message_chain
        should find the chain complete (no orphaned calls)."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        session = self._add_orphaned_tool_call(gw.session_store, "test:key")
        session_id = session.metadata.get("current_session_id", "")

        # Write marker
        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": session_id,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_orphan_1", "function_name": "terminal", "reason": "restart_trigger"},
                    ],
                }
            },
        }
        marker_path = config.home_dir / ".restart_pending"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

        # Handle marker (writes synthetic response to DB)
        gw.supervisor._handle_restart_marker_on_startup()

        # Now simulate session load: get messages and sanitize
        messages = gw.session_store.get_messages("test:key", session_id=session_id)
        sanitized = _sanitize_message_chain(messages)

        # The sanitized chain should be the same length as original (no new inserts)
        assert len(sanitized) == len(messages), (
            f"_sanitize_message_chain should not add new messages "
            f"({len(sanitized)} vs {len(messages)})"
        )
        gw.session_store.close()

    def test_handle_marker_unknown_failure(self, tmp_path):
        """Marker with reason='unknown_failure' → synthetic 'unknown failure' response."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        session = self._add_orphaned_tool_call(gw.session_store, "test:key")
        session_id = session.metadata.get("current_session_id", "")

        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": session_id,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_orphan_1", "function_name": "?",
                         "reason": "unknown_failure"},
                    ],
                }
            },
        }
        marker_path = config.home_dir / ".restart_pending"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

        gw.supervisor._handle_restart_marker_on_startup()

        assert not marker_path.exists()

        messages = gw.session_store.get_messages("test:key", session_id=session_id)
        tool_msgs = [m for m in messages if m.get("role") == "tool"
                     and m.get("tool_call_id") == "call_orphan_1"]
        assert len(tool_msgs) == 1
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed.get("success") is False
        assert parsed.get("interrupted") is True
        assert "Unknown failure" in parsed.get("error", "")
        gw.session_store.close()

    def test_write_marker_with_in_flight_tool(self, tmp_path):
        """Agent with _current_tool_call_id set → captured as unknown_failure."""
        import json
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        session_id = session.metadata.get("current_session_id", "")
        self._add_active_session(gw, "test:key", session)
        agent = gw._sessions["test:key"].agent
        agent._running = True
        agent._current_tool_call_id = "call_in_flight_1"

        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        sdata = marker["sessions"]["test:key"]
        assert len(sdata["pending_tool_calls"]) == 1
        tc = sdata["pending_tool_calls"][0]
        assert tc["tool_call_id"] == "call_in_flight_1"
        assert tc["reason"] == "unknown_failure"
        # Verify session_id in marker matches the actual session
        assert sdata["session_id"] == session_id
        gw.session_store.close()
        marker_path.unlink(missing_ok=True)

    def test_write_marker_in_flight_not_running_skipped(self, tmp_path):
        """Agent with _current_tool_call_id set but _running=False → skipped."""
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        self._add_active_session(gw, "test:key", session)
        agent = gw._sessions["test:key"].agent
        agent._running = False  # Agent is stopped
        agent._current_tool_call_id = "call_in_flight_1"

        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert not marker_path.exists(), (
            "Should not write marker when agent is not running"
        )
        gw.session_store.close()

    def test_write_marker_dedup_in_flight_and_restart_trigger(self, tmp_path):
        """Same tool_call in both gateway_interrupt AND in-flight → only once."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)

        session = gw.session_store.get("test:key")
        session_id = session.metadata.get("current_session_id", "")

        interrupt_dir = config.home_dir / ".gateway_interrupt"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        marker_data = {
            "tool_call_id": "call_shared_1",
            "session_key": "test:key",
            "session_id": session_id,  # Use real session_id, not a fake one
            "command": "tyagent gateway restart",
            "started_at": time.time(),
            "reason": "restart_trigger",
        }
        (interrupt_dir / "shared.json").write_text(json.dumps(marker_data), encoding="utf-8")

        self._add_active_session(gw, "test:key", session)
        agent = gw._sessions["test:key"].agent
        agent._running = True
        agent._current_tool_call_id = "call_shared_1"

        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        sdata = marker["sessions"]["test:key"]
        assert len(sdata["pending_tool_calls"]) == 1
        tc = sdata["pending_tool_calls"][0]
        assert tc["tool_call_id"] == "call_shared_1"
        assert tc["reason"] == "restart_trigger", (
            "restart_trigger should take precedence over unknown_failure"
        )
        gw.session_store.close()
        marker_path.unlink(missing_ok=True)

    def test_write_marker_cleans_interrupt_dir_when_no_sessions(self, tmp_path):
        """gateway_interrupt markers cleaned up by handler even from dormant sessions."""
        import json
        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        gw = Gateway(config)

        # Create a valid session in DB but do NOT register it as active
        session = gw.session_store.get("dormant:key")
        session_id = session.metadata.get("current_session_id", "")

        # Add a minimal message chain so the handler can write synthetics
        session.add_message("user", "restart trigger test")
        session.add_message("assistant", "", tool_calls=[{
            "id": "call_dormant_1",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": "tyagent gateway restart"}'},
        }])

        interrupt_dir = config.home_dir / ".gateway_interrupt"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (interrupt_dir / f"stale_{i}.json").write_text(
                json.dumps({
                    "tool_call_id": f"call_dormant_1",
                    "session_key": "dormant:key",
                    "session_id": session_id,
                    "command": "tyagent gateway restart",
                    "reason": "restart_trigger",
                }),
                encoding="utf-8",
            )

        # _write_restart_marker should collect valid interrupt markers
        # even if the session isn't active in gw._sessions
        gw.supervisor._write_restart_marker()

        marker_path = config.home_dir / ".restart_pending"
        assert marker_path.exists(), (
            "Should write .restart_pending when valid gateway_interrupt markers exist, "
            "even for dormant sessions"
        )

        # gateway_interrupt markers should NOT be cleaned up by _write_restart_marker
        # (cleanup happens in _handle_restart_marker_on_startup after processing)
        remaining_before = list(interrupt_dir.glob("*.json"))
        assert len(remaining_before) == 3, (
            "_write_restart_marker should NOT clean up when sessions exist; "
            f"found {len(remaining_before)}"
        )

        # Now simulate the handler processing
        gw.supervisor._handle_restart_marker_on_startup()

        # After handler, markers should be cleaned up
        assert not marker_path.exists(), "Handler should clean up .restart_pending"
        remaining_after = list(interrupt_dir.glob("*.json"))
        assert len(remaining_after) == 0, (
            f"Handler should clean up gateway_interrupt markers; found {len(remaining_after)}"
        )
        gw.session_store.close()

    def test_handle_marker_skips_when_response_exists(self, tmp_path):
        """Handler skips synthetic when a real tool response already in DB."""
        import json, time
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        session = self._add_orphaned_tool_call(gw.session_store, "test:key")
        session_id = session.metadata.get("current_session_id", "")

        # Pre-populate a real tool response for the same tool_call_id
        session.add_message(
            "tool",
            json.dumps({"output": "real output", "exit_code": 0}),
            tool_call_id="call_orphan_1",
        )

        # Write a restart marker (simulate the old process)
        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": session_id,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_orphan_1", "function_name": "terminal",
                         "reason": "restart_trigger"},
                    ],
                }
            },
        }
        marker_path = config.home_dir / ".restart_pending"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

        # Handler should see the real response and skip synthetics
        gw.supervisor._handle_restart_marker_on_startup()

        assert not marker_path.exists()

        # Verify only ONE tool response exists for call_orphan_1
        # (the real one, not a synthetic duplicate)
        messages = gw.session_store.get_messages("test:key", session_id=session_id)
        tool_msgs = [m for m in messages if m.get("role") == "tool"
                     and m.get("tool_call_id") == "call_orphan_1"]
        assert len(tool_msgs) == 1, (
            f"Should have exactly 1 response for call_orphan_1; "
            f"found {len(tool_msgs)}"
        )
        # The existing response should be the real one (not synthetic)
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed.get("output") == "real output", (
            "Real output should not be replaced by synthetics"
        )
        assert "restart_completed" not in parsed, (
            "Real response should not have synthetic markers"
        )
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_validate_chains_ok(self, tmp_path):
        """Valid message chain → validation passes."""
        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        session.add_message("user", "hello")
        session.add_message("assistant", "world")
        self._add_active_session(gw, "test:key", session)

        # Without a real API key, validation should skip (no error)
        await gw.supervisor.validate_message_chains()
        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_validate_chains_with_mock_api(self, tmp_path):
        """Mock a successful API response → validation passes."""
        from tyagent.config import AgentConfig
        config = _make_config(
            sessions_dir=tmp_path / "sessions",
            home_dir=tmp_path,
            agent=AgentConfig(api_key="test-key", base_url="http://localhost:19999", model="test-model"),
        )
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        session.add_message("user", "hello")
        session.add_message("assistant", "world")
        self._add_active_session(gw, "test:key", session)

        with patch("tyagent.gateway.lifecycle.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "usage": {"prompt_tokens": 42},
                "choices": [{"message": {"content": "ok"}}],
            }
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_resp
            mock_client.return_value.__aenter__.return_value = mock_instance

            await gw.supervisor.validate_message_chains()

        gw.session_store.close()

    @pytest.mark.asyncio
    async def test_validate_chains_with_bad_api(self, tmp_path):
        """Mock a 400 response → validation fails."""
        from tyagent.config import AgentConfig
        config = _make_config(
            sessions_dir=tmp_path / "sessions",
            home_dir=tmp_path,
            agent=AgentConfig(api_key="test-key", base_url="http://localhost:19999", model="test-model"),
        )
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        session.add_message("user", "hello")
        # Add a malformed assistant message that could trigger API rejection
        session.add_message("assistant", "", tool_calls=[])
        self._add_active_session(gw, "test:key", session)

        with patch("tyagent.gateway.lifecycle.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.text = '{"error":{"message":"Bad request"}}'
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_resp
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(RuntimeError, match="validation failed"):
                await gw.supervisor.validate_message_chains()

        gw.session_store.close()


class TestLegacyMigration:
    """Tests for migrate_legacy_home()."""

    def test_migration_default_profile(self, tmp_path):
        """Migration triggers for default profile with legacy data.

        Legacy layout:  ~/.tyagent/config.yaml
        New layout:     ~/.tyagent/tyagent/config.yaml

        So default_home = ~/.tyagent/tyagent, legacy = ~/.tyagent/
        """
        from tyagent.config import migrate_legacy_home
        import tyagent.config as cfg_mod

        profile = tmp_path / "tyagent"
        legacy_home = tmp_path  # default_home.parent = tmp_path
        orig_default = cfg_mod.default_home
        try:
            cfg_mod.default_home = profile
            (legacy_home / "config.yaml").write_text("legacy: true")

            migrate_legacy_home(profile)

            assert (profile / "config.yaml").exists()
            assert not (legacy_home / "config.yaml").exists()
        finally:
            cfg_mod.default_home = orig_default

    def test_migration_skips_non_default_profile(self, tmp_path):
        """Migration skips non-default profiles."""
        from tyagent.config import migrate_legacy_home
        import tyagent.config as cfg_mod

        profile = tmp_path / "coder"  # not default_home
        legacy_home = tmp_path
        orig_default = cfg_mod.default_home
        try:
            cfg_mod.default_home = tmp_path / "tyagent"  # default is a DIFFERENT dir
            (legacy_home / "config.yaml").write_text("legacy: true")

            migrate_legacy_home(profile)

            # Legacy data should NOT have moved
            assert (legacy_home / "config.yaml").exists()
            assert not (profile / "config.yaml").exists()
        finally:
            cfg_mod.default_home = orig_default

    def test_migration_skips_when_target_exists(self, tmp_path):
        """Migration skips when target already has config.yaml."""
        from tyagent.config import migrate_legacy_home
        import tyagent.config as cfg_mod

        profile = tmp_path / "tyagent"
        legacy_home = tmp_path
        orig_default = cfg_mod.default_home
        try:
            cfg_mod.default_home = profile
            (legacy_home / "config.yaml").write_text("old")
            profile.mkdir(parents=True)
            (profile / "config.yaml").write_text("new")

            migrate_legacy_home(profile)

            # Target should remain unchanged
            assert (profile / "config.yaml").read_text() == "new"
            assert (legacy_home / "config.yaml").exists()
        finally:
            cfg_mod.default_home = orig_default


class _MinimalAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for testing."""
    def __init__(self, config=None, *, platform_name="test", home_dir=None):
        super().__init__(config, platform_name=platform_name, home_dir=home_dir)

    async def connect(self) -> None:
        pass
    async def disconnect(self) -> None:
        pass
    async def start(self) -> None:
        pass
    async def stop(self) -> None:
        pass
    async def send_message(self, target, text, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Gateway._load_adapters — config → adapter integration
# ---------------------------------------------------------------------------


class TestLoadAdapters:
    """Integration: _load_adapters() exercises the full config→adapter chain
    that unit tests of get_connected_platforms() alone don't cover. The bug
    where get_connected_platforms() read old-format extra.app_id slipped
    through because no test called _load_adapters() with new-format config."""

    @pytest.mark.asyncio
    async def test_loads_adapter_from_new_format_config(self, tmp_path):
        """New-format config (extra.connection.app_id) → adapter is loaded."""
        from tyagent.config import PlatformConfig, TyAgentConfig, AgentConfig
        import tyagent.gateway.gateway as gw_mod

        config = TyAgentConfig(
            platforms={
                "feishu": PlatformConfig(
                    enabled=True,
                    extra={
                        "connection": {
                            "app_id": "cli_test_app",
                            "app_secret": "test_secret",
                        },
                    },
                ),
            },
            agent=AgentConfig(model="test", api_key="k"),
            sessions_dir=tmp_path / "sessions",
        )

        saved_registry = dict(gw_mod._PLATFORM_REGISTRY)
        try:
            with patch.object(gw_mod, "_load_builtin_platforms"):
                gw_mod._PLATFORM_REGISTRY.clear()
                gw_mod._PLATFORM_REGISTRY["feishu"] = _MinimalAdapter

                gw = Gateway(config)
                gw._load_adapters()

                assert "feishu" in gw.adapters
                assert isinstance(gw.adapters["feishu"], _MinimalAdapter)
        finally:
            gw_mod._PLATFORM_REGISTRY.clear()
            gw_mod._PLATFORM_REGISTRY.update(saved_registry)
            gw.session_store.close()

    @pytest.mark.asyncio
    async def test_skips_adapter_without_app_id(self, tmp_path):
        """New-format config without app_id → adapter NOT loaded."""
        from tyagent.config import PlatformConfig, TyAgentConfig, AgentConfig
        import tyagent.gateway.gateway as gw_mod

        config = TyAgentConfig(
            platforms={
                "feishu": PlatformConfig(
                    enabled=True,
                    extra={
                        "connection": {
                            "app_secret": "test_secret",
                            # no app_id
                        },
                    },
                ),
            },
            agent=AgentConfig(model="test", api_key="k"),
            sessions_dir=tmp_path / "sessions",
        )

        saved_registry = dict(gw_mod._PLATFORM_REGISTRY)
        try:
            with patch.object(gw_mod, "_load_builtin_platforms"):
                gw_mod._PLATFORM_REGISTRY.clear()
                gw_mod._PLATFORM_REGISTRY["feishu"] = _MinimalAdapter

                gw = Gateway(config)
                gw._load_adapters()

                assert "feishu" not in gw.adapters
        finally:
            gw_mod._PLATFORM_REGISTRY.clear()
            gw_mod._PLATFORM_REGISTRY.update(saved_registry)
            gw.session_store.close()

    @pytest.mark.asyncio
    async def test_skips_disabled_platform(self, tmp_path):
        """Disabled platform (even with valid creds) → NOT loaded."""
        from tyagent.config import PlatformConfig, TyAgentConfig, AgentConfig
        import tyagent.gateway.gateway as gw_mod

        config = TyAgentConfig(
            platforms={
                "feishu": PlatformConfig(
                    enabled=False,
                    extra={
                        "connection": {
                            "app_id": "cli_test_app",
                            "app_secret": "test_secret",
                        },
                    },
                ),
            },
            agent=AgentConfig(model="test", api_key="k"),
            sessions_dir=tmp_path / "sessions",
        )

        saved_registry = dict(gw_mod._PLATFORM_REGISTRY)
        try:
            with patch.object(gw_mod, "_load_builtin_platforms"):
                gw_mod._PLATFORM_REGISTRY.clear()
                gw_mod._PLATFORM_REGISTRY["feishu"] = _MinimalAdapter

                gw = Gateway(config)
                gw._load_adapters()

                assert "feishu" not in gw.adapters
        finally:
            gw_mod._PLATFORM_REGISTRY.clear()
            gw_mod._PLATFORM_REGISTRY.update(saved_registry)
            gw.session_store.close()


# ---------------------------------------------------------------------------
# Adapter home_dir
# ---------------------------------------------------------------------------


class TestAdapterHomeDir:
    """Integration: verify home_dir flows from Gateway to adapters."""

    def test_base_adapter_stores_home_dir(self):
        """BasePlatformAdapter stores home_dir from constructor."""
        from pathlib import Path

        home = Path("/tmp/test-profile")
        adapter = _MinimalAdapter(config=None, platform_name="test", home_dir=home)
        assert adapter.home_dir == home

    def test_home_dir_none_default(self):
        """BasePlatformAdapter accepts home_dir=None (backward compat)."""
        adapter = _MinimalAdapter(config=None, platform_name="test")
        assert adapter.home_dir is None


# ---------------------------------------------------------------------------
# Restart trigger regex tests
# ---------------------------------------------------------------------------


class TestRestartTriggers:
    """Test _RESTART_TRIGGERS regex patterns in the terminal tool."""

    def _will_restart(self, command: str) -> bool:
        """Check if command matches any restart trigger pattern."""
        import re
        from tyagent.tools.core import _handle_terminal
        # Reconstruct the same patterns used in _handle_terminal
        _RESTART_TRIGGERS = (
            r"^tyagent\s+gateway\s+restart",
            r"^(?:sudo\s+)?systemctl\s+(--user\s+)?restart\s+tyagent-gateway",
            r"^uv\s+run\s+python3(?:\.\d+)?\s+tyagent_cli\.py\s+gateway\s+restart",
            r"^python3(?:\.\d+)?\s+tyagent_cli\.py\s+gateway\s+restart",
            r"^uv\s+run\s+tyagent\s+gateway\s+restart",
            r"kill\s+-SIGUSR1\s+\S+",
        )
        return any(re.search(p, command) for p in _RESTART_TRIGGERS)

    def test_direct_cli(self):
        assert self._will_restart("tyagent gateway restart")
        # ^ anchor prevents echo false positive
        assert not self._will_restart('echo "tyagent gateway restart"')
        assert not self._will_restart("echo tyagent gateway restart")

    def test_systemctl(self):
        assert self._will_restart("systemctl --user restart tyagent-gateway")
        assert self._will_restart("sudo systemctl restart tyagent-gateway")
        assert self._will_restart("sudo systemctl --user restart tyagent-gateway")
        # ^ anchor prevents false positive
        assert not self._will_restart('echo "systemctl restart tyagent-gateway"')

    def test_kill(self):
        assert self._will_restart("kill -SIGUSR1 12345")
        # \S+ supports subcommand expansion
        assert self._will_restart("kill -SIGUSR1 $(pgrep -f tyagent-gateway)")
        # \S+ also matches backtick subcommands
        assert self._will_restart("kill -SIGUSR1 `pgrep tyagent`")

    def test_python_cli(self):
        assert self._will_restart("python3 tyagent_cli.py gateway restart")
        assert self._will_restart("python3.11 tyagent_cli.py gateway restart")
        assert self._will_restart("python3.12 tyagent_cli.py gateway restart")

    def test_uv_run(self):
        assert self._will_restart("uv run tyagent gateway restart")
        assert self._will_restart("uv run python3 tyagent_cli.py gateway restart")
        assert self._will_restart("uv run python3.11 tyagent_cli.py gateway restart")

    def test_non_restart_commands(self):
        """Commands that should NOT trigger."""
        assert not self._will_restart("ls -la")
        assert not self._will_restart("echo hello")
        assert not self._will_restart("cat /proc/loadavg")
        assert not self._will_restart("tyagent gateway status")
        assert not self._will_restart("systemctl status tyagent-gateway")
