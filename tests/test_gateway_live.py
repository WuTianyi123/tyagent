"""Gateway integration tests — real Gateway orchestration against agent with FakeLLM.

Tests exercise message routing, command dispatch, actor model, session lifecycle,
and error propagation through FakeAdapter.
"""

from __future__ import annotations

import asyncio
from pathlib import Path as P

import pytest

from tyagent.config import AgentConfig, TyAgentConfig
from tyagent.gateway import Gateway


pytestmark = pytest.mark.asyncio


# ============================================================
# Helpers
# ============================================================


def _make_gw(tmp_path, fake_llm, fake_adapter):
    """Create a Gateway with real agent (FakeLLM) and FakeAdapter.

    The Gateway's _get_or_create_agent uses clone() to create
    per-session agents. make_test_agent already monkey-patches
    clone() to carry the FakeLLM.
    """
    from tests.conftest import make_test_agent

    agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

    config = TyAgentConfig(
        agent=AgentConfig(
            model="test-model",
            api_key="test-key",
            base_url="http://fake.test/v1",
            max_tool_turns=50,
        ),
        sessions_dir=tmp_path / "sessions",
    )

    gw = Gateway(config, agent=agent)
    gw.adapters["feishu"] = fake_adapter
    gw._running = True  # Required for _consume_output loop
    return gw


async def _send_and_poll(gw, fake_adapter, text, chat_id="chat1", timeout=10.0):
    """Send a message and poll for adapter output until delivered or timeout."""
    from tests.conftest import make_event
    event = make_event(text=text, chat_id=chat_id)
    await gw._on_message(event)

    # Poll for output delivery
    for _ in range(int(timeout / 0.2)):
        if fake_adapter.sent_messages:
            return
        await asyncio.sleep(0.2)

    # One final check
    if fake_adapter.sent_messages:
        return


# ============================================================
# Tests
# ============================================================


class TestMessageRouting:
    async def test_message_produces_agent_response(self, tmp_path, fake_llm, fake_adapter):
        """A user message goes through agent and produces a response."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)
        fake_llm.respond("Hello from the agent!")

        await _send_and_poll(gw, fake_adapter, "Hi there")

        all_text = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "Hello from the agent!" in all_text, f"got messages: {fake_adapter.sent_messages}"
        gw.session_store.close()

    async def test_message_persisted_to_session(self, tmp_path, fake_llm, fake_adapter):
        """User and assistant messages are persisted to SessionStore."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)
        fake_llm.respond("Persisted response")

        await _send_and_poll(gw, fake_adapter, "Hello world")

        count = gw.session_store.get_message_count("test:chat1")
        assert count >= 1
        gw.session_store.close()

    async def test_two_different_chats(self, tmp_path, fake_llm, fake_adapter):
        """Two different chat_ids get independent agent sessions."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)
        fake_llm.chain(
            FakeLLMResponse(content="Response A"),
            FakeLLMResponse(content="Response B"),
        )

        await _send_and_poll(gw, fake_adapter, "msg A", "chatA")
        fake_adapter.clear()
        await _send_and_poll(gw, fake_adapter, "msg B", "chatB")

        all_text = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "Response B" in all_text, f"got: {fake_adapter.sent_messages}"
        gw.session_store.close()

    async def test_agent_error_delivers_to_user(self, tmp_path, fake_llm, fake_adapter):
        """When LLM returns error, user gets error message."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)
        fake_llm.respond("__ERROR__:500:internal error")

        await _send_and_poll(gw, fake_adapter, "crash me")

        all_text = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "error" in all_text.lower() or "500" in all_text, f"got: {fake_adapter.sent_messages}"
        gw.session_store.close()


class TestCommands:
    async def test_status_command(self, tmp_path, fake_llm, fake_adapter):
        """Status command returns session info."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)

        from tests.conftest import make_event
        await gw._on_message(make_event(text="/status", chat_id="cmd_status"))

        assert len(fake_adapter.sent_messages) >= 1
        status_text = fake_adapter.sent_messages[0][1]
        assert "tyagent" in status_text.lower()
        gw.session_store.close()

    async def test_new_command_archives(self, tmp_path, fake_llm, fake_adapter):
        """/new archives current session and creates fresh one."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)

        # First send a message to create session history
        fake_llm.respond("First response")
        await _send_and_poll(gw, fake_adapter, "message 1", "chat_new")
        fake_adapter.clear()

        # Then /new
        await gw._on_message(make_event(text="/new", chat_id="chat_new"))

        assert len(fake_adapter.sent_messages) >= 1
        session = gw.session_store.get("test:chat_new")
        assert "current_session_id" in session.metadata
        gw.session_store.close()

    async def test_unknown_platform_skipped(self, tmp_path, fake_llm, fake_adapter):
        """Message from unknown platform returns None."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)

        from tests.conftest import make_event
        event = make_event(text="hello", chat_id="noexist", platform="unknown")
        result = await gw._on_message(event)

        assert result is None
        gw.session_store.close()


class TestDrain:
    async def test_message_rejected_during_drain(self, tmp_path, fake_llm, fake_adapter):
        """Messages during drain are politely rejected."""
        gw = _make_gw(tmp_path, fake_llm, fake_adapter)
        gw._draining = True

        from tests.conftest import make_event
        await gw._on_message(make_event(text="hello during drain", chat_id="drain_test"))

        assert any("restart" in t.lower() for _, t, _ in fake_adapter.sent_messages)
        gw.session_store.close()


# ============================================================
# Imports
# ============================================================


from tests.conftest import FakeLLMResponse, make_event
