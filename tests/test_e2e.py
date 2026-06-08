"""End-to-end integration tests — full stack: user message → Gateway → agent → FakeLLM → tool → output → FakeAdapter.

Tests the entire production code path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tyagent.config import AgentConfig, TyAgentConfig
from tyagent.gateway import Gateway


pytestmark = pytest.mark.asyncio


# ============================================================
# Helpers
# ============================================================


def _make_e2e(tmp_path, fake_llm, fake_adapter):
    """Create a full test stack: Gateway + agent (via FakeLLM) + FakeAdapter."""
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
    gw._running = True
    return gw


async def _send(gw, fake_adapter, text, chat_id="e2e_chat", timeout=10.0):
    """Send a user message through _on_message and poll for output."""
    from tests.conftest import make_event
    await gw._on_message(make_event(text=text, chat_id=chat_id))

    for _ in range(int(timeout / 0.2)):
        if fake_adapter.sent_messages:
            return
        await asyncio.sleep(0.2)


# ============================================================
# Tests
# ============================================================


class TestEndToEnd:
    async def test_full_conversation_flow(self, tmp_path, fake_llm, fake_adapter):
        """User sends message, agent responds, output delivered to adapter."""
        gw = _make_e2e(tmp_path, fake_llm, fake_adapter)
        fake_llm.respond("Hello! How can I help you today?")

        await _send(gw, fake_adapter, "Hi there!")

        all_text = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "Hello" in all_text
        gw.session_store.close()

    async def test_tool_call_integration(self, tmp_path, fake_llm, fake_adapter):
        """Agent calls tools, executes them, returns result to user."""
        gw = _make_e2e(tmp_path, fake_llm, fake_adapter)

        fake_llm.chain(
            FakeLLMResponse(tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "conftest.py"})},
            }]),
            FakeLLMResponse(content="I found the test configuration file."),
        )

        await _send(gw, fake_adapter, "Read the conftest")

        all_text = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "test configuration" in all_text or "found" in all_text.lower()
        gw.session_store.close()

    async def test_multi_turn_conversation(self, tmp_path, fake_llm, fake_adapter):
        """Multiple messages in same session maintain context."""
        gw = _make_e2e(tmp_path, fake_llm, fake_adapter)

        fake_llm.chain(
            FakeLLMResponse(content="Nice to meet you!"),
            FakeLLMResponse(content="My name is TyAgent."),
        )

        await _send(gw, fake_adapter, "Hello", "multi_chat")
        all1 = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "Nice" in all1

        fake_adapter.clear()
        await _send(gw, fake_adapter, "What's your name?", "multi_chat")
        all2 = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "TyAgent" in all2

        gw.session_store.close()

    async def test_command_flow(self, tmp_path, fake_llm, fake_adapter):
        """Commands are dispatched without touching the agent."""
        gw = _make_e2e(tmp_path, fake_llm, fake_adapter)

        from tests.conftest import make_event
        await gw._on_message(make_event(text="/status", chat_id="cmd_e2e"))

        assert len(fake_adapter.sent_messages) >= 1
        status_text = fake_adapter.sent_messages[0][1]
        assert "tyagent" in status_text.lower()
        gw.session_store.close()

    async def test_session_persistence(self, tmp_path, fake_llm, fake_adapter):
        """Messages persist across turns and survive /new."""
        gw = _make_e2e(tmp_path, fake_llm, fake_adapter)

        fake_llm.respond("Message one response")
        await _send(gw, fake_adapter, "Message one")
        fake_adapter.clear()

        fake_llm.respond("Message two response")
        await _send(gw, fake_adapter, "Message two")

        # Both messages should be in session store
        count = gw.session_store.get_message_count("test:e2e_chat")
        assert count >= 2

        # /new archives and starts fresh
        fake_adapter.clear()
        from tests.conftest import make_event
        await gw._on_message(make_event(text="/new", chat_id="e2e_chat"))

        session = gw.session_store.get("test:e2e_chat")
        assert "current_session_id" in session.metadata
        gw.session_store.close()

    async def test_error_recovery(self, tmp_path, fake_llm, fake_adapter):
        """After LLM error, user sees error message and can continue."""
        gw = _make_e2e(tmp_path, fake_llm, fake_adapter)

        # First message: error
        fake_llm.respond("__ERROR__:500:internal error")
        await _send(gw, fake_adapter, "bad request")

        all1 = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "error" in all1.lower() or "500" in all1, f"got: {fake_adapter.sent_messages}"

        # Second message: should still work
        fake_adapter.clear()
        fake_llm.respond("Recovered successfully!")
        await _send(gw, fake_adapter, "try again")

        all2 = " ".join(t for _, t, _ in fake_adapter.sent_messages)
        assert "Recovered" in all2, f"got: {fake_adapter.sent_messages}"

        gw.session_store.close()


# ============================================================
# Imports
# ============================================================


from tests.conftest import FakeLLMResponse
