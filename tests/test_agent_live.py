"""Agent integration tests — real TyAgent loop against FakeLLM at HTTP boundary.

Tests exercise the full agent lifecycle: chat(), _run_turn(), tool dispatch,
actor model (start/send_message/stop), streaming, error recovery,
and compaction. The only mock is at the HTTP boundary (FakeLLM replaces
httpx.AsyncClient) — everything else runs real production code.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tyagent.agent import TyAgent, AgentError
from tests.conftest import FakeLLM, FakeLLMResponse


pytestmark = pytest.mark.asyncio


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_agent(fake_llm, tmp_path):
    """Agent with FakeLLM for basic tests — no actor model."""
    from tests.conftest import make_test_agent
    return make_test_agent(fake_llm, home_dir=tmp_path / "home")


# ============================================================
# Basic chat — single turn
# ============================================================


class TestChatBasic:
    """Agent responds to user messages with text from FakeLLM."""

    async def test_simple_response(self, fake_llm, simple_agent):
        """Single-turn text response."""
        fake_llm.respond("Hello from the agent!")

        result = await simple_agent.chat([{"role": "user", "content": "Hi"}])

        assert result == "Hello from the agent!"
        assert fake_llm.call_count() == 1

    async def test_system_prompt_in_request(self, fake_llm, simple_agent):
        """System prompt is in the API request messages."""
        fake_llm.respond("ok")

        await simple_agent.chat([{"role": "user", "content": "hello"}])

        msgs = fake_llm.last_messages
        assert msgs is not None
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    async def test_multiple_messages_accumulate_in_history(self, fake_llm, simple_agent):
        """Conversation history is preserved across turns (chat() API)."""
        fake_llm.respond("First reply")
        messages = [{"role": "user", "content": "msg1"}]
        await simple_agent.chat(messages)
        assert len(messages) == 2  # user + assistant

        fake_llm.respond("Second reply")
        messages.append({"role": "user", "content": "msg2"})
        await simple_agent.chat(messages)
        assert len(messages) == 4  # user, assistant, user, assistant


# ============================================================
# Tool call loop
# ============================================================


class TestToolLoop:
    """Agent executes tool calls and feeds results back to LLM."""

    async def test_single_tool_call_then_response(self, fake_llm, simple_agent):
        """LLM calls a tool, agent executes it, LLM responds based on result."""
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[tc("t1", "read_file", {"path": "conftest.py"})]),
            FakeLLMResponse(content="The file says: hello world"),
        )

        result = await simple_agent.chat([{"role": "user", "content": "read /test.txt"}])

        assert "hello world" in result.lower()
        assert fake_llm.call_count() == 2

        # Verify tool result was fed to second LLM call
        msgs = fake_llm.last_messages
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        # Tool result should contain file content or error
        assert len(tool_msgs) >= 1

    async def test_multiple_tool_calls_in_one_turn(self, fake_llm, simple_agent):
        """LLM makes multiple tool calls at once, agent executes all."""
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[
                tc("t1", "read_file", {"path": "/a.txt"}),
                tc("t2", "read_file", {"path": "/b.txt"}),
            ]),
            FakeLLMResponse(content="Got both files"),
        )

        result = await simple_agent.chat([{"role": "user", "content": "read files"}])

        assert "Got both files" == result
        assert fake_llm.call_count() == 2

    async def test_tool_error_recovery(self, fake_llm, simple_agent):
        """When a tool fails, error is returned as JSON and LLM can recover."""
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[tc("t1", "unknown_tool_xyz", {})]),
            FakeLLMResponse(content="I tried an unknown tool, sorry"),
        )

        result = await simple_agent.chat([{"role": "user", "content": "try bad tool"}])

        # First LLM call had tool_calls, second call gets tool error + responds
        assert fake_llm.call_count() == 2
        # Check that tool error message was fed back
        msgs = fake_llm.last_messages
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "error" in tool_msgs[0]["content"].lower()

    async def test_max_tool_turns_limit(self, fake_llm, simple_agent):
        """Agent stops after max_tool_turns tool calls even if LLM keeps calling tools."""
        # Program 100 tool calls (more than max_tool_turns=50)
        for _ in range(100):
            fake_llm._responses.append(
                FakeLLMResponse(tool_calls=[tc(f"t{i}", "read_file", {"path": "/x"}) for i in range(1)])
            )

        # Create agent with very low max_tool_turns
        from tests.conftest import make_test_agent
        agent = make_test_agent(fake_llm, max_tool_turns=3)

        result = await agent.chat([{"role": "user", "content": "do stuff"}])

        # Should stop after 3 tool turns, returning the content (which is None here)
        # The last _run_turn returns the content from the final API call
        assert fake_llm.call_count() <= 4  # 1 initial + up to 3 tool turns
        await agent.close()


# ============================================================
# Actor model
# ============================================================


class TestActorModel:
    """Agent permanent loop: start → send_message → produce output → stop."""

    async def test_start_send_stop(self, fake_llm, tmp_path):
        """Basic actor model lifecycle."""
        from tests.conftest import make_test_agent
        agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

        fake_llm.respond("Loop response")

        await agent.start()
        await agent.send_message("hello")
        output = await asyncio.wait_for(agent._output_queue.get(), timeout=2)
        assert output.text == "Loop response"

        await agent.stop()
        assert agent._running is False

    async def test_multiple_messages(self, fake_llm, tmp_path):
        """Agent handles multiple messages in sequence."""
        from tests.conftest import make_test_agent
        agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

        fake_llm.chain(
            FakeLLMResponse(content="First"),
            FakeLLMResponse(content="Second"),
            FakeLLMResponse(content="Third"),
        )

        await agent.start()

        await agent.send_message("msg 1")
        out1 = await asyncio.wait_for(agent._output_queue.get(), timeout=2)
        assert out1.text == "First"

        await agent.send_message("msg 2")
        out2 = await asyncio.wait_for(agent._output_queue.get(), timeout=2)
        assert out2.text == "Second"

        await agent.send_message("msg 3")
        out3 = await asyncio.wait_for(agent._output_queue.get(), timeout=2)
        assert out3.text == "Third"

        # Messages accumulated
        assert len(agent._messages) >= 6  # 3 user + 3 assistant

        await agent.stop()

    async def test_reply_target_preserved(self, fake_llm, tmp_path):
        """ReplyTarget from send_message is attached to AgentOutput."""
        from tests.conftest import make_test_agent
        from tyagent.types import ReplyTarget
        agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

        fake_llm.respond("Targeted reply")
        rt = ReplyTarget(platform="feishu", chat_id="chat99", message_id="msg88")

        await agent.start()
        await agent.send_message("hi", reply_target=rt)
        output = await asyncio.wait_for(agent._output_queue.get(), timeout=2)
        assert output.reply_target is rt
        assert output.reply_target.chat_id == "chat99"

        await agent.stop()

    async def test_send_without_start_raises(self, tmp_path):
        """send_message raises RuntimeError if start() was not called."""
        from tests.conftest import make_test_agent
        agent = make_test_agent(FakeLLM(), home_dir=tmp_path / "home")

        with pytest.raises(RuntimeError):
            await agent.send_message("hi")

    async def test_double_stop_is_safe(self, fake_llm, tmp_path):
        """Calling stop() twice doesn't crash."""
        from tests.conftest import make_test_agent
        agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

        fake_llm.respond("ok")
        await agent.start()
        await agent.send_message("hi")
        await asyncio.wait_for(agent._output_queue.get(), timeout=2)
        await agent.stop()
        await agent.stop()  # second stop should be no-op
        assert agent._running is False

    async def test_empty_message_skipped(self, fake_llm, tmp_path):
        """Empty text in send_message doesn't trigger a turn."""
        from tests.conftest import make_test_agent
        agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

        fake_llm.respond("Should not be sent")
        await agent.start()
        # Send empty message — should not trigger turn
        await agent.send_message("")
        await agent.stop()

        # No API calls should have been made
        assert fake_llm.call_count() == 0


# ============================================================
# Streaming
# ============================================================


class TestStreaming:
    """Streaming API: delta callbacks and full content aggregation."""

    async def test_stream_deltas_received(self, fake_llm, simple_agent):
        """Stream delta callback receives progressive chunks."""
        fake_llm.respond("Progressive streaming content")

        deltas = []
        result = await simple_agent.chat(
            [{"role": "user", "content": "stream please"}],
            stream=True,
            stream_delta_callback=lambda d: deltas.append(d),
        )

        assert "Progressive streaming content" == result
        assert len(deltas) > 0
        assert "".join(deltas) == "Progressive streaming content"

    async def test_stream_reasoning_callback(self, fake_llm, simple_agent):
        """Reasoning callback receives model's chain-of-thought."""
        fake_llm._responses.append(FakeLLMResponse(
            content="Answer",
            reasoning_content="I need to think about this carefully.",
        ))

        reasoning_parts = []
        result = await simple_agent.chat(
            [{"role": "user", "content": "complex question"}],
            stream=True,
            reasoning_callback=lambda r: reasoning_parts.append(r),
        )

        assert result == "Answer"
        assert len(reasoning_parts) > 0


# ============================================================
# Error handling
# ============================================================


class TestErrors:
    """Agent error handling and recovery."""

    async def test_http_error_raises_agent_error(self, fake_llm, simple_agent):
        """HTTP error from LLM raises AgentError."""
        fake_llm.respond("__ERROR__:429:too many requests")

        with pytest.raises(AgentError) as exc_info:
            await simple_agent.chat([{"role": "user", "content": "hi"}])

        assert "429" in str(exc_info.value)

    async def test_agent_error_in_loop_produces_error_output(self, fake_llm, tmp_path):
        """In actor model, LLM errors produce error output (not crash)."""
        from tests.conftest import make_test_agent
        agent = make_test_agent(fake_llm, home_dir=tmp_path / "home")

        fake_llm.respond("__ERROR__:500:internal server error")

        await agent.start()
        await agent.send_message("hi")
        output = await asyncio.wait_for(agent._output_queue.get(), timeout=2)

        assert "error" in output.text.lower() or "500" in output.text
        await agent.stop()


# ============================================================
# Helpers
# ============================================================




def tc(call_id: str, name: str, args: dict) -> dict:
    """Build a tool call dict."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
