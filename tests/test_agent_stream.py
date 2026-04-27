"""Unit tests for TyAgent.chat() streaming support."""

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tyagent.agent import AgentError, TyAgent


# ---------------------------------------------------------------------------
# Helpers: build mock SSE responses for httpx AsyncClient.stream()
# ---------------------------------------------------------------------------


def _mock_async_stream(chunks: List[str], status_code: int = 200):
    """
    Create a mock HTTP response object for httpx AsyncClient.stream().
    Returns an async context manager that yields a mock response with
    aiter_lines() as an async method returning an async generator.
    """
    lines = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            lines.append(line)

    async def _line_gen():
        for line in lines:
            yield line

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = _line_gen

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    return mock_ctx


def _make_sse_chunks(text_chunks: List[str], tool_calls: List[Dict] = None,
                     reasoning_chunks: List[str] = None,
                     usage: Dict[str, int] = None) -> List[str]:
    """Build SSE event strings from text/tool/reasoning chunks.

    Returns a list of raw SSE strings like "data: {json}\n\n".
    Each chunk in the returned list is one SSE event.
    """
    events: List[str] = []
    # Reasoning before content, if any
    if reasoning_chunks:
        for rc in reasoning_chunks:
            delta = {"reasoning_content": rc}
            events.append(f"data: {json.dumps({'choices': [{'delta': delta, 'index': 0}]})}\n\n")
    # Text content
    for tc in text_chunks:
        delta = {"content": tc}
        events.append(f"data: {json.dumps({'choices': [{'delta': delta, 'index': 0}]})}\n\n")
    # Tool calls
    if tool_calls:
        for tc_delta in tool_calls:
            delta = {"tool_calls": [tc_delta]}
            events.append(f"data: {json.dumps({'choices': [{'delta': delta, 'index': 0}]})}\n\n")
    # Usage (no choices, just usage)
    if usage:
        events.append(f"data: {json.dumps({'usage': usage})}\n\n")
    # DONE marker
    events.append("data: [DONE]\n\n")
    return events


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return TyAgent(api_key="test-key", base_url="https://api.test/v1")


# ---------------------------------------------------------------------------
# Basic streaming test
# ---------------------------------------------------------------------------


class TestChatStreamBasic:
    @pytest.mark.asyncio
    async def test_stream_text_only(self, agent):
        """Streaming a single text response works."""
        sse_chunks = _make_sse_chunks(
            text_chunks=["Hello", " ", "world", "!"],
            usage={"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "say hi"}]
            result = await agent.chat(messages, stream=True)

        assert result == "Hello world!"
        assert len(messages) == 3  # system + user + assistant
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hello world!"
        assert agent.last_usage == {
            "prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22,
        }

    @pytest.mark.asyncio
    async def test_stream_delta_callback_called(self, agent):
        """stream_delta_callback is called for each text chunk."""
        captured: list = []

        def cb(chunk):
            captured.append(chunk)

        sse_chunks = _make_sse_chunks(
            text_chunks=["Hello", " ", "world", "!"],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "say hi"}]
            await agent.chat(messages, stream=True, stream_delta_callback=cb)

        assert captured == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_reasoning_callback_called(self, agent):
        """reasoning_callback is called for reasoning_content chunks."""
        captured: list = []

        def cb(chunk):
            captured.append(chunk)

        sse_chunks = _make_sse_chunks(
            text_chunks=["Hello world!"],
            reasoning_chunks=["Thinking", " step", " by step..."],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "think"}]
            await agent.chat(messages, stream=True, reasoning_callback=cb)

        assert captured == ["Thinking", " step", " by step..."]

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, agent):
        """Streaming with tool calls works via accumulation."""
        sse_chunks = _make_sse_chunks(
            text_chunks=["Let me check..."],
            tool_calls=[
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path":'}},
                {"index": 0, "function": {"arguments": ' "/tmp/test.txt"}'}},
            ],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx), \
             patch("tyagent.tools.registry.registry") as mock_reg:
            mock_reg.dispatch.return_value = "hello"
            messages = [{"role": "user", "content": "read file"}]
            result = await agent.chat(
                messages, stream=True,
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        # After tool, the LLM should respond — but we only have one stream turn
        # Returns empty since no second API call for final text
        # Actually, the test agent only has the stream return, no follow-up
        # Let's check the assistant message was built correctly
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Let me check..."
        assert len(messages[2]["tool_calls"]) == 1
        assert messages[2]["tool_calls"][0]["function"]["name"] == "read_file"
        assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "/tmp/test.txt"}'

    @pytest.mark.asyncio
    async def test_on_segment_break_called(self, agent):
        """on_segment_break fires between assistant message and tool execution."""
        break_called = False

        def on_break():
            nonlocal break_called
            break_called = True

        sse_chunks = _make_sse_chunks(
            text_chunks=["Checking..."],
            tool_calls=[
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'}},
            ],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        # Need a second response (after tool execution) to finish the loop
        final_chunks = _make_sse_chunks(
            text_chunks=["Result: hello"],
            usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        )
        mock_ctx2 = _mock_async_stream(final_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx) as mock_stream, \
             patch("tyagent.tools.registry.registry") as mock_reg:
            mock_reg.dispatch.return_value = "hello"
            # After first stream, second call goes through stream too
            mock_stream.side_effect = [mock_ctx, mock_ctx2]

            messages = [{"role": "user", "content": "read file"}]
            result = await agent.chat(
                messages, stream=True,
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                on_segment_break=on_break,
            )

        assert break_called, "on_segment_break should have been called"
        assert result == "Result: hello"

    @pytest.mark.asyncio
    async def test_stream_no_tool_calls_simple(self, agent):
        """Streaming without tool calls returns content directly."""
        sse_chunks = _make_sse_chunks(
            text_chunks=["Hello world!"],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "say hi"}]
            result = await agent.chat(messages, stream=True)

        assert result == "Hello world!"


# ---------------------------------------------------------------------------
# Backward compatibility: stream=False still works
# ---------------------------------------------------------------------------


class TestBackwardCompatible:
    @pytest.mark.asyncio
    async def test_non_stream_still_works(self, agent):
        """Setting stream=False (default) uses the old code path."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [{"role": "user", "content": "hi"}]
            result = await agent.chat(messages, stream=False)

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_non_stream_tool_loop_still_works(self, agent):
        """Non-streaming tool loop is unchanged."""
        tool_resp = MagicMock()
        tool_resp.status_code = 200
        tool_resp.raise_for_status = MagicMock()
        tool_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/test.txt"}',
                        },
                    }],
                }
            }]
        }

        final_resp = MagicMock()
        final_resp.status_code = 200
        final_resp.raise_for_status = MagicMock()
        final_resp.json.return_value = {
            "choices": [{"message": {"content": "File contents: hello"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [tool_resp, final_resp]
            with patch("tyagent.tools.registry.registry") as mock_reg:
                mock_reg.dispatch.return_value = "hello"
                messages = [{"role": "user", "content": "read the file"}]
                result = await agent.chat(
                    messages,
                    stream=False,
                    tools=[{"type": "function", "function": {"name": "read_file"}}],
                )

        assert result == "File contents: hello"
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_non_stream_with_on_message(self, agent):
        """on_message callback still works in non-stream mode."""
        captured = []

        def cb(role, content, **kwargs):
            captured.append((role, content, kwargs))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello", "reasoning_content": "Thinking..."}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [{"role": "user", "content": "hi"}]
            result = await agent.chat(messages, stream=False, on_message=cb)

        assert result == "Hello"
        assert len(captured) == 1
        assert captured[0][0] == "assistant"
        assert captured[0][1] == "Hello"
        assert captured[0][2] == {"reasoning": "Thinking..."}


# ---------------------------------------------------------------------------
# Error handling in streaming
# ---------------------------------------------------------------------------


class TestChatStreamErrors:
    @pytest.mark.asyncio
    async def test_stream_http_error(self, agent):
        """HTTP error during streaming raises AgentError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=mock_resp
        )
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "hi"}]
            with pytest.raises(AgentError, match="429"):
                await agent.chat(messages, stream=True)

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, agent):
        """Connection error during streaming raises AgentError."""
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "hi"}]
            with pytest.raises(AgentError, match="ConnectError"):
                await agent.chat(messages, stream=True)


# ---------------------------------------------------------------------------
# on_message callback with streaming
# ---------------------------------------------------------------------------


class TestChatStreamOnMessage:
    @pytest.mark.asyncio
    async def test_stream_on_message_called(self, agent):
        """on_message callback is called in streaming mode."""
        captured = []

        def cb(role, content, **kwargs):
            captured.append((role, content, kwargs))

        sse_chunks = _make_sse_chunks(
            text_chunks=["Hello ", "world", "!"],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "say hi"}]
            result = await agent.chat(messages, stream=True, on_message=cb)

        assert result == "Hello world!"
        assert len(captured) == 1
        assert captured[0][0] == "assistant"
        assert captured[0][1] == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_on_message_with_reasoning(self, agent):
        """on_message receives reasoning in streaming mode."""
        captured = []

        def cb(role, content, **kwargs):
            captured.append((role, content, kwargs))

        sse_chunks = _make_sse_chunks(
            text_chunks=["Hello!"],
            reasoning_chunks=["Let me think..."],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "hi"}]
            result = await agent.chat(messages, stream=True, on_message=cb)

        assert result == "Hello!"
        assert len(captured) == 1
        assert captured[0][2] == {"reasoning": "Let me think..."}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestChatStreamEdgeCases:
    @pytest.mark.asyncio
    async def test_stream_empty_content(self, agent):
        """Streaming with empty content returns empty string."""
        sse_chunks = _make_sse_chunks(text_chunks=[])
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "say nothing"}]
            result = await agent.chat(messages, stream=True)

        assert result == ""

    @pytest.mark.asyncio
    async def test_stream_only_reasoning(self, agent):
        """When only reasoning_content is returned (no text), return it."""
        sse_chunks = _make_sse_chunks(
            text_chunks=[],
            reasoning_chunks=["Just thinking out loud..."],
        )
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "think"}]
            result = await agent.chat(messages, stream=True)

        assert result == "Just thinking out loud..."

    @pytest.mark.asyncio
    async def test_stream_delta_callback_none_does_not_crash(self, agent):
        """Calling stream=True without callbacks works fine."""
        sse_chunks = _make_sse_chunks(text_chunks=["Hello"])
        mock_ctx = _mock_async_stream(sse_chunks)

        with patch.object(agent._client, "stream", return_value=mock_ctx):
            messages = [{"role": "user", "content": "hi"}]
            result = await agent.chat(messages, stream=True)

        assert result == "Hello"
