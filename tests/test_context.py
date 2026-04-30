"""Unit tests for tyagent.context — deterministic tool-message-dropping compression."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tyagent.context import build_api_messages, summarize_middle


# ---------------------------------------------------------------------------
# build_api_messages — deterministic tool-message-dropping
# ---------------------------------------------------------------------------


class TestBuildApiMessages:
    def test_short_messages_no_tool_to_drop(self):
        """Short messages without old tool results are returned as-is."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = build_api_messages(msgs)
        assert result == msgs  # same content, nothing to drop

    def test_empty_messages(self):
        assert build_api_messages([]) == []

    def test_drops_tool_messages_before_last_user(self):
        """Tool messages before the last user message should be dropped."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "thinking...",
             "tool_calls": [{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "file content", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Here's what I found."},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "Answer."},
        ]
        result = build_api_messages(msgs)

        # system + first user kept
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "first question"

        # assistant text kept, tool_calls dropped
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "thinking..."
        assert "tool_calls" not in result[2]

        # tool message skipped entirely
        assert result[3]["role"] == "assistant"
        assert result[3]["content"] == "Here's what I found."

        # second user + its assistant kept
        assert result[4]["role"] == "user"
        assert result[4]["content"] == "second question"
        assert result[5]["role"] == "assistant"
        assert result[5]["content"] == "Answer."

        assert len(result) == 6  # originally 7, dropped 1 tool msg

    def test_keeps_tool_chain_after_last_user(self):
        """Tool messages after the last user should be preserved intact."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "content", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Done."},
        ]
        result = build_api_messages(msgs)

        assert len(result) == 5  # all preserved — last user is at idx 1
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]  # tool_calls preserved
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"

    def test_no_user_message(self):
        """If no user message found, return original."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hello?"},
        ]
        result = build_api_messages(msgs)
        assert result is msgs

    def test_all_roles_kept_except_tool_before_last_user(self):
        """Only tool messages before last user are dropped; everything else kept."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1",
             "tool_calls": [{"id": "t1", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "content": "r1", "tool_call_id": "t1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a3",
             "tool_calls": [{"id": "t2", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "content": "r2", "tool_call_id": "t2"},
            {"role": "assistant", "content": "a4"},
        ]
        result = build_api_messages(msgs)

        # Messages kept: sys, u1, a1(no tc), a2, u2, a3(with tc), tool(r2), a4
        kept_roles = [m["role"] for m in result]
        assert kept_roles == ["system", "user", "assistant", "assistant", "user",
                               "assistant", "tool", "assistant"]

        # Old assistant (idx 2) has tool_calls stripped
        assert "tool_calls" not in result[2]
        # New assistant (idx 5) still has tool_calls
        assert "tool_calls" in result[5]

    def test_reasoning_content_preserved(self):
        """reasoning_content should be preserved for old assistant messages."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 100_000},
            {"role": "assistant", "content": "answer", "reasoning_content": "thinking..."},
            {"role": "user", "content": "y" * 100_000},
        ]
        result = build_api_messages(msgs)

        assert result[2]["reasoning_content"] == "thinking..."

    def test_original_not_modified(self):
        """Compression should not modify the original messages list."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 100_000},
            {"role": "assistant", "content": "a1",
             "tool_calls": [{"id": "t1", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "content": "r1", "tool_call_id": "t1"},
            {"role": "user", "content": "y" * 100_000},
        ]
        original_len = len(msgs)
        build_api_messages(msgs)
        assert len(msgs) == original_len
        # tool_calls should still be there
        assert "tool_calls" in msgs[2]

    def test_compression_reduces_count(self):
        """Compression should reduce message count when tool msgs are present."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}" + "x" * 10_000})
            msgs.append({"role": "assistant", "content": None,
                          "tool_calls": [{"id": f"t{i}", "function": {"name": "func", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "content": f"r{i}", "tool_call_id": f"t{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        result = build_api_messages(msgs)
        assert len(result) < len(msgs)

    def test_tool_chain_after_last_user_preserved(self):
        """When all tool calls are after the last user, nothing is dropped."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "done"},
        ]
        result = build_api_messages(msgs)
        assert result == msgs


# ---------------------------------------------------------------------------
# summarize_middle — LLM-based context summarization (compression)
# ---------------------------------------------------------------------------


class _MockResponse:
    """Minimal mock for httpx.Response used by summarize_middle tests."""

    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("mock error", request=None, response=self)

    def json(self) -> dict:
        return self._json_data


def _build_mock_client(json_data: dict) -> AsyncMock:
    """Build a mock httpx.AsyncClient that returns a canned JSON response."""
    mock_resp = _MockResponse(200, json_data)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_resp)
    return client


def _build_failing_client() -> AsyncMock:
    """Build a mock client that raises HTTPStatusError."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
        "mock error", request=None, response=_MockResponse(500, {}),
    ))
    return client


class TestSummarizeMiddle:
    @pytest.mark.asyncio
    async def test_short_conversation_returns_none(self):
        """< 3 messages is too short for level-2 compression."""
        result = await summarize_middle(
            [{"role": "user", "content": "hi"}],
            http_client=None, model="m", api_key="k",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_user_message_returns_none(self):
        """Without a user message, nothing to anchor the tail."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = await summarize_middle(
            msgs, http_client=None, model="m", api_key="k",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_system_only_before_user_returns_none(self):
        """If the only pre-user content is a system message, no summarization needed."""
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ]
        result = await summarize_middle(
            msgs, http_client=None, model="m", api_key="k",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_replaces_pre_user_with_summary(self):
        """Successful summarization replaces pre-user non-system messages with a summary."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "User asked about X. System replied Y."}}]
        })

        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current query"},
        ]

        result = await summarize_middle(
            msgs, client,
            model="test-model", api_key="test-key",
            base_url="https://api.test.com/v1",
        )

        assert result is not None
        assert len(result) == 3  # system + summary + last user
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["role"] == "system"
        assert "Summary" in result[1]["content"]
        assert "User asked about X" in result[1]["content"]
        assert result[2] == {"role": "user", "content": "current query"}

        call_kwargs = client.post.call_args[1]
        assert call_kwargs["json"]["model"] == "test-model"
        assert call_kwargs["json"]["max_tokens"] == 512
        assert call_kwargs["json"]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_preserves_system_prompts_separately(self):
        """System prompts before the last user are preserved, not summarized."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "User made a query."}}]
        })

        msgs = [
            {"role": "system", "content": "Primary instructions."},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "system", "content": "Extra context."},
            {"role": "user", "content": "final q"},
        ]

        result = await summarize_middle(
            msgs, client, model="m", api_key="k", base_url="https://api.test.com/v1"
        )

        assert result is not None
        assert result[0] == {"role": "system", "content": "Primary instructions."}
        assert result[1] == {"role": "system", "content": "Extra context."}
        assert result[2]["role"] == "system"
        assert "Summary" in result[2]["content"]
        assert result[3] == {"role": "user", "content": "final q"}

    @pytest.mark.asyncio
    async def test_api_failure_returns_none(self):
        """When the LLM API call fails, None is returned gracefully."""
        client = _build_failing_client()

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new"},
        ]

        result = await summarize_middle(
            msgs, client, model="m", api_key="k", base_url="https://api.test.com/v1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_summary_returns_none(self):
        """If the LLM returns an empty summary, return None."""
        client = _build_mock_client({
            "choices": [{"message": {"content": ""}}]
        })

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new"},
        ]

        result = await summarize_middle(
            msgs, client, model="m", api_key="k", base_url="https://api.test.com/v1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_chain_after_last_user_preserved(self):
        """Active tool chain after the last user is preserved after summarization."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "Old tool calls summarized."}}]
        })

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old q"},
            {"role": "tool", "content": "old result", "tool_call_id": "t1"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new q"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "t2", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "new result", "tool_call_id": "t2"},
            {"role": "assistant", "content": "done"},
        ]

        result = await summarize_middle(
            msgs, client, model="m", api_key="k", base_url="https://api.test.com/v1"
        )

        assert result is not None
        assert result[0] == {"role": "system", "content": "sys"}
        assert "Summary" in result[1]["content"]
        tail_start = 2
        assert result[tail_start] == {"role": "user", "content": "new q"}
        assert "tool_calls" in result[tail_start + 1]
        assert result[tail_start + 2]["role"] == "tool"
        assert result[tail_start + 3]["role"] == "assistant"
        assert len(result) == 6  # system + summary + 4 tail messages
