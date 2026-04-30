"""Unit tests for tyagent.context — single-pass token-based compression."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from tyagent.context import compress_context, _SUMMARIZE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockResponse:
    """Minimal mock for httpx.Response."""

    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("mock error", request=None, response=self)

    def json(self) -> dict:
        return self._json_data


def _build_mock_client(json_data: dict) -> AsyncMock:
    mock_resp = _MockResponse(200, json_data)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_resp)
    return client


def _build_failing_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
        "mock error", request=None, response=_MockResponse(500, {}),
    ))
    return client


# ---------------------------------------------------------------------------
# compress_context
# ---------------------------------------------------------------------------


class TestCompressContext:
    @pytest.mark.asyncio
    async def test_few_messages_returns_none(self):
        """< 3 messages is too short."""
        result = await compress_context(
            [{"role": "user", "content": "hi"}],
            http_client=None, model="m", api_key="k", base_url="https://api.t",
            token_history=[], context_window=128000, cut_ratio=0.5,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_token_history_returns_none(self):
        """No token history means no cut point."""
        result = await compress_context(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            http_client=None, model="m", api_key="k", base_url="https://api.t",
            token_history=[], context_window=128000, cut_ratio=0.5,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_basic_compression(self):
        """Cut at 50%, summarize pre-cut, keep tail."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "User asked questions. Assistant answered."}}]
        })

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "answer 2"},
            {"role": "user", "content": "question 3"},
            {"role": "assistant", "content": "answer 3"},
        ]

        # Simulate token history from prior API calls.
        # After message 4, we had ~2000 tokens (below 50% of 128K).
        # After message 6, we had ~65000 tokens (above 50%).
        # The cut should land at message 4 (after "answer 2", a complete reply).
        token_history = [(4, 2000), (6, 65000)]

        result = await compress_context(
            messages, client,
            model="test-model", api_key="test-key",
            base_url="https://api.test.com/v1",
            token_history=token_history,
            context_window=128000,
            cut_ratio=0.5,
        )

        assert result is not None
        # Structure: system + summary system msg + tail (from aligned cut)
        assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result[1]["role"] == "system"
        assert "Summary" in result[1]["content"]
        # cut at msg 5 (after "answer 2", index 5 is "user: question 3")
        # aligned should be index 4+1=5
        assert len(result) >= 3  # system + summary + at least one tail msg

    @pytest.mark.asyncio
    async def test_cut_aligned_to_complete_reply(self):
        """If cut falls mid-tool-chain, align backward to complete reply boundary."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "Early conversation summarized."}}]
        })

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},  # complete reply at idx 2
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "function": {"name": "read_file", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "file content", "tool_call_id": "t1"},
            {"role": "assistant", "content": "a2"},  # complete reply at idx 6
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]

        # token history: after msg 4 (user "u2") we have 2K tokens (below target)
        # after msg 7 (assistant "a2") we have 65K tokens (above target)
        # cut_idx=7, then align backward to index 6+1=7 (a2 is complete reply)
        token_history = [(4, 2000), (7, 65000)]

        result = await compress_context(
            messages, client,
            model="m", api_key="k", base_url="https://api.t",
            token_history=token_history,
            context_window=128000, cut_ratio=0.5,
        )

        assert result is not None
        # tail should start with "u3" (msg 7, idx 7 since aligned=i+1=7)
        assert result[-2] == {"role": "user", "content": "u3"}
        assert result[-1] == {"role": "assistant", "content": "a3"}

    @pytest.mark.asyncio
    async def test_preserves_system_messages(self):
        """System prompts are kept verbatim, not summarized."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "Summary goes here."}}]
        })

        messages = [
            {"role": "system", "content": "Instruction A."},
            {"role": "system", "content": "Instruction B."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

        token_history = [(4, 2000), (6, 65000)]

        result = await compress_context(
            messages, client,
            model="m", api_key="k", base_url="https://api.t",
            token_history=token_history,
            context_window=128000, cut_ratio=0.5,
        )

        assert result is not None
        assert result[0] == {"role": "system", "content": "Instruction A."}
        assert result[1] == {"role": "system", "content": "Instruction B."}
        assert result[2]["role"] == "system"
        assert "Summary" in result[2]["content"]

    @pytest.mark.asyncio
    async def test_api_failure_returns_none(self):
        """LLM call failure is handled gracefully."""
        client = _build_failing_client()

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

        token_history = [(3, 1000), (5, 65000)]

        result = await compress_context(
            messages, client,
            model="m", api_key="k", base_url="https://api.t",
            token_history=token_history,
            context_window=128000, cut_ratio=0.5,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_summary_returns_none(self):
        """Empty LLM response should be treated as failure."""
        client = _build_mock_client({
            "choices": [{"message": {"content": ""}}]
        })

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

        token_history = [(3, 1000), (5, 65000)]

        result = await compress_context(
            messages, client,
            model="m", api_key="k", base_url="https://api.t",
            token_history=token_history,
            context_window=128000, cut_ratio=0.5,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_all_entries_above_target_uses_earliest(self):
        """When every token_history entry exceeds target, use the earliest as cut."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "Summary."}}]
        })

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

        # All entries above 50% of 128K (64000)
        token_history = [(3, 100000), (5, 150000)]

        result = await compress_context(
            messages, client,
            model="m", api_key="k", base_url="https://api.t",
            token_history=token_history,
            context_window=128000, cut_ratio=0.5,
        )

        assert result is not None
        # Should use msg_count=3 as fallback, then align
        # msg_count=3 → index 3 is "assistant: a1" (complete reply) → aligned=4
        # tail starts at index 4: "user: q2", "assistant: a2"
        assert result[0] == {"role": "system", "content": "sys"}
        assert "Summary" in result[1]["content"]
        # tail messages preserved
        assert result[2] == {"role": "user", "content": "q2"}
        assert result[3] == {"role": "assistant", "content": "a2"}

    @pytest.mark.asyncio
    async def test_uses_compression_model(self):
        """Verifies the LLM is called with the specified model and prompt."""
        client = _build_mock_client({
            "choices": [{"message": {"content": "Summary."}}]
        })

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "current"},
            {"role": "assistant", "content": "reply"},
        ]

        token_history = [(3, 1000), (5, 65000)]

        await compress_context(
            messages, client,
            model="cheap-model", api_key="compress-key",
            base_url="https://compress.api/v1",
            token_history=token_history,
            context_window=128000, cut_ratio=0.5,
        )

        call_kwargs = client.post.call_args[1]
        assert call_kwargs["json"]["model"] == "cheap-model"
        assert "max_tokens" not in call_kwargs["json"]  # let LLM decide length
        assert call_kwargs["json"]["temperature"] == 0.3
        assert call_kwargs["json"]["messages"][0]["content"] == _SUMMARIZE_SYSTEM_PROMPT
        assert "Authorization" in call_kwargs["headers"]
