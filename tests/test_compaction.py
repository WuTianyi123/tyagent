"""Unit tests for tyagent.compaction (Codex-style compaction module).

Tests the core logic in isolation:
- is_summary_message
- collect_user_messages
- select_tail_messages
- build_compacted_history
- total_token_estimate
- run_compact (with mock HTTP client)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tyagent.compaction import (
    COMPACT_USER_MESSAGE_MAX_TOKENS,
    SUMMARY_PREFIX,
    build_compacted_history,
    collect_user_messages,
    is_summary_message,
    run_compact,
    select_tail_messages,
    total_token_estimate,
)


# ── is_summary_message ───────────────────────────────────────────────────────


class TestIsSummaryMessage:
    def test_prefix_match(self):
        assert is_summary_message(SUMMARY_PREFIX + "\nsummary text")

    def test_no_match(self):
        assert not is_summary_message("regular user message")

    def test_empty(self):
        assert not is_summary_message("")

    def test_prefix_without_newline(self):
        # Starts with prefix but no newline—still matches
        assert is_summary_message(SUMMARY_PREFIX + " (no newline)")


# ── collect_user_messages ────────────────────────────────────────────────────


class TestCollectUserMessages:
    def test_basic(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "another"},
        ]
        assert collect_user_messages(msgs) == ["hello", "another"]

    def test_skips_summaries(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nold summary"},
            {"role": "user", "content": "second"},
        ]
        assert collect_user_messages(msgs) == ["first", "second"]

    def test_skips_non_user(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "reply"},
        ]
        assert collect_user_messages(msgs) == []

    def test_multimodal_content(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "..."}}]},
            {"role": "assistant", "content": "reply"},
        ]
        result = collect_user_messages(msgs)
        assert len(result) == 1
        assert "hello" in result[0]

    def test_empty_list(self):
        assert collect_user_messages([]) == []

    def test_non_string_content(self):
        msgs = [
            {"role": "user", "content": 123},
        ]
        assert collect_user_messages(msgs) == []


# ── select_tail_messages ─────────────────────────────────────────────────────


class TestSelectTailMessages:
    def test_all_fit(self):
        msgs = ["short"]
        result = select_tail_messages(msgs, max_tokens=100)
        assert result == ["short"]

    def test_truncates_last(self):
        # A 60-byte message at 4 bytes/token -> ~15 tokens, exceeds budget=10.
        # The message is dropped entirely since it doesn't fit — no truncation.
        long_msg = "a" * 60
        msgs = ["first", long_msg]
        result = select_tail_messages(msgs, max_tokens=10)
        # Reverse iteration: long_msg doesn't fit → dropped, "first" not reached
        assert result == []

    def test_reverse_order(self):
        msgs = ["oldest", "middle", "newest"]
        result = select_tail_messages(msgs, max_tokens=100)
        assert result == ["oldest", "middle", "newest"]  # same order

    def test_max_tokens_zero(self):
        assert select_tail_messages(["a"], max_tokens=0) == []

    def test_max_tokens_negative(self):
        assert select_tail_messages(["a"], max_tokens=-1) == []

    def test_empty_input(self):
        assert select_tail_messages([]) == []


# ── build_compacted_history ──────────────────────────────────────────────────


class TestBuildCompactedHistory:
    def test_structure(self):
        selected = ["msg1", "msg2"]
        summary = "task done\n- progress"
        result = build_compacted_history(selected, summary)

        assert len(result) == 3
        # User messages preserved
        assert result[0] == {"role": "user", "content": "msg1"}
        assert result[1] == {"role": "user", "content": "msg2"}
        # Summary injected as user message with prefix
        assert result[2]["role"] == "user"
        assert result[2]["content"].startswith(SUMMARY_PREFIX)
        assert "task done" in result[2]["content"]

    def test_empty_selected(self):
        result = build_compacted_history([], "summary")
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_summary(self):
        result = build_compacted_history(["msg"], "")
        assert len(result) == 2
        assert "\n" in result[1]["content"]


# ── total_token_estimate ─────────────────────────────────────────────────────


class TestTotalTokenEstimate:
    def test_ascii_messages(self):
        msgs = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]
        est = total_token_estimate(msgs)
        assert est >= 3  # (~12 chars / 4) + 2*10 overhead

    def test_cjk_messages(self):
        # CJK chars are 3 bytes each in UTF-8, so estimate is higher
        msgs = [
            {"role": "user", "content": "你好世界"},
        ]
        est = total_token_estimate(msgs)
        # 4 chars * 3 bytes = 12 bytes / 4 = 3 tokens + 10 overhead = 13
        assert est >= 3

    def test_includes_system_prompt(self):
        msgs = [{"role": "user", "content": "hi"}]
        est = total_token_estimate(msgs, system_prompt="You are a bot.")
        assert est >= 3

    def test_empty_messages(self):
        est = total_token_estimate([])
        assert est >= 0

    def test_tool_calls_estimated(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"arguments": '{"path": "/tmp/x"}'}}
            ]},
        ]
        est = total_token_estimate(msgs)
        assert est >= 10  # overhead


# ── run_compact (mock HTTP) ─────────────────────────────────────────────────


class TestRunCompact:
    @pytest.mark.asyncio
    async def test_successful_compaction(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Summary text"}}]
        }
        client.post.return_value = mock_resp

        messages = [{"role": "user", "content": "hello"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client,
        )

        assert result is not None
        assert len(result) >= 1
        assert result[-1]["role"] == "user"
        assert SUMMARY_PREFIX in result[-1]["content"]

    @pytest.mark.asyncio
    async def test_no_user_messages(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        result = await run_compact(
            [{"role": "assistant", "content": "hi"}],
            model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client,
        )
        assert result is None
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_on_api_error(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        # First two calls fail, third succeeds
        client.post.side_effect = [
            httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)),
            httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock(status_code=500)),
            MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Summary"}}]}
            ),
        ]

        messages = [{"role": "user", "content": "hello"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=2,
        )

        assert result is not None
        assert client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        client.post.return_value = mock_resp

        messages = [{"role": "user", "content": "hello"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=2,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_summary_response(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        client.post.return_value = mock_resp

        messages = [{"role": "user", "content": "hello"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_retry(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timeout")
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Summary"}}]}
            )

        client.post.side_effect = side_effect

        messages = [{"role": "user", "content": "hello"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=1,
        )
        assert result is not None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_url_strips_trailing_slash(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Summary"}}]
        }
        client.post.return_value = mock_resp

        messages = [{"role": "user", "content": "hello"}]
        await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1/",
            http_client=client,
        )
        # Should POST to /chat/completions, not //chat/completions
        call_url = client.post.call_args[0][0]
        assert "//chat" not in call_url
        assert call_url.endswith("/chat/completions")
# ── Context overflow recovery ───────────────────────────────────────────


class TestContextOverflow:
    """Tests for Codex-style context overflow recovery in run_compact."""

    @pytest.mark.asyncio
    async def test_context_overflow_removes_oldest_message(self):
        """HTTP 400 with 'context_length' removes oldest message and retries."""
        client = AsyncMock(spec=httpx.AsyncClient)

        # Track calls to verify shrinking
        posted_bodies = []

        async def side_effect(*args, **kwargs):
            body = kwargs["json"]["messages"][0]["content"]
            posted_bodies.append(len(body))
            if len(posted_bodies) == 1:
                # First call: context overflow
                return MagicMock(
                    status_code=400,
                    text="context_length exceeded, model cannot process",
                )
            # Second call: success
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Final summary"}}]},
            )

        client.post.side_effect = side_effect

        messages = [
            {"role": "user", "content": "oldest message"},
            {"role": "assistant", "content": "response to oldest"},
            {"role": "user", "content": "newest message"},
        ]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=2,
        )

        assert result is not None
        assert "Final summary" in result[-1]["content"] or SUMMARY_PREFIX in result[-1]["content"]
        # Second call should have shorter body (one message removed)
        assert posted_bodies[1] < posted_bodies[0]

    @pytest.mark.asyncio
    async def test_413_triggers_overflow_recovery(self):
        """HTTP 413 always triggers overflow recovery regardless of body content."""
        client = AsyncMock(spec=httpx.AsyncClient)

        async def side_effect(*args, **kwargs):
            if not hasattr(side_effect, "called"):
                side_effect.called = True
                return MagicMock(status_code=413, text="Request Entity Too Large")
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Summary"}}]},
            )

        client.post.side_effect = side_effect

        messages = [{"role": "user", "content": "msg1"}, {"role": "user", "content": "msg2"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_single_message_overflow_gives_up(self):
        """Context overflow with only 1 message left returns None."""
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock(
            status_code=400,
            text="context_length exceeded",
        )
        client.post.return_value = mock_resp

        messages = [{"role": "user", "content": "only message"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=2,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_overflows_progressive_shrink(self):
        """Multiple context overflows progressively remove messages."""
        client = AsyncMock(spec=httpx.AsyncClient)

        posted_bodies = []

        async def side_effect(*args, **kwargs):
            body = kwargs["json"]["messages"][0]["content"]
            posted_bodies.append(body)
            call = len(posted_bodies)
            if call <= 2:
                # First two calls overflow
                return MagicMock(status_code=400, text="context_length exceeded")
            # Third call succeeds
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Final summary"}}]},
            )

        client.post.side_effect = side_effect

        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
        ]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=2,
        )

        assert result is not None
        # Each overflow call should have shorter input
        assert len(posted_bodies) == 3
        assert len(posted_bodies[1]) < len(posted_bodies[0])
        assert len(posted_bodies[2]) < len(posted_bodies[1])

    @pytest.mark.asyncio
    async def test_non_overflow_400_uses_transient_retry_budget(self):
        """Non-overflow 400 errors consume transient retry budget, not shrink input."""
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock(status_code=400, text="Bad Request: invalid model")
        client.post.return_value = mock_resp

        messages = [{"role": "user", "content": "msg1"}, {"role": "user", "content": "msg2"}]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=1,
        )

        assert result is None
        # Should have called 2 times (initial + 1 retry), not shrunk
        assert client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_overflow_resets_transient_budget(self):
        """Context overflow should reset remaining_transient to max_retries."""
        client = AsyncMock(spec=httpx.AsyncClient)

        posted_bodies = []

        async def side_effect(*args, **kwargs):
            body = kwargs["json"]["messages"][0]["content"]
            posted_bodies.append(len(body))
            call = len(posted_bodies)
            if call == 1:
                # First call: overflow
                return MagicMock(status_code=400, text="context_length exceeded")
            if call == 2:
                # Second call: timeout (should still have full transient budget)
                raise httpx.TimeoutException("timeout")
            # Third call: success
            return MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Summary after reset"}}]},
            )

        client.post.side_effect = side_effect

        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "user", "content": "b" * 100},
            {"role": "user", "content": "c" * 100},
        ]
        result = await run_compact(
            messages, model="test-model",
            api_key="key", base_url="https://api.test/v1",
            http_client=client, max_retries=1,
        )

        assert result is not None
        # Overflow (call 1) reset budget from 1 to 1
        # Timeout (call 2) consumed the 1 transient retry → remaining_transient=0 → but slept → continue
        # Call 3: success
        # Total: 3 calls (overflow + retry after timeout + success)
        assert client.post.call_count == 3
