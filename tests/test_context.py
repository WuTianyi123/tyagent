"""Unit tests for ty_agent.context — context compression."""

import pytest

from ty_agent.context import (
    _DEFAULT_MAX_CHARS,
    compress_messages,
    estimate_tokens,
    should_compress,
    _summarize_middle,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        tokens = estimate_tokens(msgs)
        assert tokens > 0
        assert tokens < 100  # "Hello world" is very short

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 2000},
        ]
        tokens = estimate_tokens(msgs)
        assert tokens > 500  # ~3000 chars / 3.5

    def test_tool_calls_counted(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"path": "/very/long/path/to/file.txt"}',
                        }
                    }
                ],
            }
        ]
        tokens = estimate_tokens(msgs)
        assert tokens > 0

    def test_none_content(self):
        msgs = [{"role": "assistant", "content": None}]
        assert estimate_tokens(msgs) == 0


# ---------------------------------------------------------------------------
# should_compress
# ---------------------------------------------------------------------------


class TestShouldCompress:
    def test_short_messages_no_compress(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        assert should_compress(msgs, max_tokens=100000) is False

    def test_long_messages_should_compress(self):
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "x" * 200_000},
        ]
        assert should_compress(msgs, max_chars=100_000) is True

    def test_token_budget(self):
        # ~30k chars = ~8.5k tokens
        msgs = [{"role": "user", "content": "a" * 30_000}]
        assert should_compress(msgs, max_tokens=5000) is True
        assert should_compress(msgs, max_tokens=20000) is False


# ---------------------------------------------------------------------------
# compress_messages
# ---------------------------------------------------------------------------


class TestCompressMessages:
    def test_short_messages_not_compressed(self):
        """Messages within budget should be returned as-is (same object)."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = compress_messages(msgs, max_chars=1_000_000)
        assert result is msgs  # same object, not copied

    def test_few_messages_not_compressed(self):
        """Very few messages should not be compressed even if total chars exceed."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 200_000},
        ]
        result = compress_messages(msgs, max_chars=100)
        # Only 2 messages, not enough for head + middle + tail
        assert result is msgs

    def test_compressed_output_structure(self):
        """Compressed output should have head + summary + tail."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Message {i}: " + "x" * 5000})

        result = compress_messages(msgs, max_chars=10_000, head_size=2, tail_ratio=0.2)

        # Should have fewer messages than original
        assert len(result) < len(msgs)
        # First messages should be preserved
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]
        # Last messages should be preserved
        assert result[-1] is msgs[-1]
        # There should be a summary message in between
        summary_msgs = [m for m in result if m.get("_compressed")]
        assert len(summary_msgs) == 1
        assert "Context Summary" in summary_msgs[0]["content"]

    def test_original_not_modified(self):
        """Compression should not modify the original messages list."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Message {i}: " + "y" * 5000})
        original_len = len(msgs)

        compress_messages(msgs, max_chars=10_000)
        assert len(msgs) == original_len

    def test_tail_preserved_correctly(self):
        """The most recent messages should be in the tail."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Message {i}: " + "z" * 3000})

        result = compress_messages(msgs, max_chars=10_000, head_size=2, tail_ratio=0.2)

        # Check that the last few original messages appear in result
        last_original = msgs[-1]
        assert last_original in result

    def test_compressed_total_smaller(self):
        """Compressed output should be significantly smaller."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": "x" * 5000})

        original_chars = sum(len(m.get("content", "") or "") for m in msgs)
        result = compress_messages(msgs, max_chars=20_000)
        compressed_chars = sum(len(m.get("content", "") or "") for m in result)

        assert compressed_chars < original_chars * 0.5


# ---------------------------------------------------------------------------
# _summarize_middle
# ---------------------------------------------------------------------------


class TestSummarizeMiddle:
    def test_empty(self):
        result = _summarize_middle([])
        assert "messages" in result.lower()

    def test_user_messages_captured(self):
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "How do I install it?"},
        ]
        result = _summarize_middle(msgs)
        assert "Python" in result

    def test_tool_results_captured(self):
        msgs = [
            {"role": "tool", "content": "File contents: hello world"},
        ]
        result = _summarize_middle(msgs)
        assert "Tool operations" in result

    def test_long_content_truncated(self):
        msgs = [
            {"role": "user", "content": "x" * 10000},
        ]
        result = _summarize_middle(msgs)
        # Summary should be much shorter than 10000 chars
        assert len(result) < 500
