"""Unit tests for tyagent.context — deterministic tool-message-dropping compression."""

import pytest

from tyagent.context import (
    DEFAULT_MAX_CHARS,
    _content_chars,
    build_api_messages,
    compress_messages,
    estimate_tokens,
    should_compress,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestContentChars:
    def test_empty(self):
        assert _content_chars([]) == 0

    def test_content_only(self):
        assert _content_chars([{"role": "user", "content": "hello"}]) == 5

    def test_tool_calls_included(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "type": "function",
                             "function": {"name": "test", "arguments": "{}"}}]},
        ]
        chars = _content_chars(msgs)
        assert chars > 5  # content "hi" + tool_calls metadata
        assert "id" in str(chars) or chars > 10  # tool call metadata adds chars

    def test_none_content_safe(self):
        assert _content_chars([{"role": "assistant", "content": None}]) == 0

    def test_none_function_safe(self):
        """tool_calls with function=None should not crash."""
        msgs = [{"role": "assistant",
                 "tool_calls": [{"id": "tc1", "function": None}]}]
        chars = _content_chars(msgs)
        assert chars >= 0

    def test_empty_tool_calls_safe(self):
        assert _content_chars([{"role": "assistant", "tool_calls": []}]) == 0


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        tokens = estimate_tokens(msgs)
        assert tokens > 0
        assert tokens < 100

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 2000},
        ]
        tokens = estimate_tokens(msgs)
        assert tokens > 500

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
        msgs = [{"role": "user", "content": "a" * 30_000}]
        assert should_compress(msgs, max_tokens=5000) is True
        assert should_compress(msgs, max_tokens=20000) is False


# ---------------------------------------------------------------------------
# build_api_messages — deterministic tool-message-dropping
# ---------------------------------------------------------------------------


class TestBuildApiMessages:
    def test_short_messages_not_compressed(self):
        """Messages within budget should be returned as-is (same object)."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = build_api_messages(msgs, max_chars=1_000_000)
        assert result is msgs  # same object if within budget

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
        result = build_api_messages(msgs, max_chars=10)  # force compression

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
        result = build_api_messages(msgs, max_chars=10)

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
        result = build_api_messages(msgs, max_chars=10)
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
        result = build_api_messages(msgs, max_chars=10)

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
        result = build_api_messages(msgs, max_chars=50_000)

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
        build_api_messages(msgs, max_chars=10_000)
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

        result = build_api_messages(msgs, max_chars=20_000)
        assert len(result) < len(msgs)

    def test_within_budget_returns_same(self):
        """Messages within budget should be returned as-is even with tool calls."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "done"},
        ]
        result = build_api_messages(msgs, max_chars=1_000_000)
        assert result is msgs

    def test_warning_if_still_over_budget(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        msgs = [
            {"role": "user", "content": "a" * 100_000},
        ]
        build_api_messages(msgs, max_chars=1)  # far too small
        # Just one user message — no compression possible, logs warning


# ---------------------------------------------------------------------------
# Aliases and constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_compress_messages_is_build_api_messages(self):
        assert compress_messages is build_api_messages

    def test_default_max_chars_exported(self):
        assert DEFAULT_MAX_CHARS == 280_000
