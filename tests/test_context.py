"""Unit tests for tyagent.context — deterministic tool-message-dropping compression."""

import pytest

from tyagent.context import build_api_messages


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
