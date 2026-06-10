"""Tests for compaction summary persistence.

Tests that after compaction:
  1. Summary user message is persisted to SessionStore
  2. A new session_id is generated for the compacted era
  3. Old messages remain under the previous session_id
  4. Reloading from store preserves the compacted state
  5. Second compaction filters out the old summary
  6. Sub-agent (child mode) messages are handled correctly
"""

from __future__ import annotations

import asyncio
import json as _json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tyagent.session import SessionStore
from tyagent.compaction import (
    SUMMARY_PREFIX,
    build_compacted_history,
    collect_user_messages,
    is_summary_message,
    run_compact,
    total_token_estimate,
)

pytestmark = pytest.mark.asyncio


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_messages(n: int, *, base: int = 0) -> list[dict]:
    """Create a sequence of user/assistant conversation messages."""
    msgs = []
    for i in range(n):
        text = f"Message number {base + i} with some content to make tokens " * 5
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": f"Response to message {base + i} " * 5})
    return msgs


async def _fake_compact(
    messages: list[dict],
    mock_client: AsyncMock,
    summary_text: str = "This is the compaction summary.",
) -> list[dict]:
    """Run compaction against a mock HTTP client that returns *summary_text*."""
    mock_client.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "choices": [{"message": {"content": summary_text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
    )
    result = await run_compact(
        messages,
        model="test-model",
        api_key="test-key",
        base_url="http://fake.test/v1",
        http_client=mock_client,
    )
    assert result is not None, "Compaction failed"
    return result


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCompactionPersistence:
    """Unit-level: compaction output can be stored and reloaded."""

    def test_compacted_messages_contain_summary(self, tmp_path):
        """build_compacted_history produces a summary user message."""
        tail = ["msg A", "msg B"]
        summary = "This is a summary."
        history = build_compacted_history(tail, summary)

        # All entries are user messages
        assert all(m["role"] == "user" for m in history)
        # Last entry is the summary
        assert history[-1]["content"] == f"{SUMMARY_PREFIX}\n{summary}"
        # is_summary_message detects it
        assert is_summary_message(history[-1]["content"])
        # Other entries are NOT summaries
        for m in history[:-1]:
            assert not is_summary_message(m["content"])

    async def test_compacted_persisted_and_reloaded(self, tmp_path):
        """Compacted messages survive store close/reopen."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session = store.get("test:persist")

        # Create a bunch of messages that would need compaction
        msgs = _make_messages(30)
        for m in msgs:
            session.add_message(m["role"], m["content"])

        # Simulate compaction via mock
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        compacted = await _fake_compact(msgs, mock_client)

        # Simulate the new-era write: freshen session, write compacted msgs
        old_sid = session.metadata["current_session_id"]
        store.freshen_session("test:persist")
        session = store.get("test:persist")  # re-get for updated metadata
        new_sid = session.metadata["current_session_id"]
        assert new_sid != old_sid

        for m in compacted:
            store.add_message(
                "test:persist", m["role"], m.get("content", ""),
                session_id=new_sid,
            )

        # Verify: new sid has summary
        new_msgs = store.get_messages("test:persist", session_id=new_sid)
        assert len(new_msgs) == len(compacted)
        assert any(is_summary_message(m["content"] or "") for m in new_msgs)

        # Old sid still has original messages
        assert len(store.get_messages("test:persist", session_id=old_sid)) == len(msgs)

        # Close and reopen
        store.close()
        store2 = SessionStore(sessions_dir=tmp_path / "sessions")
        reloaded = store2.get_messages("test:persist", session_id=new_sid)
        assert len(reloaded) == len(compacted)
        assert any(is_summary_message(m["content"] or "") for m in reloaded)
        store2.close()

    async def test_second_compaction_filters_old_summary(self, tmp_path):
        """collect_user_messages skips previous compaction summaries."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session = store.get("test:double")

        # First compaction: create messages with summary
        msgs1 = _make_messages(20, base=0)
        for m in msgs1:
            session.add_message(m["role"], m["content"])

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        compacted1 = await _fake_compact(msgs1, mock_client, summary_text="Summary 1")

        store.freshen_session("test:double")
        session = store.get("test:double")  # re-get for updated metadata
        sid1 = session.metadata["current_session_id"]
        for m in compacted1:
            store.add_message("test:double", m["role"], m.get("content", ""), session_id=sid1)

        # Add more messages after compaction
        msgs2 = _make_messages(20, base=20)
        for m in msgs2:
            session.add_message(m["role"], m["content"])

        # Second compaction: should skip "Summary 1" when collecting user messages
        # and generate a NEW summary
        full_msgs = store.get_messages("test:double", session_id=sid1)
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "Summary 2"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )
        compacted2 = await run_compact(
            full_msgs,
            model="test-model",
            api_key="test-key",
            base_url="http://fake.test/v1",
            http_client=mock_client,
        )
        assert compacted2 is not None

        # New summary is "Summary 2", not "Summary 1"
        summary_contents = [m["content"] for m in compacted2 if is_summary_message(m["content"] or "")]
        assert len(summary_contents) == 1
        assert "Summary 2" in summary_contents[0]
        assert "Summary 1" not in summary_contents[0]

        store.close()

    def test_summary_message_detected_across_formats(self):
        """is_summary_message works with real summary prefix."""
        assert is_summary_message(f"{SUMMARY_PREFIX}\nSome summary")
        assert is_summary_message(f"{SUMMARY_PREFIX}\nAny text here")
        # It must start with prefix + newline (Codex-aligned)
        assert not is_summary_message(f"{SUMMARY_PREFIX} without newline")
        assert not is_summary_message("Regular message")
        assert not is_summary_message("")


class TestCompactionSessionIsolation:
    """Compacted and original messages are isolated by session_id."""

    def test_freshen_preserves_old_messages(self, tmp_path):
        """After freshen_session, old messages are still accessible."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session = store.get("test:isolate")

        # Add some messages under the current session_id
        old_sid = session.metadata["current_session_id"]
        for i in range(5):
            session.add_message("user", f"msg {i}")

        # Freshen (generates new current_session_id in DB)
        store.freshen_session("test:isolate")
        # Re-get to pick up updated metadata
        session = store.get("test:isolate")
        new_sid = session.metadata["current_session_id"]
        assert new_sid != old_sid

        # Old messages still there
        old_msgs = store.get_messages("test:isolate", session_id=old_sid)
        assert len(old_msgs) == 5

        # New session starts empty
        new_msgs = store.get_messages("test:isolate", session_id=new_sid)
        assert len(new_msgs) == 0

        store.close()

    def test_add_after_freshen_uses_new_sid(self, tmp_path):
        """Messages added after freshen_session go to new session_id."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session = store.get("test:after")

        old_sid = session.metadata["current_session_id"]
        session.add_message("user", "before")

        store.freshen_session("test:after")
        # Re-get to pick up updated metadata with new session_id
        session = store.get("test:after")
        session.add_message("user", "after")

        assert len(store.get_messages("test:after", session_id=old_sid)) == 1
        new_sid = session.metadata["current_session_id"]
        assert len(store.get_messages("test:after", session_id=new_sid)) == 1

        store.close()


class TestCompactionSubAgents:
    """Sub-agent (child mode) messages are handled during compaction."""

    async def test_child_messages_not_persisted_to_parent(self, tmp_path):
        """Child agent messages go to a different session_key, not mixed in."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")

        parent = store.get("test:parent")
        child = store.get("test:parent//spawn_0")

        parent.add_message("user", "do something")
        child.add_message("user", "child task")
        child.add_message("assistant", "child done")

        parent_msgs = store.get_messages("test:parent")
        child_msgs = store.get_messages("test:parent//spawn_0")

        assert len(parent_msgs) == 1
        assert len(child_msgs) == 2
        # Child messages don't leak into parent
        assert all(m["content"] != "child task" for m in parent_msgs)

        store.close()
