"""Unit tests for tyagent.session — Session and SessionStore (SQLite-backed)."""

import time
from pathlib import Path

import pytest

from tyagent.session import Session, SessionError, SessionStore


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


class TestSession:
    def test_defaults(self):
        s = Session(session_key="test")
        assert s.created_at > 0
        assert s.updated_at > 0
        assert s.metadata == {}

    def test_messages_without_store(self):
        """Session without a store returns empty messages."""
        s = Session(session_key="test")
        assert s.messages == []

    def test_add_message_without_store_raises(self):
        """add_message without store should raise SessionError."""
        s = Session(session_key="test")
        with pytest.raises(SessionError):
            s.add_message("user", "hello")

    def test_to_dict(self):
        s = Session(session_key="roundtrip", created_at=100.0, updated_at=200.0)
        s.metadata["foo"] = "bar"
        d = s.to_dict()
        assert d["session_key"] == "roundtrip"
        assert "messages" not in d
        assert d["metadata"] == {"foo": "bar"}

    def test_from_dict_roundtrip(self):
        d = {
            "session_key": "test_key",
            "created_at": 100.0,
            "updated_at": 200.0,
            "metadata": {"key": "val"},
        }
        s = Session.from_dict(d)
        assert s.session_key == "test_key"
        assert s.created_at == 100.0
        assert s.metadata == {"key": "val"}

    def test_from_dict_missing_fields(self):
        d = {"session_key": "minimal"}
        s = Session.from_dict(d)
        assert s.metadata == {}
        assert s.created_at > 0


# ---------------------------------------------------------------------------
# SessionStore — SQLite-backed
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_get_creates_new(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        s = store.get("new_key")
        assert s.session_key == "new_key"
        assert store.get_message_count("new_key") == 0
        store.close()

    def test_get_returns_same(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        s1 = store.get("key1")
        store.add_message("key1", "user", "hello")
        s2 = store.get("key1")
        assert s2.session_key == s1.session_key
        store.close()

    def test_empty_session_key_raises(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        with pytest.raises(SessionError, match="must not be empty"):
            store.get("")
        store.close()

    def test_session_messages_property(self, tmp_path):
        """Session.messages should return messages from DB."""
        store = SessionStore(sessions_dir=tmp_path)
        s = store.get("key1")
        assert s.messages == []
        store.add_message("key1", "user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["content"] == "hello"
        store.close()

    def test_session_add_message_via_session(self, tmp_path):
        """session.add_message() should delegate to store."""
        store = SessionStore(sessions_dir=tmp_path)
        s = store.get("key1")
        msg_id = s.add_message("user", "via_session")
        assert msg_id > 0
        assert len(s.messages) == 1
        assert s.messages[0]["content"] == "via_session"
        store.close()

    def test_add_and_count_messages(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("key1")
        store.add_message("key1", "user", "hello")
        store.add_message("key1", "assistant", "world")
        assert store.get_message_count("key1") == 2
        store.close()

    def test_get_messages(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("s1")
        store.add_message("s1", "user", "hi")
        store.add_message("s1", "assistant", "hello")
        msgs = store.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hi"
        assert msgs[1]["content"] == "hello"
        store.close()

    def test_add_message_with_extras(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("s1")
        store.add_message(
            "s1", "assistant", None,
            tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "test"}}],
            reasoning="thinking...",
        )
        msgs = store.get_messages("s1")
        assert msgs[0]["tool_calls"] is not None
        assert msgs[0]["reasoning_content"] == "thinking..."
        store.close()

    def test_add_message_with_tool_call_id(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("s1")
        store.add_message("s1", "tool", "result", tool_call_id="tc1")
        msgs = store.get_messages("s1")
        assert msgs[0]["tool_call_id"] == "tc1"
        store.close()

    def test_archive_and_recreate(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("key1")
        store.add_message("key1", "user", "old message")
        assert store.get_message_count("key1") == 1

        store.archive("key1")
        fresh = store.get_or_create_after_archive("key1")
        assert fresh.session_key == "key1"
        # Old messages should still be in DB
        assert store.get_message_count("key1") == 1
        store.close()

    def test_reset_legacy(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("key1")
        store.add_message("key1", "user", "hello")
        store.reset("key1")
        fresh = store.get_or_create_after_archive("key1")
        assert store.get_message_count("key1") == 1
        store.close()

    def test_delete(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("key1")
        store.add_message("key1", "user", "hi")
        store.delete("key1")
        store.get("key1")
        assert store.get_message_count("key1") == 0
        store.close()

    def test_all_session_keys(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("a")
        store.get("b")
        store.get("c")
        keys = store.all_session_keys()
        assert set(keys) == {"a", "b", "c"}
        store.close()

    def test_in_memory_store(self):
        """Store without sessions_dir works (uses temp dir internally)."""
        store = SessionStore()
        store.get("mem1")
        store.add_message("mem1", "user", "hi")
        msgs = store.get_messages("mem1")
        assert msgs[0]["content"] == "hi"
        store.close()

    def test_save_is_noop(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("k")
        store.add_message("k", "user", "hi")
        store.save("k")
        store.close()

    def test_integrity_check(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        assert store.integrity_check() == []
        store.close()

    def test_context_manager(self, tmp_path):
        with SessionStore(sessions_dir=tmp_path) as store:
            s = store.get("ctx_test")
            s.add_message("user", "hello")
            assert len(s.messages) == 1

    def test_session_updated_at_after_add(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        s = store.get("key")
        old_updated = s.updated_at
        time.sleep(0.01)
        s.add_message("user", "hi")
        assert s.updated_at > old_updated
        store.close()


# ---------------------------------------------------------------------------
# Prune old sessions
# ---------------------------------------------------------------------------


class TestPruneOldSessions:
    def test_prune_zero_days(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("keep")
        assert store.prune_old_sessions(max_age_days=0) == 0
        store.close()

    def test_prune_nothing_if_all_recent(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("recent1")
        store.get("recent2")
        assert store.prune_old_sessions(max_age_days=90) == 0
        assert len(store.all_session_keys()) == 2
        store.close()


# ---------------------------------------------------------------------------
# Resume-pending / suspend recovery
# ---------------------------------------------------------------------------


class TestRecoveryMethods:
    def test_mark_resume_pending(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("rk1")
        result = store.mark_resume_pending("rk1", reason="restart_timeout")
        assert result is True
        assert store.is_resume_pending("rk1") is True
        s = store.get("rk1")
        assert s.metadata.get("resume_pending") is True
        assert s.metadata.get("resume_reason") == "restart_timeout"
        assert "resume_marked_at" in s.metadata
        store.close()

    def test_mark_resume_pending_on_suspended_returns_false(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("sk1")
        store.suspend_session("sk1", reason="crash_recovery")
        result = store.mark_resume_pending("sk1", reason="restart_timeout")
        assert result is False
        # Should still be suspended, not resume_pending
        assert store.is_suspended("sk1") is True
        assert store.is_resume_pending("sk1") is False
        store.close()

    def test_clear_resume_pending(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("rk2")
        store.mark_resume_pending("rk2")
        assert store.is_resume_pending("rk2") is True
        result = store.clear_resume_pending("rk2")
        assert result is True
        assert store.is_resume_pending("rk2") is False
        s = store.get("rk2")
        assert "resume_pending" not in s.metadata
        assert "resume_reason" not in s.metadata
        assert "resume_marked_at" not in s.metadata
        store.close()

    def test_clear_resume_pending_not_set(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("rk3")
        result = store.clear_resume_pending("rk3")
        assert result is False
        store.close()

    def test_suspend_session(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("sk2")
        result = store.suspend_session("sk2", reason="crash_recovery")
        assert result is True
        assert store.is_suspended("sk2") is True
        s = store.get("sk2")
        assert s.metadata.get("suspended") is True
        assert s.metadata.get("suspend_reason") == "crash_recovery"
        assert "suspend_at" in s.metadata
        store.close()

    def test_suspend_recently_active(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("active1")
        store.get("old1")
        # Use a large max_age to cover sessions just created
        count = store.suspend_recently_active(max_age_seconds=1000)
        assert count >= 1  # at least one session was active recently
        s = store.get("active1")
        assert s.metadata.get("suspended") is True
        store.close()

    def test_suspend_recently_active_skips_resume_pending(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("skip1")
        store.mark_resume_pending("skip1", reason="restart_timeout")
        store.get("target1")
        count = store.suspend_recently_active(max_age_seconds=1000)
        # skip1 should NOT be suspended (has resume_pending)
        assert store.is_suspended("skip1") is False
        # target1 should be suspended
        assert store.is_suspended("target1") is True
        store.close()

    def test_suspend_recently_active_skips_already_suspended(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("already_suspended")
        store.suspend_session("already_suspended")
        store.get("fresh1")
        count = store.suspend_recently_active(max_age_seconds=1000)
        assert store.is_suspended("already_suspended") is True
        assert store.is_suspended("fresh1") is True
        store.close()

    def test_is_suspended(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("ck1")
        assert store.is_suspended("ck1") is False
        store.suspend_session("ck1")
        assert store.is_suspended("ck1") is True
        store.close()

    def test_is_resume_pending(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("ck2")
        assert store.is_resume_pending("ck2") is False
        store.mark_resume_pending("ck2")
        assert store.is_resume_pending("ck2") is True
        store.close()

    def test_suspend_default_reason(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("def_reason")
        store.suspend_session("def_reason")
        s = store.get("def_reason")
        assert s.metadata.get("suspend_reason") == "crash_recovery"
        store.close()

    def test_mark_resume_pending_default_reason(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("def_resume")
        store.mark_resume_pending("def_resume")
        s = store.get("def_resume")
        assert s.metadata.get("resume_reason") == "restart_timeout"
        store.close()
