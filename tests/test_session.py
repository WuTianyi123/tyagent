"""Unit tests for ty_agent.session — Session and SessionStore."""

import json
import time
from pathlib import Path

import pytest

from ty_agent.session import Session, SessionStore, _sanitize_session_key


# ---------------------------------------------------------------------------
# _sanitize_session_key
# ---------------------------------------------------------------------------


class TestSanitizeSessionKey:
    def test_safe_key_unchanged(self):
        assert _sanitize_session_key("abc123") == "abc123"

    def test_safe_key_with_colon_dash(self):
        assert _sanitize_session_key("feishu:chat_id") == "feishu:chat_id"

    def test_unsafe_chars_hashed(self):
        key = "hello world!"  # space and ! are unsafe
        result = _sanitize_session_key(key)
        assert result != key
        assert len(result) == 32  # sha256 hex prefix

    def test_chinese_chars_hashed(self):
        key = "飞书:用户"
        result = _sanitize_session_key(key)
        assert len(result) == 32

    def test_consistent_hash(self):
        key = "a/b\\c"
        assert _sanitize_session_key(key) == _sanitize_session_key(key)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


class TestSession:
    def test_defaults(self):
        s = Session(session_key="test")
        assert s.messages == []
        assert s.metadata == {}
        assert s.created_at > 0
        assert s.updated_at > 0

    def test_add_message(self):
        s = Session(session_key="test")
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"
        assert s.messages[0]["content"] == "hello"

    def test_add_message_with_extras(self):
        s = Session(session_key="test")
        s.add_message("tool", "result", tool_call_id="tc1")
        assert s.messages[0]["tool_call_id"] == "tc1"

    def test_add_message_updates_timestamp(self):
        s = Session(session_key="test")
        before = s.updated_at
        time.sleep(0.01)
        s.add_message("user", "hi")
        assert s.updated_at >= before

    def test_clear(self):
        s = Session(session_key="test")
        s.add_message("user", "hello")
        s.clear()
        assert s.messages == []
        assert s.updated_at > s.created_at

    def test_to_dict_roundtrip(self):
        s = Session(session_key="roundtrip")
        s.add_message("user", "hello")
        s.add_message("assistant", "world")
        s.metadata["foo"] = "bar"

        d = s.to_dict()
        s2 = Session.from_dict(d)
        assert s2.session_key == "roundtrip"
        assert len(s2.messages) == 2
        assert s2.metadata == {"foo": "bar"}
        assert s2.created_at == s.created_at

    def test_from_dict_missing_fields(self):
        d = {"session_key": "minimal"}
        s = Session.from_dict(d)
        assert s.messages == []
        assert s.metadata == {}
        assert s.created_at > 0


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_get_creates_new(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        s = store.get("new_key")
        assert s.session_key == "new_key"
        assert s.messages == []

    def test_get_returns_same(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        s1 = store.get("key1")
        s1.add_message("user", "hello")
        s2 = store.get("key1")
        assert s2 is s1

    def test_reset(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        s = store.get("key1")
        s.add_message("user", "hello")
        store.reset("key1")
        assert s.messages == []

    def test_delete(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("key1").add_message("user", "hi")
        store.save("key1")
        store.delete("key1")
        # get creates a new one
        s = store.get("key1")
        assert s.messages == []

    def test_save_and_load(self, tmp_path):
        store1 = SessionStore(sessions_dir=tmp_path)
        s = store1.get("persist_test")
        s.add_message("user", "hello")
        s.add_message("assistant", "world")
        store1.save("persist_test")

        # New store loading from same dir
        store2 = SessionStore(sessions_dir=tmp_path)
        s2 = store2.get("persist_test")
        assert len(s2.messages) == 2
        assert s2.messages[0]["content"] == "hello"

    def test_load_corrupt_file(self, tmp_path):
        # Write a corrupt JSON file
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        # Should not crash, just log warning
        store = SessionStore(sessions_dir=tmp_path)
        assert store.all_session_keys() == []

    def test_load_non_dict_json(self, tmp_path):
        bad_file = tmp_path / "array.json"
        bad_file.write_text("[1, 2, 3]", encoding="utf-8")
        store = SessionStore(sessions_dir=tmp_path)
        assert store.all_session_keys() == []

    def test_all_session_keys(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("a")
        store.get("b")
        store.get("c")
        keys = store.all_session_keys()
        assert set(keys) == {"a", "b", "c"}

    def test_in_memory_store(self):
        """Store without sessions_dir works in-memory only."""
        store = SessionStore()
        store.get("mem1").add_message("user", "hi")
        assert store.get("mem1").messages[0]["content"] == "hi"

    def test_reset_nonexistent(self, tmp_path):
        """Resetting a key that was never added should not crash."""
        store = SessionStore(sessions_dir=tmp_path)
        store.reset("no_such_key")  # should not raise

    def test_save_sanitized_key(self, tmp_path):
        """Keys with unsafe chars get hashed for filename."""
        store = SessionStore(sessions_dir=tmp_path)
        store.get("safe/key").add_message("user", "hi")
        store.save("safe/key")
        # Should create a file with hashed name
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        assert "/" not in json_files[0].name


# ---------------------------------------------------------------------------
# SessionStore.prune_old_sessions
# ---------------------------------------------------------------------------


class TestPruneOldSessions:
    def test_prune_removes_old(self, tmp_path):
        # Build a session file with an old timestamp manually
        old_data = {
            "session_key": "old_session",
            "messages": [{"role": "user", "content": "old"}],
            "created_at": time.time() - 100 * 86400,
            "updated_at": time.time() - 100 * 86400,
            "metadata": {},
        }
        (tmp_path / "old_session.json").write_text(
            json.dumps(old_data), encoding="utf-8"
        )

        store = SessionStore(sessions_dir=tmp_path)
        # Also add a recent one
        store.get("new_session").add_message("user", "new")
        store.save("new_session")

        removed = store.prune_old_sessions(max_age_days=90)
        assert removed == 1
        assert "old_session" not in store.all_session_keys()
        assert "new_session" in store.all_session_keys()

    def test_prune_zero_days(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("keep")
        assert store.prune_old_sessions(max_age_days=0) == 0

    def test_prune_nothing_if_all_recent(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        store.get("recent1")
        store.get("recent2")
        assert store.prune_old_sessions(max_age_days=90) == 0
        assert len(store.all_session_keys()) == 2
