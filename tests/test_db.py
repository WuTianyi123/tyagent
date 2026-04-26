"""Unit tests for tyagent.db — SQLite storage layer."""

import json
import time
from pathlib import Path

import pytest

from tyagent.db import Database


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------


class TestDatabaseInit:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        assert db_path.exists()
        db.close()

    def test_schema_created(self, tmp_path):
        db = Database(tmp_path / "sessions.db")
        # Verify tables exist
        tables = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [row["name"] for row in tables]
        assert "sessions" in table_names
        assert "messages" in table_names
        db.close()

    def test_wal_mode(self, tmp_path):
        db = Database(tmp_path / "test.db")
        cur = db._conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0] == "wal"
        db.close()

    def test_integrity_ok(self, tmp_path):
        db = Database(tmp_path / "test.db")
        assert db.integrity_check() == []
        db.close()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


class TestSessionCRUD:
    def test_get_or_create_new(self, tmp_path):
        db = Database(tmp_path / "test.db")
        session, created = db.get_or_create_session("test_key")
        assert created is True
        assert session["session_key"] == "test_key"
        assert session["created_at"] > 0
        assert session["metadata"] == {}
        db.close()

    def test_get_or_create_existing(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("key1")
        session2, created = db.get_or_create_session("key1")
        assert created is False
        assert session2["session_key"] == "key1"
        db.close()

    def test_archive_and_recreate(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("test_key")
        db.archive_session("test_key")
        session = db.get_or_create_session_after_archive("test_key")
        assert session["session_key"] == "test_key"
        assert "archived_at" not in session["metadata"]
        db.close()

    def test_archive_nonexistent(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.archive_session("no_such_key")  # should not raise
        db.close()

    def test_get_all_session_keys(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("a")
        db.get_or_create_session("b")
        db.get_or_create_session("c")
        keys = db.get_all_session_keys()
        assert len(keys) == 3
        assert set(keys) == {"a", "b", "c"}
        db.close()

    def test_delete_session(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("delete_me")
        db.add_message("delete_me", "user", "hi")
        db.delete_session("delete_me")
        assert "delete_me" not in db.get_all_session_keys()
        assert db.get_message_count("delete_me") == 0
        db.close()

    def test_update_updated_at(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("key")
        old = db.get_or_create_session("key")[0]["updated_at"]
        time.sleep(0.01)
        db.update_session_updated_at("key")
        new = db.get_or_create_session("key")[0]["updated_at"]
        assert new > old
        db.close()


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------


class TestMessageCRUD:
    def test_add_and_get_messages(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        msg_id = db.add_message("s1", "user", "hello")
        assert msg_id > 0
        msgs = db.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        db.close()

    def test_multiple_messages(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        db.add_message("s1", "user", "hello")
        db.add_message("s1", "assistant", "world")
        db.add_message("s1", "tool", "result", tool_call_id="tc1")
        msgs = db.get_messages("s1")
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "tc1"
        db.close()

    def test_assistant_with_tool_calls(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        tool_calls = [
            {
                "id": "tc_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/test"}'},
            }
        ]
        db.add_message("s1", "assistant", content=None, tool_calls=tool_calls)
        msgs = db.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["tool_calls"] == tool_calls
        assert "tool_calls" in msgs[0]
        db.close()

    def test_message_with_reasoning(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        db.add_message("s1", "assistant", "answer", reasoning="thinking...")
        msgs = db.get_messages("s1")
        assert msgs[0]["reasoning_content"] == "thinking..."
        db.close()

    def test_message_count(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        assert db.get_message_count("s1") == 0
        db.add_message("s1", "user", "a")
        db.add_message("s1", "assistant", "b")
        assert db.get_message_count("s1") == 2
        db.close()

    def test_messages_ordered_by_time(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        db.add_message("s1", "user", "first")
        import time as _time
        _time.sleep(0.01)
        db.add_message("s1", "user", "second")
        msgs = db.get_messages("s1")
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"
        db.close()


# ---------------------------------------------------------------------------
# Import / Migration
# ---------------------------------------------------------------------------


class TestImportMessages:
    def test_import(self, tmp_path):
        db = Database(tmp_path / "test.db")
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        count = db.import_messages("migrated_key", msgs, created_at=1000.0)
        assert count == 2
        assert db.get_message_count("migrated_key") == 2
        db.close()

    def test_import_with_tool_calls(self, tmp_path):
        db = Database(tmp_path / "test.db")
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "test"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]
        count = db.import_messages("s1", msgs)
        assert count == 2
        msgs_out = db.get_messages("s1")
        assert msgs_out[0]["tool_calls"] is not None
        assert msgs_out[1]["tool_call_id"] == "tc1"
        db.close()

    def test_import_existing_session_merges(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        db.add_message("s1", "user", "existing")
        count = db.import_messages("s1", [{"role": "assistant", "content": "new"}])
        assert count == 1
        assert db.get_message_count("s1") == 2
        db.close()


# ---------------------------------------------------------------------------
# Concurrency and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_messages_for_nonexistent_session(self, tmp_path):
        db = Database(tmp_path / "test.db")
        msgs = db.get_messages("no_such_key")
        assert msgs == []

    def test_count_for_nonexistent_session(self, tmp_path):
        db = Database(tmp_path / "test.db")
        assert db.get_message_count("no_such_key") == 0

    def test_unarchived_session_not_recreated(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("key")
        s = db.get_or_create_session_after_archive("key")
        assert s["session_key"] == "key"
        assert "archived_at" not in s["metadata"]
        db.close()

    def test_archive_twice(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("key")
        db.archive_session("key")
        db.archive_session("key")  # second archive should not raise
        s = db.get_or_create_session_after_archive("key")
        assert s["session_key"] == "key"
        db.close()

    def test_after_archive_without_existing_session(self, tmp_path):
        """get_or_create_session_after_archive on nonexistent session should create one."""
        db = Database(tmp_path / "test.db")
        s = db.get_or_create_session_after_archive("nonexistent")
        assert s["session_key"] == "nonexistent"
        assert s["metadata"] == {}
        assert s["created_at"] > 0
        db.close()

    def test_large_content(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        big = "x" * 100_000
        db.add_message("s1", "user", big)
        msgs = db.get_messages("s1")
        assert len(msgs[0]["content"]) == 100_000
        db.close()

    def test_special_chars(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("s1")
        special = "你好世界 🔥 \n\t\"'`"
        db.add_message("s1", "user", special)
        msgs = db.get_messages("s1")
        assert msgs[0]["content"] == special
        db.close()

    def test_delete_sessions_older_than(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("old")
        db.add_message("old", "user", "old msg")
        import time
        cutoff = time.time() + 1  # far in the future
        count = db.delete_sessions_older_than(cutoff)
        assert count == 1
        assert db.get_all_session_keys() == []
        db.close()

    def test_delete_sessions_older_than_recent_only(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.get_or_create_session("recent")
        db.add_message("recent", "user", "recent msg")
        cutoff = time.time() - 1  # 1 second ago
        count = db.delete_sessions_older_than(cutoff)
        assert count == 0
        assert "recent" in db.get_all_session_keys()
        db.close()
