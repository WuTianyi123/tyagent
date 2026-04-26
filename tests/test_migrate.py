"""Unit tests for tyagent.migrate — JSON to SQLite migration."""

import json
import time
from pathlib import Path

import pytest

from tyagent.migrate import migrate_from_json, verify_migration


def _make_session_file(
    tmp_path: Path,
    name: str,
    messages: list,
    *,
    archived: bool = False,
    created_at: float = 0,
) -> Path:
    """Create a JSON session file in tmp_path."""
    data = {
        "session_key": f"feishu:{name}",
        "messages": messages,
        "created_at": created_at or time.time(),
        "updated_at": time.time(),
        "metadata": {},
    }
    fname = f"feishu_{name}.json"
    if archived:
        fname = f"feishu_{name}__archived_20260401_120000.json"
        data["metadata"]["archived_at"] = time.time()
    path = tmp_path / fname
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestMigrateFromJson:
    def test_empty_directory(self, tmp_path):
        """No JSON files should result in 0 sessions imported."""
        count = migrate_from_json(tmp_path)
        assert count == 0

    def test_single_session(self, tmp_path):
        _make_session_file(tmp_path, "chat1", [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        count = migrate_from_json(tmp_path)
        assert count == 1

    def test_multiple_sessions(self, tmp_path):
        _make_session_file(tmp_path, "a", [{"role": "user", "content": "1"}])
        _make_session_file(tmp_path, "b", [{"role": "user", "content": "2"}])
        _make_session_file(tmp_path, "c", [{"role": "user", "content": "3"}])
        count = migrate_from_json(tmp_path)
        assert count == 3

    def test_session_with_tool_calls(self, tmp_path):
        _make_session_file(tmp_path, "tools", [
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "file content", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Here is the content."},
        ])
        count = migrate_from_json(tmp_path)
        assert count == 1

    def test_archived_sessions(self, tmp_path):
        _make_session_file(tmp_path, "old", [{"role": "user", "content": "old"}], archived=True)
        _make_session_file(tmp_path, "new", [{"role": "user", "content": "new"}])
        count = migrate_from_json(tmp_path)
        assert count == 2

    def test_nonexistent_directory_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            migrate_from_json(Path("/nonexistent/path"))

    def test_empty_messages_file(self, tmp_path):
        """A session file with no messages should be skipped."""
        _make_session_file(tmp_path, "empty", [])
        count = migrate_from_json(tmp_path)
        assert count == 0

    def test_corrupt_json_file(self, tmp_path):
        """A corrupt JSON file should not crash the migration."""
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        _make_session_file(tmp_path, "good", [{"role": "user", "content": "ok"}])
        count = migrate_from_json(tmp_path)
        assert count == 1


class TestVerifyMigration:
    def test_verify_no_db(self, tmp_path):
        """Verification without a DB should report error."""
        result = verify_migration(tmp_path)
        assert "error" in result

    def test_verify_after_migration(self, tmp_path):
        _make_session_file(tmp_path, "a", [{"role": "user", "content": "hi"}])
        _make_session_file(tmp_path, "b", [{"role": "user", "content": "hello"}])
        migrate_from_json(tmp_path)
        result = verify_migration(tmp_path)
        assert result["json_files"] == 2
        assert result["db_sessions"] == 2
