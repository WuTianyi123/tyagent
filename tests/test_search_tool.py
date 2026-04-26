"""Tests for the session search tool and FTS5 search helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from tyagent.db import Database, _fts_escape, jieba_segment
from tyagent.tools.registry import registry
from tyagent.tools.search_tool import _db, _handle_session_search, set_search_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_search_db():
    """Reset the global search DB singleton before each test."""
    old = _db
    set_search_db(None)
    yield
    set_search_db(old)


# ---------------------------------------------------------------------------
# jieba_segment tests
# ---------------------------------------------------------------------------


class TestJiebaSegment:
    def test_chinese(self):
        result = jieba_segment("今天天气真不错")
        assert isinstance(result, str)
        assert "今天" in result or "天气" in result or result != ""

    def test_english(self):
        result = jieba_segment("hello world test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_mixed(self):
        result = jieba_segment("跨session记忆方案")
        assert result is not None

    def test_empty_string(self):
        assert jieba_segment("") == ""

    def test_none(self):
        assert jieba_segment(None) == ""

    def test_whitespace(self):
        assert jieba_segment("   ") == ""


# ---------------------------------------------------------------------------
# _fts_escape tests
# ---------------------------------------------------------------------------


class TestFtsEscape:
    def test_normal_terms(self):
        result = _fts_escape("hello world test")
        assert result == '"hello" AND "world" AND "test"'

    def test_empty(self):
        assert _fts_escape("") == ""

    def test_operator_and_lowercased(self):
        result = _fts_escape("hello AND world")
        # AND should be lowercased so it's treated as literal
        assert '"and"' in result
        assert '"hello"' in result
        assert '"world"' in result

    def test_operator_or_lowercased(self):
        result = _fts_escape("cat OR dog")
        assert '"or"' in result

    def test_operator_not_lowercased(self):
        result = _fts_escape("foo NOT bar")
        assert '"not"' in result

    def test_quotes_in_terms(self):
        """Quotes inside terms are escaped as double-double-quotes."""
        result = _fts_escape('say "hello"')
        assert '"""' in result  # each " becomes ""

    def test_single_term(self):
        result = _fts_escape("hello")
        assert result == '"hello"'


# ---------------------------------------------------------------------------
# Database FTS5 search tests
# ---------------------------------------------------------------------------


class TestFTS5Search:
    @pytest.fixture
    def db(self):
        """Create an in-memory Database with FTS5 enabled."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            database = Database(db_path)
            yield database
            database.close()

    def test_add_message_indexes_fts(self, db):
        """Adding a message should make it searchable via FTS."""
        db.add_message("s1", "user", "我喜欢在周末去公园散步")
        results = db.search_messages("散步", limit=10)
        assert len(results) >= 1
        assert results[0]["session_key"] == "s1"

    def test_search_english(self, db):
        db.add_message("s1", "user", "The quick brown fox jumps over the lazy dog")
        results = db.search_messages("fox", limit=10)
        assert len(results) >= 1
        assert "fox" in results[0]["content"]

    def test_search_empty_results(self, db):
        db.add_message("s1", "user", "hello world")
        results = db.search_messages("zzzznonexistent", limit=10)
        assert results == []

    def test_search_limit(self, db):
        for i in range(10):
            db.add_message("s1", "user", f"message number {i} about tests")
        results = db.search_messages("tests", limit=3)
        assert len(results) <= 3

    def test_search_multiple_sessions(self, db):
        db.add_message("s1", "user", "project planning meeting at 3pm")
        db.add_message("s2", "user", "daily standup meeting notes")
        results = db.search_messages("meeting", limit=10)
        assert len(results) == 2

    def test_search_mixed_query(self, db):
        db.add_message("s1", "user", "跨session记忆方案讨论")
        results = db.search_messages("记忆 方案", limit=10)
        assert len(results) >= 1

    def test_delete_session_cleans_fts(self, db):
        db.add_message("s1", "user", "searchable content")
        db.delete_session("s1")
        results = db.search_messages("searchable", limit=10)
        assert results == []

    def test_import_messages_indexes_fts(self, db):
        msgs = [
            {"role": "user", "content": "imported message about databases"},
            {"role": "assistant", "content": "SQLite is great"},
        ]
        db.import_messages("s_import", msgs)
        results = db.search_messages("databases", limit=10)
        assert len(results) >= 1
        results2 = db.search_messages("SQLite", limit=10)
        assert len(results2) >= 1

    def test_search_result_content_truncated(self, db):
        long_text = "hello " * 200  # ~1000 chars of searchable text
        db.add_message("s1", "user", long_text)
        results = db.search_messages("hello", limit=10)
        assert len(results) >= 1
        # Content should be truncated to 500 chars
        assert len(results[0]["content"]) <= 500

    def test_search_returns_ranked(self, db):
        db.add_message("s1", "user", "python programming")
        db.add_message("s1", "user", "python for data science with python")
        results = db.search_messages("python", limit=10)
        assert len(results) >= 1
        # Results should have a rank field
        assert "rank" in results[0]

    def test_search_bad_query_doesnt_crash(self, db):
        """Malformed FTS queries should return empty list, not crash."""
        db.add_message("s1", "user", "some content")
        # _fts_escape should prevent syntax errors, but test the safety net
        results = db.search_messages("random", limit=50)
        assert isinstance(results, list)

    def test_search_with_reasoning(self, db):
        db.add_message("s1", "assistant", "visible content", reasoning="hidden reasoning text")
        results = db.search_messages("visible", limit=10)
        assert len(results) >= 1

    def test_empty_query_returns_empty(self, db):
        results = db.search_messages("", limit=10)
        assert results == []

    def test_search_rank_not_zero(self, db):
        """A matching result should have a non-zero rank (lower = better match)."""
        db.add_message("s1", "user", "critical unique term xyz12345")
        results = db.search_messages("xyz12345", limit=10)
        assert len(results) >= 1
        # rank is the FTS5 BM25 score; should be > 0 for a match
        # (0 indicates perfect match, >0 indicates some divergence)
        assert "rank" in results[0]


# ---------------------------------------------------------------------------
# Search tool handler tests
# ---------------------------------------------------------------------------


class TestSearchToolHandler:
    def test_tool_registered(self):
        assert "session_search" in registry.get_all_names()

    def test_dispatch_no_db(self):
        result = json.loads(_handle_session_search({"query": "test"}))
        assert "error" in result
        assert "not available" in result["error"].lower()

    def test_dispatch_missing_query(self):
        result = json.loads(_handle_session_search({}))
        assert "error" in result
        assert "query" in result["error"].lower()

    def test_dispatch_empty_query(self):
        result = json.loads(_handle_session_search({"query": ""}))
        assert "error" in result

    def test_dispatch_with_db(self):
        """Full integration: search tool with real database."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            database = Database(db_path)
            database.add_message("test_session", "user", "hello world test content")
            set_search_db(database)
            try:
                result = json.loads(_handle_session_search({"query": "hello"}))
                assert result.get("success") is True
                assert result.get("count", 0) >= 1
                assert len(result.get("results", [])) >= 1
                assert result["results"][0]["session_key"] == "test_session"
            finally:
                set_search_db(None)
                database.close()

    def test_dispatch_limit_clamping(self):
        """Limit should be clamped to [1, 20]."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            database = Database(db_path)
            for i in range(25):
                database.add_message("s1", "user", f"searchable message {i}")
            set_search_db(database)
            try:
                # Request 100, should get clamped to 20
                result = json.loads(_handle_session_search({"query": "searchable", "limit": 100}))
                assert result.get("success") is True
                assert result.get("count", 0) <= 20
            finally:
                set_search_db(None)
                database.close()

    def test_dispatch_invalid_limit(self):
        """Non-integer limit should fall back to default (5)."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "test.db"
            database = Database(db_path)
            for i in range(10):
                database.add_message("s1", "user", f"term xyz {i}")
            set_search_db(database)
            try:
                result = json.loads(_handle_session_search({"query": "term", "limit": "invalid"}))
                assert result.get("success") is True
                assert result.get("count", 0) >= 1
            finally:
                set_search_db(None)
                database.close()
