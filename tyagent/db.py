"""SQLite storage layer for tyagent sessions and messages.

Provides thread-safe database operations using WAL mode.
Messages are NEVER deleted from the database — they are the single source of truth.
Compression is done at query time (build_api_messages).

Schema v2 adds FTS5 full-text search with jieba segmentation for Chinese support.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import threading

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Chinese-capable FTS via jieba
# ---------------------------------------------------------------------------

_DICT_LOADED: bool = False
_dict_lock = threading.Lock()


def load_jieba_dict(dict_path: str | Path | None = None) -> None:
    """Load custom jieba dictionary once (idempotent, thread-safe)."""
    global _DICT_LOADED
    if _DICT_LOADED:
        return
    with _dict_lock:
        # Double-checked locking
        if _DICT_LOADED:
            return
        import jieba

        if dict_path and Path(dict_path).exists():
            jieba.load_userdict(str(dict_path))
        _DICT_LOADED = True


def jieba_segment(text: str | None) -> str:
    """Segment text with jieba and join with spaces for FTS5 indexing.

    Returns empty string for None/empty/whitespace-only input.
    """
    if not text or not text.strip():
        return ""
    import jieba

    return " ".join(jieba.cut(text))


# -- FTS5 query sanitisation ------------------------------------------------

_FTS_SPECIAL = frozenset({"AND", "OR", "NOT", "NEAR"})


def _fts_escape(segmented: str) -> str:
    """Wrap each term in double quotes so FTS5 does not interpret operators.

    This prevents FTS5 syntax errors from user queries containing special
    characters like ``*``, ``"``, ``AND``, ``OR``, etc.
    """
    if not segmented:
        return ""
    terms = segmented.split()
    quoted: List[str] = []
    for t in terms:
        t_upper = t.upper()
        if t_upper in _FTS_SPECIAL:
            # Lower-case FTS operators so they are treated as literal terms
            t = t.lower()
        t_escaped = t.replace('"', '""')
        quoted.append(f'"{t_escaped}"')
    return " AND ".join(quoted)


# ---------------------------------------------------------------------------
# Database connection management
# ---------------------------------------------------------------------------


class Database:
    """Thread-safe SQLite database for session storage.

    Uses WAL mode for concurrent read/write performance.
    All write operations are serialized via a lock.
    Schema v2 adds FTS5 full-text search with jieba segmentation.
    """

    def __init__(self, db_path: Path, jieba_dict_path: str | Path | None = None):
        self._db_path = db_path
        self._lock = Lock()
        self._conn: Optional[sqlite3.Connection] = None
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Load jieba custom dictionary before any FTS operations
        load_jieba_dict(jieba_dict_path)
        self._connect()
        self._init_schema()

    def _connect(self) -> None:
        """Open or create the SQLite database."""
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=10,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._lock:
            cur = self._conn.execute("PRAGMA user_version")
            version = cur.fetchone()[0]

            if version < 1:
                logger.info("Initializing database schema (v1)")
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_key TEXT PRIMARY KEY,
                        created_at  REAL NOT NULL DEFAULT (strftime('%s','now')),
                        updated_at  REAL NOT NULL DEFAULT (strftime('%s','now')),
                        metadata    TEXT DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_key  TEXT NOT NULL REFERENCES sessions(session_key),
                        role         TEXT NOT NULL,
                        content      TEXT DEFAULT '',
                        tool_calls   TEXT DEFAULT NULL,
                        tool_call_id TEXT DEFAULT NULL,
                        reasoning    TEXT DEFAULT NULL,
                        created_at   REAL NOT NULL DEFAULT (strftime('%s','now'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session_time
                        ON messages(session_key, created_at);
                """)
                self._conn.execute("PRAGMA user_version = 1")
                self._conn.commit()
                version = 1

            if version < 2:
                logger.info("Upgrading database schema to v2 (FTS5 + jieba)")
                self._conn.executescript("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        content,
                        reasoning,
                        tokenize='unicode61'
                    );
                """)
                self._conn.execute("PRAGMA user_version = 2")
                self._conn.commit()
                # Backfill existing messages into the FTS index
                self._backfill_fts()

            if version < 3:
                logger.info("Upgrading database schema to v3 (session_id)")
                self._conn.executescript(
                    "ALTER TABLE messages ADD COLUMN session_id TEXT NOT NULL DEFAULT '';"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session "
                    "ON messages(session_key, session_id)"
                )
                # Backfill: assign stable session_id per session_key
                cur = self._conn.execute(
                    "SELECT DISTINCT session_key FROM sessions"
                )
                for row in cur.fetchall():
                    sk = row["session_key"]
                    sid = f"v0_{sk}"
                    self._conn.execute(
                        "UPDATE messages SET session_id = ? WHERE session_key = ?",
                        (sid, sk),
                    )
                self._conn.execute("PRAGMA user_version = 3")
                self._conn.commit()
                version = 3

    # ------------------------------------------------------------------
    # FTS backfill
    # ------------------------------------------------------------------

    def _backfill_fts(self) -> None:
        """Populate messages_fts from all existing messages (schema upgrade)."""
        cur = self._conn.execute("SELECT COUNT(*) FROM messages_fts")
        if cur.fetchone()[0] > 0:
            return  # already populated
        cur = self._conn.execute("SELECT id, content, reasoning FROM messages")
        rows = cur.fetchall()
        for row in rows:
            seg_content = jieba_segment(row["content"])
            seg_reasoning = jieba_segment(row["reasoning"])
            self._conn.execute(
                "INSERT INTO messages_fts (rowid, content, reasoning) VALUES (?, ?, ?)",
                (row["id"], seg_content, seg_reasoning),
            )
        self._conn.commit()
        if rows:
            logger.info("Backfilled %d messages into FTS index", len(rows))

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_key: str) -> Tuple[Dict[str, Any], bool]:
        """Get an existing session or create a new one.

        Uses INSERT OR IGNORE for atomic create-if-not-exists,
        avoiding TOCTOU race conditions entirely.

        Returns:
            (session_dict, created) where created is True if a new row was inserted.
        """
        with self._lock:
            now = time.time()
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_key, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, '{}')",
                (session_key, now, now),
            )
            # Check if INSERT actually inserted (changes() = 1) or was ignored (0)
            cur = self._conn.execute("SELECT changes()")
            inserted = cur.fetchone()[0] > 0
            cur = self._conn.execute(
                "SELECT session_key, created_at, updated_at, metadata "
                "FROM sessions WHERE session_key = ?",
                (session_key,),
            )
            row = cur.fetchone()
            self._conn.commit()
            if row is None:
                raise RuntimeError(
                    f"Failed to create or find session: {session_key}"
                )
            return _row_to_session(row), inserted

    def archive_session(self, session_key: str) -> None:
        """Mark a session as archived by setting metadata.archived_at.

        The session row and all its messages remain in the database.
        Call get_or_create_session_after_archive() to start a fresh session
        with the same key (old messages stay in the DB for reference).
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT metadata FROM sessions WHERE session_key = ?",
                (session_key,),
            )
            row = cur.fetchone()
            if row is None:
                return

            metadata = json.loads(row["metadata"] or "{}")
            metadata["archived_at"] = time.time()

            self._conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE session_key = ?",
                (json.dumps(metadata, ensure_ascii=False), time.time(), session_key),
            )
            self._conn.commit()

    def get_or_create_session_after_archive(
        self, session_key: str
    ) -> Dict[str, Any]:
        """Get existing session; if archived, reset it to a fresh state.

        The old messages remain in the DB (associated with this session_key).
        The session row is updated to remove archived_at and reset metadata.
        This is used after archive_session() to start fresh while preserving
        the archived messages.

        If the session does not exist at all, a new one is created.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_key, created_at, updated_at, metadata "
                "FROM sessions WHERE session_key = ?",
                (session_key,),
            )
            row = cur.fetchone()

            if row:
                metadata = json.loads(row["metadata"] or "{}")
                if "archived_at" not in metadata:
                    return _row_to_session(row)

                # Reset the existing row to a fresh state
                now = time.time()
                self._conn.execute(
                    "UPDATE sessions SET metadata = '{}', "
                    "created_at = ?, updated_at = ? WHERE session_key = ?",
                    (now, now, session_key),
                )
                self._conn.commit()
            else:
                # Session doesn't exist — create it
                now = time.time()
                self._conn.execute(
                    "INSERT INTO sessions (session_key, created_at, updated_at, metadata) "
                    "VALUES (?, ?, ?, '{}')",
                    (session_key, now, now),
                )
                self._conn.commit()

            # Read back to ensure consistency
            cur = self._conn.execute(
                "SELECT session_key, created_at, updated_at, metadata "
                "FROM sessions WHERE session_key = ?",
                (session_key,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    f"Failed to create or find session after archive: {session_key}"
                )
            return _row_to_session(row)

    def update_session_updated_at(self, session_key: str) -> None:
        """Update the updated_at timestamp of a session."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_key = ?",
                (time.time(), session_key),
            )
            self._conn.commit()

    def get_all_session_keys(self) -> List[str]:
        """Return all session keys from the database."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_key FROM sessions ORDER BY updated_at DESC"
            )
            return [row["session_key"] for row in cur.fetchall()]

    def get_all_session_dicts(self) -> List[Dict[str, Any]]:
        """Return all sessions as dicts with metadata parsed, ordered by updated_at DESC."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_key, created_at, updated_at, metadata "
                "FROM sessions ORDER BY updated_at DESC"
            )
            return [_row_to_session(row) for row in cur.fetchall()]

    def update_session_metadata(self, session_key: str, metadata: dict) -> None:
        """Replace the metadata of a session with the given dict.

        Serializes metadata as JSON. Also updates updated_at to the current time.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE session_key = ?",
                (json.dumps(metadata, ensure_ascii=False), time.time(), session_key),
            )
            self._conn.commit()

    def delete_session(self, session_key: str) -> None:
        """Delete a session and its messages (including FTS index)."""
        with self._lock:
            self._delete_messages_for_session(session_key)
            self._conn.execute(
                "DELETE FROM sessions WHERE session_key = ?", (session_key,)
            )
            self._conn.commit()

    def _delete_messages_for_session(self, session_key: str) -> None:
        """Delete messages (incl. FTS index) for a session, keeping session row.
        Caller must hold self._lock.
        """
        self._conn.execute(
            "DELETE FROM messages_fts WHERE rowid IN "
            "(SELECT id FROM messages WHERE session_key = ?)",
            (session_key,),
        )
        self._conn.execute(
            "DELETE FROM messages WHERE session_key = ?", (session_key,)
        )
        self._conn.commit()

    def delete_sessions_older_than(self, cutoff: float) -> int:
        """Delete all sessions with updated_at older than cutoff.

        Returns the number of sessions deleted.
        Messages (including FTS entries) for deleted sessions are also removed.
        """
        with self._lock:
            # Delete FTS entries for old sessions
            self._conn.execute(
                "DELETE FROM messages_fts WHERE rowid IN "
                "(SELECT id FROM messages WHERE session_key IN "
                "(SELECT session_key FROM sessions WHERE updated_at < ?))",
                (cutoff,),
            )
            # Delete messages first to respect foreign keys
            self._conn.execute(
                "DELETE FROM messages WHERE session_key IN "
                "(SELECT session_key FROM sessions WHERE updated_at < ?)",
                (cutoff,),
            )
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_key: str,
        role: str,
        content: str = "",
        *,
        session_id: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> int:
        """Append a message to a session. Returns the message ID.

        Also indexes the message content into the FTS5 full-text search table.
        If the session does not exist, it is automatically created.
        Also updates the session's updated_at timestamp.

        Args:
            session_id: Logical session identifier for isolation.
                        Messages are grouped by session_id so /new can
                        create a fresh context without deleting data.
        """
        with self._lock:
            now = time.time()

            # Ensure session exists (avoid foreign key IntegrityError)
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_key, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, '{}')",
                (session_key, now, now),
            )

            tool_calls_json = (
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
            )
            self._conn.execute(
                "INSERT INTO messages "
                "(session_key, role, content, tool_calls, tool_call_id, reasoning, created_at, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_key,
                    role,
                    content,
                    tool_calls_json,
                    tool_call_id,
                    reasoning,
                    now,
                    session_id,
                ),
            )
            cur = self._conn.execute("SELECT last_insert_rowid()")
            rowid = cur.fetchone()[0]

            # Index into FTS5
            seg_content = jieba_segment(content)
            seg_reasoning = jieba_segment(reasoning)
            self._conn.execute(
                "INSERT INTO messages_fts (rowid, content, reasoning) VALUES (?, ?, ?)",
                (rowid, seg_content, seg_reasoning),
            )

            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_key = ?",
                (now, session_key),
            )
            self._conn.commit()
            return rowid

    def get_messages(
        self, session_key: str, session_id: str = ""
    ) -> List[Dict[str, Any]]:
        """Get messages for a session (optionally filtered by session_id).

        When session_id is provided, only messages with that session_id are
        returned. This enables session-level isolation: /new creates a fresh
        session_id so old messages are excluded without being deleted.
        """
        with self._lock:
            if session_id:
                cur = self._conn.execute(
                    "SELECT role, content, tool_calls, tool_call_id, reasoning, created_at "
                    "FROM messages WHERE session_key = ? AND session_id = ? "
                    "ORDER BY created_at ASC, id ASC",
                    (session_key, session_id),
                )
            else:
                cur = self._conn.execute(
                    "SELECT role, content, tool_calls, tool_call_id, reasoning, created_at "
                    "FROM messages WHERE session_key = ? "
                    "ORDER BY created_at ASC, id ASC",
                    (session_key,),
                )
            return [_row_to_message(row) for row in cur.fetchall()]

    def get_message_count(self, session_key: str, session_id: str = "") -> int:
        """Return the number of messages in a session (optionally filtered by session_id)."""
        with self._lock:
            if session_id:
                cur = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM messages WHERE session_key = ? AND session_id = ?",
                    (session_key, session_id),
                )
            else:
                cur = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM messages WHERE session_key = ?",
                    (session_key,),
                )
            return cur.fetchone()["cnt"]

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    def search_messages(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search messages using FTS5 with jieba segmentation.

        Args:
            query: Search query in Chinese, English, or mixed.
            limit: Maximum results (default 5, max 50).

        Returns:
            List of dicts with keys: session_key, role, content (truncated),
            created_at, rank.
        """
        segmented = jieba_segment(query)
        fts_query = _fts_escape(segmented)
        if not fts_query:
            return []

        limit = min(max(limit, 1), 50)

        with self._lock:
            try:
                cur = self._conn.execute(
                    """SELECT m.session_key, m.role, m.content, m.created_at,
                              rank
                       FROM messages_fts f
                       JOIN messages m ON m.id = f.rowid
                       WHERE messages_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, limit),
                )
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 query error: %s (query: %s)", exc, fts_query)
                return []

            results: List[Dict[str, Any]] = []
            for row in cur.fetchall():
                results.append({
                    "session_key": row["session_key"],
                    "role": row["role"],
                    "content": (row["content"] or "")[:500],
                    "created_at": row["created_at"],
                    "rank": row["rank"],
                })
        return results

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    def import_messages(
        self,
        session_key: str,
        messages: List[Dict[str, Any]],
        *,
        created_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Import a batch of messages into a session (for migration).

        Each message gets a slightly incremented timestamp so that
        ordering within a batch is preserved. If created_at is provided,
        it is used as the base timestamp for the first message.
        Also indexes imported messages into the FTS5 table.

        Returns the number of messages imported.
        """
        base_ts = created_at if created_at is not None else time.time()

        with self._lock:
            # Create or get session
            cur = self._conn.execute(
                "SELECT session_key FROM sessions WHERE session_key = ?",
                (session_key,),
            )
            if cur.fetchone() is None:
                self._conn.execute(
                    "INSERT INTO sessions (session_key, created_at, updated_at, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_key,
                        base_ts,
                        base_ts,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )

            count = 0
            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                tc_id = msg.get("tool_call_id")
                reasoning = msg.get("reasoning_content")

                tool_calls_json = (
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
                )

                # Increment timestamp slightly per message to preserve order
                msg_ts = base_ts + (i * 0.001)

                self._conn.execute(
                    "INSERT INTO messages "
                    "(session_key, role, content, tool_calls, tool_call_id, reasoning, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_key, role, content, tool_calls_json, tc_id, reasoning, msg_ts),
                )

                # Index into FTS5
                seg_content = jieba_segment(content)
                seg_reasoning = jieba_segment(reasoning)
                # last_insert_rowid() is connection-scoped, safe in this loop
                rowid_cur = self._conn.execute("SELECT last_insert_rowid()")
                msg_rowid = rowid_cur.fetchone()[0]
                self._conn.execute(
                    "INSERT INTO messages_fts (rowid, content, reasoning) VALUES (?, ?, ?)",
                    (msg_rowid, seg_content, seg_reasoning),
                )

                count += 1

            # Update session updated_at to reflect the import
            if count > 0:
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_key = ?",
                    (base_ts + (count * 0.001), session_key),
                )

            self._conn.commit()
            return count

    def integrity_check(self) -> List[str]:
        """Run PRAGMA integrity_check. Returns empty list if OK, error messages otherwise."""
        with self._lock:
            cur = self._conn.execute("PRAGMA integrity_check")
            results = [row[0] for row in cur.fetchall()]
            return [r for r in results if r != "ok"]

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        if self._conn:
            with self._lock:
                if self._conn:
                    self._conn.close()
                    self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_session(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a SQLite row to a session dict."""
    return {
        "session_key": row["session_key"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "metadata": json.loads(row["metadata"] or "{}"),
    }


def _row_to_message(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a SQLite row to a message dict suitable for LLM API.

    Preserves null/None fields accurately — some models (e.g. DeepSeek
    thinking mode) require content=None to remain None instead of "".
    """
    raw_content = row["content"]
    msg: Dict[str, Any] = {
        "role": row["role"],
        "content": raw_content if raw_content is not None else None,
    }
    if row["tool_calls"] is not None:
        msg["tool_calls"] = json.loads(row["tool_calls"])
    if row["tool_call_id"] is not None:
        msg["tool_call_id"] = row["tool_call_id"]
    if row["reasoning"] is not None:
        msg["reasoning_content"] = row["reasoning"]
    return msg
