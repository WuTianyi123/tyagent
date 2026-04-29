"""Session management for tyagent.

Lightweight Session and SessionStore backed by SQLite (db.py).
Messages are persisted immediately on add_message() — no explicit save() needed.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tyagent.db import Database

logger = logging.getLogger(__name__)

# Sentinel for "no store provided"
_UNSET = object()

class SessionError(Exception):
    """Raised when a session operation fails."""
    pass


@dataclass
class Session:
    """A conversation session backed by database.

    This is a lightweight data object. Messages are lazily loaded from
    the database via the .messages property. The .add_message() method
    delegates to the store for immediate persistence.
    """

    session_key: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _store: Union[SessionStore, object] = field(default=_UNSET, repr=False, compare=False)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get all messages for this session from the database.

        Only messages with the current session_id are returned.
        This enables session isolation — /new generates a fresh session_id
        so old messages are excluded without being deleted.
        """
        if self._store is not _UNSET:
            current_sid = self.metadata.get("current_session_id", "")
            if current_sid:
                return self._store.get_messages(self.session_key, session_id=current_sid)
            return self._store.get_messages(self.session_key)
        return []

    def add_message(
        self,
        role: str,
        content: Optional[str] = "",
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> int:
        """Append a message to this session. Persisted immediately.

        Messages are tagged with the current session_id (from metadata)
        for isolation across /new boundaries.

        Returns the message ID.

        Raises:
            SessionError: If this session was not created by a SessionStore.
        """
        if self._store is _UNSET:
            raise SessionError(
                "Session has no store backing — use SessionStore.get() "
                "or SessionStore.add_message() instead."
            )
        current_sid = self.metadata.get("current_session_id", "")
        msg_id = self._store.add_message(
            self.session_key, role, content,
            session_id=current_sid,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            reasoning=reasoning,
        )
        self.updated_at = time.time()
        return msg_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session metadata to a dict.

        NOTE: Messages are NOT included — they live in the database.
        Use SessionStore.get_messages() to retrieve them.
        """
        return {
            "session_key": self.session_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        """Create a Session from a metadata dict.

        NOTE: The returned Session has no _store reference, so .messages
        will return [] and .add_message() will raise SessionError.
        Use SessionStore.get() to obtain a fully functional Session.
        """
        session_key = data.get("session_key")
        if not session_key:
            raise SessionError("from_dict requires a non-empty 'session_key'")
        return cls(
            session_key=session_key,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )


class SessionStore:
    """Session store backed by SQLite database.

    All session and message data is persisted in a SQLite database.
    Messages are written immediately on add_message() — no explicit save() needed.

    Use close() to release resources. Supports context manager protocol.
    """

    def __init__(self, sessions_dir: Optional[Path] = None):
        if sessions_dir:
            self._db_dir = sessions_dir
            self._temp_dir: Optional[Path] = None
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="tyagent_session_"))
            self._db_dir = self._temp_dir
        self._db = Database(self._db_dir / "sessions.db")

    def close(self) -> None:
        """Close the database and clean up temporary directories if any."""
        self._db.close()
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug("Cleaned up temporary session dir: %s", self._temp_dir)
            except OSError as exc:
                logger.warning("Failed to clean up temp dir %s: %s", self._temp_dir, exc)

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def db(self) -> Database:
        return self._db

    def _build_session(self, session_dict: Dict[str, Any]) -> Session:
        """Create a Session with a back-reference to this store.

        Args:
            session_dict: Must contain keys: session_key, created_at,
                updated_at, metadata.

        Raises:
            KeyError: If any required key is missing from session_dict.
        """
        return Session(
            session_key=session_dict["session_key"],
            created_at=session_dict["created_at"],
            updated_at=session_dict["updated_at"],
            metadata=session_dict["metadata"],
            _store=self,
        )

    def get(self, session_key: str) -> Session:
        """Get or create a session.

        Returns a Session object with lazy message loading via .messages property.

        On first creation, generates a current_session_id UUID so messages
        are properly isolated by session identity.

        Raises:
            SessionError: If session_key is empty.
        """
        if not session_key:
            raise SessionError("session_key must not be empty")
        session_dict, created = self._db.get_or_create_session(session_key)
        if created:
            # Initialize current_session_id for session isolation
            import uuid
            metadata = session_dict["metadata"]
            metadata["current_session_id"] = uuid.uuid4().hex[:16]
            self._db.update_session_metadata(session_key, metadata)
            session_dict["metadata"] = metadata
        return self._build_session(session_dict)

    def add_message(
        self,
        session_key: str,
        role: str,
        content: Optional[str] = "",
        *,
        session_id: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> int:
        """Append a message to a session. Persisted immediately.

        If session_id is not provided, auto-injects the current session's
        current_session_id from metadata for isolation.

        Args:
            session_id: Logical session identifier for isolation.
        Returns the message ID.
        """
        if not session_id:
            session_dict, _ = self._db.get_or_create_session(session_key)
            session_id = session_dict["metadata"].get("current_session_id", "")
        return self._db.add_message(
            session_key, role, content,
            session_id=session_id,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            reasoning=reasoning,
        )

    def get_messages(self, session_key: str, session_id: str = "") -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by creation time.

        Args:
            session_key: The session key.
            session_id: Optional filter — only messages with this session_id
                        are returned. Used for session isolation on /new.
        """
        return self._db.get_messages(session_key, session_id=session_id)

    def get_message_count(self, session_key: str, session_id: str = "") -> int:
        """Return the number of messages in a session (optionally filtered by session_id)."""
        return self._db.get_message_count(session_key, session_id=session_id)

    def archive(self, session_key: str) -> None:
        """Archive a session: mark as archived in metadata.

        The session's messages remain in the database.
        Also saves the previous session_start so the archived state
        can be distinguished from the fresh one.
        """
        self._db.archive_session(session_key)

    def freshen_session(self, session_key: str) -> None:
        """Reset a session to a fresh state while preserving old messages.

        Generates a new current_session_id so subsequent .messages loads
        only return messages tagged with the new ID. Old messages remain
        in the database with the previous session_id.
        """
        import uuid
        session_dict, _ = self._db.get_or_create_session(session_key)
        metadata = session_dict["metadata"]
        # Save previous session_id for historical reference
        old_sid = metadata.get("current_session_id", "")
        if old_sid:
            metadata["prev_session_id"] = old_sid
        metadata["current_session_id"] = uuid.uuid4().hex[:16]
        self._db.update_session_metadata(session_key, metadata)

    def get_or_create_after_archive(self, session_key: str) -> Session:
        """Get a fresh session after archiving the old one.

        The old session's messages remain in the database (associated with
        the same session_key). The session metadata is reset to empty.
        """
        session_dict = self._db.get_or_create_session_after_archive(session_key)
        return self._build_session(session_dict)

    def reset(self, session_key: str) -> None:
        """Legacy reset — delegates to archive."""
        self.archive(session_key)

    def save(self, session_key: str) -> None:
        """No-op: messages are persisted immediately."""
        pass

    def delete(self, session_key: str) -> None:
        """Delete a session and its messages from the database."""
        self._db.delete_session(session_key)

    def all_session_keys(self) -> List[str]:
        """Return all session keys."""
        return self._db.get_all_session_keys()

    def prune_old_sessions(self, max_age_days: int = 90) -> int:
        """Remove sessions older than max_age_days.

        Uses a single efficient SQL DELETE query.
        Messages for pruned sessions are also removed.
        """
        if max_age_days <= 0:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        return self._db.delete_sessions_older_than(cutoff)

    # ------------------------------------------------------------------
    # Resume-pending / suspend recovery
    # ------------------------------------------------------------------

    def mark_resume_pending(self, session_key: str, reason: str = "restart_timeout") -> bool:
        """Mark a session as ready for resume, unless it's already suspended.

        Returns True if resume_pending was set, False if session doesn't exist or is suspended.
        """
        all_keys = self._db.get_all_session_keys()
        if session_key not in all_keys:
            return False
        session_dict, _ = self._db.get_or_create_session(session_key)
        metadata = session_dict["metadata"]
        if metadata.get("suspended"):
            return False
        metadata["resume_pending"] = True
        metadata["resume_reason"] = reason
        metadata["resume_marked_at"] = time.time()
        self._db.update_session_metadata(session_key, metadata)
        return True

    def clear_resume_pending(self, session_key: str) -> bool:
        """Clear resume_pending flags from session metadata.

        Returns True if the flag was present and cleared, False otherwise.
        """
        session_dict = self._db.get_or_create_session(session_key)[0]
        metadata = session_dict["metadata"]
        if "resume_pending" not in metadata:
            return False
        metadata.pop("resume_pending", None)
        metadata.pop("resume_reason", None)
        metadata.pop("resume_marked_at", None)
        self._db.update_session_metadata(session_key, metadata)
        return True

    def suspend_session(self, session_key: str, reason: str = "crash_recovery") -> bool:
        """Suspend a session, marking it as explicitly suspended.

        Returns True.
        """
        session_dict = self._db.get_or_create_session(session_key)[0]
        metadata = session_dict["metadata"]
        metadata["suspended"] = True
        metadata["suspend_reason"] = reason
        metadata["suspend_at"] = time.time()
        self._db.update_session_metadata(session_key, metadata)
        return True

    def suspend_recently_active(self, max_age_seconds: int = 120) -> int:
        """Suspend all recently active sessions that are not already suspended or resume_pending.

        Args:
            max_age_seconds: Sessions with updated_at within this many seconds
                            from now are considered "recently active".

        Returns:
            Number of sessions suspended.
        """
        cutoff = time.time() - max_age_seconds
        count = 0
        for session_dict in self._db.get_all_session_dicts():
            metadata = session_dict["metadata"]
            if metadata.get("resume_pending") or metadata.get("suspended"):
                continue
            if session_dict["updated_at"] >= cutoff:
                metadata["suspended"] = True
                metadata["suspend_reason"] = "crash_recovery"
                metadata["suspend_at"] = time.time()
                self._db.update_session_metadata(session_dict["session_key"], metadata)
                count += 1
        return count

    def is_suspended(self, session_key: str) -> bool:
        """Check if a session is currently suspended."""
        session_dict = self._db.get_or_create_session(session_key)[0]
        return bool(session_dict["metadata"].get("suspended"))

    def is_resume_pending(self, session_key: str) -> bool:
        """Check if a session has a pending resume flag."""
        session_dict = self._db.get_or_create_session(session_key)[0]
        return bool(session_dict["metadata"].get("resume_pending"))

    def integrity_check(self) -> List[str]:
        """Check database integrity. Returns list of errors (empty if OK)."""
        return self._db.integrity_check()
