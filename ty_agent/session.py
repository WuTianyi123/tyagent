"""Session management for ty-agent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Safe characters for session key filename
_SAFE_SESSION_KEY_RE = re.compile(r"^[a-zA-Z0-9_:-]+$")


def _sanitize_session_key(session_key: str) -> str:
    """Sanitize session key for safe filesystem use.

    If the key contains unsafe characters, hash it to prevent path traversal.
    """
    if _SAFE_SESSION_KEY_RE.match(session_key):
        return session_key
    return hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]


@dataclass
class Session:
    """A conversation session."""

    session_key: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **extras) -> None:
        msg: Dict[str, Any] = {"role": role, "content": content, **extras}
        self.messages.append(msg)
        self.updated_at = time.time()

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": self.session_key,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        return cls(
            session_key=data["session_key"],
            messages=data.get("messages", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )


class SessionStore:
    """In-memory session store with optional JSON persistence."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        self._sessions: Dict[str, Session] = {}
        self._sessions_dir = sessions_dir
        if sessions_dir:
            sessions_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    def get(self, session_key: str) -> Session:
        if session_key not in self._sessions:
            self._sessions[session_key] = Session(session_key=session_key)
        return self._sessions[session_key]

    def reset(self, session_key: str) -> None:
        """Clear messages in the session but keep the session object."""
        if session_key in self._sessions:
            self._sessions[session_key].clear()
            self._save(session_key)

    def archive(self, session_key: str) -> None:
        """Archive a session: rename the file with a timestamp suffix,
        remove from active sessions. The next get() for this key will
        create a fresh session. The archived file is preserved on disk.

        This is the preferred alternative to reset() — it keeps the
        conversation history for future reference instead of clearing it.
        """
        session = self._sessions.pop(session_key, None)
        if session is None or not self._sessions_dir:
            return

        # Rename the file with timestamp suffix
        filename = _sanitize_session_key(session_key) + ".json"
        old_path = self._sessions_dir / filename
        if old_path.exists():
            import time
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(session.updated_at))
            archive_name = f"{_sanitize_session_key(session_key)}__archived_{ts}.json"
            archive_path = self._sessions_dir / archive_name
            try:
                old_path.rename(archive_path)
                logger.info("Archived session %s → %s", session_key, archive_name)
            except OSError as exc:
                logger.warning("Failed to archive session %s: %s", session_key, exc)

    def delete(self, session_key: str) -> None:
        self._sessions.pop(session_key, None)
        if self._sessions_dir:
            filename = _sanitize_session_key(session_key) + ".json"
            path = self._sessions_dir / filename
            if path.exists():
                path.unlink()

    def save(self, session_key: str) -> None:
        self._save(session_key)

    def _save(self, session_key: str) -> None:
        if not self._sessions_dir:
            return
        session = self._sessions.get(session_key)
        if not session:
            return
        filename = _sanitize_session_key(session_key) + ".json"
        path = self._sessions_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        # Restrict permissions so only owner can read
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _load_all(self) -> None:
        if not self._sessions_dir:
            return
        for path in self._sessions_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = Session.from_dict(data)
                self._sessions[session.session_key] = session
            except Exception as exc:
                logger.warning("Failed to load session from %s: %s", path, exc)

    def all_session_keys(self) -> List[str]:
        return list(self._sessions.keys())

    def prune_old_sessions(self, max_age_days: int = 90) -> int:
        """Remove sessions older than max_age_days."""
        if max_age_days <= 0:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        for key in list(self._sessions.keys()):
            if self._sessions[key].updated_at < cutoff:
                self.delete(key)
                removed += 1
        return removed
