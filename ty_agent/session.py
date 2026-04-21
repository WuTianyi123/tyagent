"""Session management for ty-agent."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        if session_key in self._sessions:
            self._sessions[session_key].clear()
            self._save(session_key)

    def delete(self, session_key: str) -> None:
        self._sessions.pop(session_key, None)
        if self._sessions_dir:
            path = self._sessions_dir / f"{session_key}.json"
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
        path = self._sessions_dir / f"{session_key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

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
