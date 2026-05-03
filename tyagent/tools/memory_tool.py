"""Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes (environment facts, project conventions, tool quirks)
  - USER.md: what the agent knows about the user (preferences, communication style)

Both are injected into the system prompt at the gateway level.
Mid-session writes update files on disk immediately (durable).

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove
- replace/remove use short unique substring matching (not full text or IDs)
- Module-level MemoryStore singleton set by gateway at startup
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from tyagent.tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

ENTRY_DELIMITER = "\n§\n"

# Regex for extracting [[keyword]] wiki links from entry content
_WIKI_LINK_RE = re.compile(r'\[\[([^\]]+)\]\]')

# ---------------------------------------------------------------------------
# Injection / exfiltration scan
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"you\s+are\s+now\s+", "role_hijack"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|limits|rules)", "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    # Persistence via shell rc
    (r"authorized_keys", "ssh_backdoor"),
    (r"\$HOME/\.ssh|\~/\.ssh", "ssh_access"),
    (r"\$HOME/\.hermes/\.env|\~/\.hermes/\.env", "hermes_env"),
]

_INVISIBLE_CHARS = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return (
                f"Blocked: content contains invisible unicode character "
                f"U+{ord(char):04X} (possible injection)."
            )

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return (
                f"Blocked: content matches threat pattern '{pid}'. "
                f"Memory entries are injected into the system prompt and must not "
                f"contain injection or exfiltration payloads."
            )

    return None


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """Bounded curated memory with file persistence.

    Maintains two stores:
      - memory_entries: agent's personal notes
      - user_entries: user profile information

    Both are persisted as ``.md`` files with §-delimited entries.
    """

    def __init__(
        self,
        memories_dir: str | Path,
        memory_char_limit: int = 2000,
        user_char_limit: int = 1000,
    ):
        self.memories_dir = Path(memories_dir)
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Backlink index: {keyword: [(target, entry_index), ...]}
        # Indicates which entries contain [[keyword]] links.
        self._backlinks: Dict[str, List[tuple[str, int]]] = {}
        self.load_from_disk()

    # -- Path helpers -------------------------------------------------------

    def _path_for(self, target: str) -> Path:
        if target == "user":
            return self.memories_dir / "USER.md"
        return self.memories_dir / "MEMORY.md"

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]) -> None:
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    # -- Disk I/O -----------------------------------------------------------

    def load_from_disk(self) -> None:
        """Load entries from MEMORY.md and USER.md on disk."""
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_file(self._path_for("memory"))
        self.user_entries = self._read_file(self._path_for("user"))
        # Deduplicate (preserve order, keep first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))
        self._rebuild_backlinks()

    def _reload_target(self, target: str) -> None:
        """Re-read entries from disk for a target (called under lock)."""
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))
        self._set_entries(target, fresh)
        self._rebuild_backlinks()

    def save_to_disk(self, target: str) -> None:
        """Persist entries of *target* to the appropriate file."""
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _rebuild_backlinks(self) -> None:
        """Scan all entries for [[keyword]] patterns and build backlink index."""
        self._backlinks.clear()
        for target in ("memory", "user"):
            entries = self._entries_for(target)
            for idx, entry in enumerate(entries):
                for match in _WIKI_LINK_RE.finditer(entry):
                    keyword = match.group(1).strip()
                    if keyword:
                        kw_lower = keyword.lower()
                        if kw_lower not in self._backlinks:
                            self._backlinks[kw_lower] = []
                        self._backlinks[kw_lower].append((target, idx))

    def _referenced_by(self, target: str, index: int) -> List[Dict[str, Any]]:
        """Find entries that link *to* the entry at ``(target, index)``.

        Any entry containing ``[[keyword]]`` where *keyword* appears as a
        substring of this entry's content is considered a backlink.
        Self-references are excluded.
        Returns a list of {target, index, summary}.
        """
        entries = self._entries_for(target)
        if index < 0 or index >= len(entries):
            return []
        entry_text = entries[index].lower()

        results: List[Dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        for kw_lower, refs in self._backlinks.items():
            for (ref_target, ref_idx) in refs:
                # Skip self-reference
                if ref_target == target and ref_idx == index:
                    continue
                # Does the [[keyword]] from another entry match our content?
                if kw_lower in entry_text:
                    key = (ref_target, ref_idx)
                    if key not in seen:
                        seen.add(key)
                        ref_entries = self._entries_for(ref_target)
                        if ref_idx < len(ref_entries):
                            results.append({
                                "target": ref_target,
                                "index": ref_idx,
                                "summary": self._entry_summary(ref_entries[ref_idx]),
                            })

        return results

    def _success_response(self, target: str, message: str = "") -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        resp: Dict[str, Any] = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    # -- File locking -------------------------------------------------------

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Use open() with "a" so creation + append mode avoids edge cases
        # on read-only zero-byte files on some systems
        fd_obj = open(lock_path, "a+")
        try:
            fcntl.flock(fd_obj, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd_obj, fcntl.LOCK_UN)
            fd_obj.close()

    # -- Read / Write helpers -----------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Split a memory file into entries by ENTRY_DELIMITER."""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _atomic_replace(tmp_path: Path, target: Path) -> None:
        """Atomically move tmp_path onto target, preserving symlinks.

        ``os.replace`` atomically swaps tmp into place.  When *target* is a
        symlink, the symlink itself gets replaced with a regular file —
        silently detaching managed deployments that symlink MEMORY.md/USER.md
        from their profile home.  This resolves the symlink first so the
        real file is replaced in-place while the symlink survives.
        """
        real_path = os.path.realpath(target) if os.path.islink(target) else target
        os.replace(tmp_path, real_path)

    @staticmethod
    def _write_file(path: Path, entries: List[str]) -> None:
        """Atomic write via tempfile + fsync + atomic rename (symlink-safe)."""
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".mem_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            MemoryStore._atomic_replace(Path(tmp_path), path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -- Public API: add / replace / remove ---------------------------------

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = self._entries_for(target)

            if content in entries:
                return self._success_response(
                    target, "Entry already exists (no duplicate added)."
                )

            new_total = len(ENTRY_DELIMITER.join(entries + [content]))
            limit = self._char_limit(target)

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)
            self._rebuild_backlinks()

        return self._success_response(target, "Entry added.")

    def replace(
        self, target: str, old_text: str, new_content: str
    ) -> Dict[str, Any]:
        """Find entry containing *old_text*, replace with *new_content*."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {
                "success": False,
                "error": "new_content cannot be empty. Use 'remove' to delete entries.",
            }

        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {
                    "success": False,
                    "error": f"No entry matched '{old_text}'.",
                }

            if len(matches) > 1:
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [
                        e[:80] + ("..." if len(e) > 80 else "") for _, e in matches
                    ]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }

            idx = matches[0][0]
            limit = self._char_limit(target)

            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at "
                        f"{new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)
            self._rebuild_backlinks()

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing *old_text*."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {
                    "success": False,
                    "error": f"No entry matched '{old_text}'.",
                }

            if len(matches) > 1:
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [
                        e[:80] + ("..." if len(e) > 80 else "") for _, e in matches
                    ]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)
            self._rebuild_backlinks()

        return self._success_response(target, "Entry removed.")

    # -- System prompt formatting -------------------------------------------

    def get_all_formatted(self) -> str:
        """Render both stores as a compact summary index for system prompt injection.

        Each entry becomes one bullet point with first ~100 chars.
        LLM can use ``expand(keyword)`` to retrieve full entries.

        Returns empty string if both stores are empty.
        """
        parts: List[str] = []
        for target, label in (("user", "USER PROFILE (who the user is)"), ("memory", "MEMORY (your notes)")):
            entries = self._entries_for(target)
            if not entries:
                continue
            limit = self._char_limit(target)
            current = self._char_count(target)
            pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
            header = f"{label} [{pct}% — {current:,}/{limit:,} chars | {len(entries)} entries]"
            separator = "─" * 44
            bullets = "\n".join(f"  · {self._entry_summary(e)}" for e in entries)
            parts.append(f"{separator}\n{header}\n{separator}\n{bullets}")

        return "\n\n".join(parts)

    @staticmethod
    def _entry_summary(entry: str, max_chars: int = 100) -> str:
        """Return a one-line summary of an entry."""
        text = entry.replace("\n", " ↵ ")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def expand(self, target: str | None, keyword: str) -> Dict[str, Any]:
        """Find entries containing *keyword* and return full content.

        Case-insensitive substring matching (consistent with replace/remove).
        When *target* is None, searches both stores.

        Each match includes:
        - ``referenced_by``: other entries that contain ``[[keyword]]`` linking back
        - ``links_to``: keywords this entry links *to* via ``[[keyword]]``
        """
        keyword = keyword.strip()
        if not keyword:
            return {"success": False, "error": "keyword cannot be empty."}

        stores = ["memory", "user"] if target is None else [target]
        all_matches: List[Dict[str, Any]] = []

        # When searching a single store, also refresh the other store's
        # in-memory data so cross-store referenced_by is accurate.
        if target is not None:
            other = "user" if target == "memory" else "memory"
            with self._file_lock(self._path_for(other)):
                self._reload_target(other)

        for t in stores:
            with self._file_lock(self._path_for(t)):
                self._reload_target(t)
                entries = self._entries_for(t)
                for i, entry in enumerate(entries):
                    if not (keyword.lower() in entry.lower()):
                        continue
                    # Extract [[keyword]] patterns from this entry (links_to)
                    links_to = [
                        {"keyword": m.group(1).strip()}
                        for m in _WIKI_LINK_RE.finditer(entry)
                        if m.group(1).strip()
                    ]
                    all_matches.append({
                        "target": t,
                        "index": i,
                        "content": entry,
                        "char_count": len(entry),
                        "summary": self._entry_summary(entry),
                        "links_to": links_to,
                        "referenced_by": self._referenced_by(t, i),
                    })

        resp: Dict[str, Any] = {
            "success": True,
            "keyword": keyword,
            "matches": all_matches,
            "count": len(all_matches),
        }
        if not all_matches:
            resp["message"] = f"No entries matched '{keyword}'."
        return resp

    def read(self, target: str | None = None) -> Dict[str, Any]:
        """Return the current memory index as structured data (summaries only).

        If target is None, returns both stores. Otherwise returns the specified store.
        """
        stores_to_show = ["memory", "user"] if target is None else [target]

        result: Dict[str, Any] = {"success": True, "stores": {}}
        for t in stores_to_show:
            with self._file_lock(self._path_for(t)):
                self._reload_target(t)
                entries = self._entries_for(t)
                result["stores"][t] = {
                    "entries": [self._entry_summary(e) for e in entries],
                    "count": len(entries),
                    "usage": f"{self._char_count(t):,}/{self._char_limit(t):,} chars",
                }
        return result


# ---------------------------------------------------------------------------
# Module-level singleton (set by gateway at startup)
# ---------------------------------------------------------------------------

_global_store: Optional[MemoryStore] = None


def set_store(store: MemoryStore) -> None:
    global _global_store
    _global_store = store


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def _handle_memory(args: Dict[str, Any]) -> str:
    """Dispatch memory action to the global MemoryStore."""
    store = _global_store
    if store is None:
        return tool_error(
            "Memory is not available. It may be disabled or the store not initialized.",
        )

    action = args.get("action", "").strip()
    target_raw = args.get("target")
    target = target_raw.strip() if target_raw else target_raw
    content = args.get("content")
    old_text = args.get("old_text")

    if target is not None and target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.")

    if action == "add":
        if not content:
            return tool_error("content is required for 'add' action.")
        if not target:
            return tool_error("target is required for 'add' action.")
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.")
        if not content:
            return tool_error("content is required for 'replace' action.")
        if not target:
            return tool_error("target is required for 'replace' action.")
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.")
        if not target:
            return tool_error("target is required for 'remove' action.")
        result = store.remove(target, old_text)

    elif action == "expand":
        keyword = args.get("keyword", "").strip()
        if not keyword:
            return tool_error("keyword is required for 'expand' action.")
        # target is optional for expand — None means search both stores
        expand_target = target if target in ("memory", "user") else None
        result = store.expand(expand_target, keyword)

    elif action == "read":
        # target is optional for read — None means both stores
        if target is not None and target not in ("memory", "user"):
            return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user', or omit to read both.")
        result = store.read(target)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove, expand, read")

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is automatically injected into the system prompt in future turns, "
        "so keep it compact and focused on facts that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past conversations.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is — name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes — environment facts, project conventions, tool quirks, lessons learned\n\n"
            "ACTIONS: add (new entry), replace (update existing — old_text identifies it), "
            "remove (delete — old_text identifies it), "
            "expand (show full content of entries matching keyword — omitting target searches both stores), "
            "read (return the current memory index as structured data).\n\n"
            "KEYWORD: Required for 'expand'. Case-insensitive substring search within entries.\n\n"
            "TARGET: 'memory' or 'user'. Optional for expand and read (searches both when omitted). Required for add, replace, and remove.\n\n"
            "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "expand", "read"],
                "description": "The action to perform.",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store. Required for add/replace/remove; optional for expand (searches both) and read (returns both).",
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'.",
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove.",
            },
            "keyword": {
                "type": "string",
                "description": "Keyword to search for in memory entries. Required for 'expand' action.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="memory",
    schema=MEMORY_SCHEMA,
    handler=_handle_memory,
    description="Persistent memory across sessions (agent notes + user profile)",
    emoji="🧠",
)
