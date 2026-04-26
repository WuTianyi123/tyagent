# Plan: Persisent Memory System + Chinese-capable Session Search

> **Goal:** Add persistent cross-session curated memory and Chinese-capable conversation search to tyagent.
>
> **Architecture:** Two-tier. Hot memory (entry-delimited `.md` files) for facts that must always be available in the system prompt. Cold search (jieba + FTS5) for ad-hoc retrieval of past conversations. Memory tool
> records facts; search tool retrieves past sessions.
>
> **Tech Stack:** SQLite FTS5, jieba 0.42+, fcntl file locking, §-delimited entry format.
>
> **Execution Flow:** Each step: implement → test pass → subagent blind review → fix if issues → repeat until 2 consecutive clean rounds → next step. After all steps: final full blind review, 2 clean rounds → done.

---

## Execution Order (Renumbered)

These steps MUST be executed in this order because earlier steps provide dependencies for later ones:

| # | Step | Files | Depends On |
|---|------|-------|-----------|
| 1 | jieba dependency | `pyproject.toml`, `~/.tyagent/memories/jieba_dict.txt` | nothing |
| 2 | MemoryStore + tool | `tyagent/tools/memory_tool.py`, `tools/__init__.py` | Step 1 |
| 3 | Inject memory into system prompt | `tyagent/gateway.py` | Step 2 |
| 4 | FTS5 + jieba in db.py | `tyagent/db.py` | Step 1 |
| 5 | Search tool | `tyagent/tools/search_tool.py`, `tools/__init__.py` | Step 4 |
| 6 | Memory tests | `tests/test_memory_tool.py` | Step 2 |
| 7 | Search tests | `tests/test_search_tool.py` | Step 4, 5 |

---

## Step 1: jieba dependency + custom dictionary

**Objective:** Add jieba to project dependencies and create a domain-specific dictionary for correct Chinese word segmentation.

**Files:**
- Modify: `pyproject.toml`
- Create (during gateway runtime): `~/.tyagent/memories/jieba_dict.txt`

### pyproject.toml

Add `"jieba>=0.42"` to the `dependencies` list:

```toml
dependencies = [
    "aiohttp>=3.9",
    "httpx>=0.27",
    "lark-oapi>=1.5.5",
    "pyyaml>=6.0",
    "qrcode>=8.2",
    "jieba>=0.42",
]
```

### Custom dictionary (runtime)

A default custom dictionary is shipped as a constant in `db.py` and written to disk by `Database._init_schema()` or during `Database.__init__()`. This happens once, on first schema initialization.

Location: `{db_path.parent}/memories/jieba_dict.txt` (where `db_path` is the SQLite database path, default `~/.tyagent/sessions/tyagent.db`, so dict goes to `~/.tyagent/sessions/memories/jieba_dict.txt`).

Actually, better: store it alongside the memory files at `{config.home_dir}/memories/jieba_dict.txt`. The dictionary file is logically part of the memory system, not the DB. The `Database` class loads jieba dict when its module is imported (first reference to `jieba_segment`), and the dict path is resolved relative to `config.home_dir`.

Default dictionary content as a Python constant:
```python
DEFAULT_JIEBA_DICT = """\
tyagent 5
agent-browser 5
飞书 5
跨session 5
双向链接 3
记忆系统 3
会话存储 3
消息网关 3
上下文压缩 3
gRPC 5
"""
```

**Important:** `pyproject.toml` should be modified BEFORE any code using jieba is written, because `uv sync` needs to install the dependency.

**Verification:**
- `uv sync` completes successfully
- `python3 -c "import jieba; print(jieba.__version__)"` prints `0.42.1`
- `jieba.load_userdict(...)` on the custom dict succeeds
- Custom terms like "飞书" are segmented as single tokens, not split into "飞" + "书"

---

## Step 2: `tyagent/tools/memory_tool.py` (NEW)

**Objective:** A MemoryStore class + tool registration for durable curated memory. Provides `memory` tool with actions: add, replace, remove.

**Files:**
- Create: `tyagent/tools/memory_tool.py`
- Modify: `tyagent/tools/__init__.py` (add import line)

### Import conventions

Use **absolute imports** matching existing project patterns:
```python
from tyagent.tools.registry import registry, tool_error, tool_result
```

### MemoryStore class

Similar to Hermes `MemoryStore` but simpler — no frozen snapshot caching (tyagent injects memory as system messages at the gateway level, so it doesn't need the snapshot pattern).

```python
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

ENTRY_DELIMITER = "\n§\n"

class MemoryStore:
    """Bounded curated memory with file persistence."""

    def __init__(self, memories_dir: str | Path,
                 memory_char_limit: int = 2000,
                 user_char_limit: int = 1000):
        self.memories_dir = Path(memories_dir)
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.load_from_disk()

    def _path_for(self, target: str) -> Path:
        if target == "user":
            return self.memories_dir / "USER.md"
        return self.memories_dir / "MEMORY.md"

    def _read_file(self, path: Path) -> list[str]:
        """Read entries from file, split by ENTRY_DELIMITER."""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except (OSError, IOError):
            return []
        if not raw:
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _write_file(self, path: Path, entries: list[str]) -> None:
        """Atomic write via tempfile + os.replace."""
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".mem_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @contextmanager
    def _file_lock(self, path: Path):
        """Acquire exclusive file lock for read-modify-write safety."""
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDONLY | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def load_from_disk(self) -> None:
        """Load entries from MEMORY.md and USER.md."""
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_file(self._path_for("memory"))
        self.user_entries = self._read_file(self._path_for("user"))
        # Deduplicate (preserve order, keep first)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

    def _reload_target(self, target: str) -> None:
        """Re-read entries from disk for a specific target (called under lock)."""
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))
        if target == "user":
            self.user_entries = fresh
        else:
            self.memory_entries = fresh

    def _char_count(self, target: str) -> int:
        entries = self.user_entries if target == "user" else self.memory_entries
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def add(self, target: str, content: str) -> dict:
        """Append an entry. Returns dict with success/error."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = self.user_entries if target == "user" else self.memory_entries

            if content in entries:
                return {"success": True, "message": "Entry already exists (no duplicate added)."}

            new_total = len(ENTRY_DELIMITER.join(entries + [content]))
            limit = self._char_limit(target)

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": f"Memory at {current:,}/{limit:,} chars. Entry exceeds limit. Replace or remove existing entries first.",
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_target(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    # ... replace(), remove(), save_to_disk(), get_all_formatted() follow same pattern
```

Key implementation notes:
- `load_from_disk()` calls `self.memories_dir.mkdir(parents=True, exist_ok=True)` before reading, so it's safe on first run when dir doesn't exist
- `_read_file()` returns `[]` if file doesn't exist (no crash on first run)
- File lock with `fcntl.flock` for multi-session write safety
- Atomic writes via `tempfile.mkstemp` + `os.replace()`
- Re-read entries from disk under lock before mutating (to catch writes from other sessions)
- Reject exact duplicates
- Enforce per-target char budgets
- Injection security scan (prompt injection patterns, invisible unicode chars)
- `get_all_formatted()` returns a formatted string for system prompt injection

### Tool handler — module-level MemoryStore singleton

Since tyagent's `registry.dispatch()` only passes `args` (no `**kw`), use a module-level singleton set by the gateway at startup:

```python
_global_store: MemoryStore | None = None

def set_store(store: MemoryStore) -> None:
    global _global_store
    _global_store = store
```

### Schema (no `read` action — memory is always injected via system prompt)

```python
MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is automatically injected into the system prompt, so keep entries "
        "compact and focused.\n\n"
        "WHEN TO SAVE: user corrects you, user shares preferences/environment details, "
        "you discover project conventions or tool quirks, you identify stable facts "
        "useful across sessions.\n\n"
        "Do NOT save task progress or temporary state — use session_search for that.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is (name, preferences, communication style)\n"
        "- 'memory': your notes (environment facts, project conventions, tool quirks)\n\n"
        "ACTIONS:\n"
        "- add: append a new entry\n"
        "- replace: update an existing entry (old_text identifies it)\n"
        "- remove: delete an entry (old_text identifies it)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"],
                       "description": "Action to perform."},
            "target": {"type": "string", "enum": ["memory", "user"],
                       "description": "Which store to operate on."},
            "content": {"type": "string",
                        "description": "Entry content. Required for add and replace."},
            "old_text": {"type": "string",
                         "description": "Unique substring identifying entry for replace/remove."},
        },
        "required": ["action", "target"],
    },
}
```

### Registration

```python
registry.register(
    name="memory",
    schema=MEMORY_SCHEMA,
    handler=_handle_memory,
    description="Persistent memory across sessions (agent notes + user profile)",
    emoji="🧠",
)
```

### Tools init import

Append to `tyagent/tools/__init__.py`:
```python
import tyagent.tools.memory_tool  # noqa: F401
```

**Verification:**
- `pytest tests/test_memory_tool.py -v` — all new tests pass
- `pytest -x` — all existing tests still pass (regression check)

---

## Step 3: Inject memory into system prompt (gateway.py)

**Objective:** Load memory from disk, inject it as a system message so every LLM call sees it. Also wire up the MemoryStore singleton for the memory tool handler.

**Files:**
- Modify: `tyagent/gateway.py`

### Import

```python
from tyagent.tools import memory_tool
```

### Changes to `Gateway.__init__()`

After creating `self.session_store`, also create the MemoryStore and wire it up:

```python
# In Gateway.__init__(), after self.session_store is set:
memories_dir = config.home_dir / "memories"
self.memory_store = MemoryStore(memories_dir)
memory_tool.set_store(self.memory_store)
```

### Changes to `Gateway._on_message()`

Just before calling `self.agent.chat()`, inject the memory block:

```python
# After building api_messages (line 137) and before agent.chat() (line 145):
memory_block = self.memory_store.get_all_formatted()
if memory_block:
    # Build a new list (don't mutate api_messages — it may alias session.messages)
    insert_at = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
    api_messages = (
        api_messages[:insert_at]
        + [{"role": "system", "content": memory_block}]
        + api_messages[insert_at:]
    )
```

This works because:
- `agent.chat()` checks if first message is system (line 99), skips injecting default if so
- A second system message at index 1 is passed through to the API without issue
- `build_api_messages()` in `context.py` preserves all system messages (they're kept as-is)
- The memory is injected fresh each turn, so any mid-session memory changes are visible

**Note:** `tyagent_cli.py` only creates a TyAgent in one debug helper, so it doesn't need modification.

**Verification:**
- Gateway test with mocked MemoryStore shows memory block in messages list
- `pytest -x` — no regression

---

## Step 4: FTS5 + jieba in db.py

**Objective:** Add FTS5 full-text search to `Database`, with jieba segmentation for Chinese support.

**Files:**
- Modify: `tyagent/db.py`

### Schema version: v2

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,          -- jieba-segmented text (space-joined Chinese words + English)
    reasoning,        -- jieba-segmented reasoning content
    tokenize='unicode61'  -- tokenizes on spaces (fine for pre-segmented text)
);
```

Why `unicode61` on pre-segmented text? Because the text is already space-joined by jieba. `unicode61` tokenizes on whitespace → each jieba word becomes one FTS5 token. For English words, `unicode61` additionally handles case-folding (e.g., "Hello" → "hello") and Unicode normalization.

### `jieba_segment()` helper (module-level function in db.py)

```python
import jieba

_DICT_LOADED = False

def _ensure_jieba_dict(dict_path: str | Path | None = None) -> None:
    """Load custom jieba dictionary once."""
    global _DICT_LOADED
    if _DICT_LOADED:
        return
    if dict_path and Path(dict_path).exists():
        jieba.load_userdict(str(dict_path))
    _DICT_LOADED = True

def jieba_segment(text: str) -> str:
    """Segment text with jieba and join with spaces for FTS5 indexing."""
    if not text or not text.strip():
        return ""
    return " ".join(jieba.cut(text))
```

### FTS5 query sanitization

FTS5 MATCH syntax supports special operators (`*`, `"`, `AND`, `OR`, `NOT`, `NEAR`). A jieba-segmented query containing these characters could cause syntax errors. Wrap each segment term in double quotes:

```python
def _fts_escape(segmented: str) -> str:
    """Escape FTS5 query: wrap each term in quotes to avoid syntax errors from special chars."""
    if not segmented:
        return ""
    terms = segmented.split()
    quoted = []
    for t in terms:
        # Escape any double quotes inside the term
        t_escaped = t.replace('"', '""')
        quoted.append(f'"{t_escaped}"')
    return " AND ".join(quoted)
```

This makes the query a phrase search for each term (no FTS5 operator interpretation).

### Changes to `Database`:

**`__init__()`:** Accept optional `dict_path` param for jieba custom dictionary, call `_ensure_jieba_dict(dict_path)`.

**`_init_schema()`:** Extend to version 2:
- Add FTS5 table creation
- Call `_backfill_fts()` after creating FTS table if we upgraded from v1

**`add_message()`:** After inserting the message row, also INSERT into `messages_fts`. Use the auto-generated `rowid` from the message insert to match FTS rowid:

```python
rowid = cur.fetchone()[0]  # last_insert_rowid() from the message INSERT

segmented_content = jieba_segment(content) if content else ""
segmented_reasoning = jieba_segment(reasoning) if reasoning else ""

self._conn.execute(
    "INSERT INTO messages_fts (rowid, content, reasoning) VALUES (?, ?, ?)",
    (rowid, segmented_content, segmented_reasoning),
)
```

**`delete_session()`:** Also DELETE from `messages_fts`:

```python
self._conn.execute(
    "DELETE FROM messages_fts WHERE rowid IN "
    "(SELECT id FROM messages WHERE session_key = ?)",
    (session_key,)
)
```

**`import_messages()`:** Also INSERT into `messages_fts` for each imported message.

**`_backfill_fts()`:** Populate `messages_fts` from all existing messages:

```python
def _backfill_fts(self) -> None:
    """Backfill FTS index from all existing messages (for schema upgrade)."""
    cur = self._conn.execute("SELECT COUNT(*) FROM messages_fts")
    if cur.fetchone()[0] > 0:
        return  # already populated
    cur = self._conn.execute("SELECT id, content, reasoning FROM messages")
    rows = cur.fetchall()
    for row in rows:
        seg_content = jieba_segment(row["content"] or "")
        seg_reasoning = jieba_segment(row["reasoning"] or "")
        self._conn.execute(
            "INSERT INTO messages_fts (rowid, content, reasoning) VALUES (?, ?, ?)",
            (row["id"], seg_content, seg_reasoning),
        )
    self._conn.commit()
```

### New method `search_messages(query, limit=5)`

```python
def search_messages(self, query: str, limit: int = 5) -> list[dict]:
    """Search messages using FTS5 with jieba segmentation.

    Returns list of dicts: session_key, role, snippet, rank, created_at, content
    """
    segmented = jieba_segment(query)
    fts_query = _fts_escape(segmented)
    if not fts_query:
        return []

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

        results = []
        for row in cur.fetchall():
            results.append({
                "session_key": row["session_key"],
                "role": row["role"],
                "content": row["content"][:500],  # truncated for LLM context
                "created_at": row["created_at"],
                "rank": row["rank"],
            })
    return results
```

Note: The `rank` column is auto-generated by FTS5 (BM25 relevance score). Lower rank = more relevant.

### Wiring through SessionStore

`SessionStore` in `session.py` already exposes `self.db` (a `Database` instance) as a property. The search tool needs to access it. We'll add a public accessor:

```python
# In session.py, SessionStore class:
@property
def db(self) -> Database:
    return self._db
```

(Already exists — confirmed in review.)

### Database lifecycle in gateway

In `Gateway.__init__()`, after `self.session_store` is created, wire the Database to the search tool:

```python
# In Gateway.__init__():
import tyagent.tools.search_tool  # noqa: F401
search_tool.set_search_db(self.session_store.db)
```

**Verification:**
- `pytest tests/test_db.py -v` — existing tests pass
- Test `search_messages` with Chinese query returns correct results
- Test `search_messages` with English query works
- Test add_message → search returns new message
- Test delete_session → FTS results cleaned up
- Test backfill on schema upgrade

---

## Step 5: Search tool (`tyagent/tools/search_tool.py`)

**Objective:** A `session_search` tool wrapping FTS5 search for the LLM.

**Files:**
- Create: `tyagent/tools/search_tool.py`
- Modify: `tyagent/tools/__init__.py` (add import)

### Handler

```python
from __future__ import annotations

from tyagent.tools.registry import registry, tool_error, tool_result

_db: Database | None = None

def set_search_db(db: Database) -> None:
    global _db
    _db = db

def _handle_session_search(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return tool_error("query is required")
    limit = min(int(args.get("limit", 5)), 20)

    if _db is None:
        return tool_error("Search database not available")

    results = _db.search_messages(query, limit=limit)
    return tool_result(success=True, results=results, count=len(results))
```

### Schema

```python
SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": "Search past conversations using full-text search. Supports both Chinese and English queries.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (Chinese or English)"},
            "limit": {"type": "integer", "description": "Max results (default: 5, max: 20)"},
        },
        "required": ["query"],
    },
}
```

### Registration

```python
registry.register(
    name="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=_handle_session_search,
    description="Search past conversations using full-text search",
    emoji="🔍",
)
```

### Tools init import

Add to `tyagent/tools/__init__.py`:
```python
import tyagent.tools.search_tool  # noqa: F401
```

**Note:** The import of `Database` type in `search_tool.py` uses `from tyagent.db import Database` to avoid circular imports.

**Verification:**
- `pytest tests/test_search_tool.py -v` — all tests pass
- `pytest -x` — no regression

---

## Step 6: Tests for memory_tool.py

**Files:**
- Create: `tests/test_memory_tool.py`

Test classes:
- `TestMemoryStoreDefaults`: init, empty state, char limits
- `TestMemoryStoreAdd`: add entry, add duplicate (rejected), add over budget (rejected), add empty (rejected)
- `TestMemoryStoreReplace`: replace matching, replace no match (error), replace multiple matches (error)
- `TestMemoryStoreRemove`: remove matching, remove no match (error)
- `TestMemoryStorePersistence`: write then read back, file exists after save
- `TestMemoryFormatting`: `get_all_formatted()` with entries, without entries
- `TestInjectionScan`: rejects prompt injection patterns, rejects invisible unicode, allows normal content
- `TestMemoryToolHandler`: tool registered, dispatch valid/invalid args, no global store error
- `TestMemoryStoreCrossSession`: simulate two stores writing to same files (lock test)

**Verification:**
- `pytest tests/test_memory_tool.py -v` — all pass

---

## Step 7: Tests for search_tool.py

**Files:**
- Create: `tests/test_search_tool.py`

Test classes:
- `TestJiebaSegment`: Chinese text, English text, mixed text, empty string, None safe, custom dict terms
- `TestFtsEscape`: normal terms, terms with quotes, special chars, empty input
- `TestSearchDatabase`: (uses in-memory SQLite with FTS5 v2 schema)
  - Add messages, search Chinese, search English
  - Search mixed query
  - Empty results for no match
  - Limit enforcement
  - Delete session → search results cleaned up
  - Backfill from existing messages
- `TestSearchToolHandler`: Schema registration, dispatch valid query, dispatch missing query (error), dispatch without db (error)

**Verification:**
- `pytest tests/test_search_tool.py -v` — all pass

---

## Full Integration Check (after all steps)

After all 7 steps complete:
1. `pytest -x` — all ~240+ tests pass
2. Subagent full blind review of ALL new/modified files
3. Two consecutive clean rounds → done
