# Plan: Wiki-Style Progressive Disclosure Memory System

> **Goal:** Upgrade memory from flat file injection to a wiki-style bidirectional-linked memory system with progressive disclosure.
>
> **Architecture:** Three-phase evolution. Phase 1: summary-line index + `expand(keyword)` tool. Phase 2: backlinks in expand output. Phase 3: auto `[[]]` detection on add/replace.
>
> **Execution Flow:** Each phase: plan doc → implement → test pass → subagent blind review → fix if issues → repeat until 2 consecutive clean rounds → next. After all phases: final full blind review, 2 clean rounds → done.

---

## Phase 1: Summary-Index + Expand Tool

**Objective:** Memory index becomes a lightweight summary of all entries (compact, always-injected). LLM can use `expand(keyword)` to retrieve full entry content.

### Changes

#### memory_tool.py — MemoryStore changes

**New entry format:** Each memory entry is a dict stored internally:

```python
{
    "id": "m001",              # auto-assigned UUID or sequential
    "title": "short title",    # extracted from first line or auto-generated
    "summary": "one-line summary",  # first ~80 chars of content
    "content": "full entry text\nwith multiple lines",
    "links": ["keyword1", "keyword2"],  # [[]] references extracted from content
    "created_at": "timestamp",
}
```

Actually, keeping it flat is simpler — since entries are `§`-delimited, we can use a **first-line-as-title convention**:

```
Memory Index Format (what gets injected into system prompt):
══════════════════════════════════════════
USER PROFILE [entries: 2]
  · 用户中文名字二狗，英文名托尼
  · 喜欢短回复而不要冗长分析

MEMORY (your notes) [entries: 3]
  · tyagent 是一个 AI Agent 框架
  · 项目使用 pytest，所有修复必须有测试
  · SQLite 会话存储重构进行中
══════════════════════════════════════════
```

**get_all_formatted()** → produces summary-only output:
- Each entry → one bullet point with first ~100 chars
- Shows entry count per store
- LLM sees lightweight index, not full content

**New `expand(keyword)` action on `memory` tool:**

```python
# In MemoryStore:
def expand(self, target: str, keyword: str) -> dict:
    """Find entries containing keyword and return full content."""
    entries = self._entries_for(target)
    matches = []
    for i, entry in enumerate(entries):
        if keyword.lower() in entry.lower():
            # Show full entry + its position
            matches.append({
                "index": i,
                "content": entry,
                "char_count": len(entry),
            })
    return {"success": True, "matches": matches, "count": len(matches)}
```

Note: `get_all_formatted()` is also made available as **Schema-level read action** so the LLM doesn't have to rely purely on the injected index being current after mid-session writes.

#### memory_tool.py — Schema changes

Add `expand` action + `keyword` parameter (required for expand).

Old: `action enum: [add, replace, remove]`
New: `action enum: [add, replace, remove, expand]`

Add `keyword` parameter:
```python
"keyword": {
    "type": "string",
    "description": "Keyword to search for in memory entries. Required for 'expand' action.",
}
```

Also add a `read` action that calls `get_all_formatted()` directly (returns the current index without relying on system prompt injection freshness).

Old: `action enum: [add, replace, remove]`
New: `action enum: [add, replace, remove, expand, read]`

### Files Modified:
- `tyagent/tools/memory_tool.py`
- `tests/test_memory_tool.py` (add expand + read tests)

### Verification:
- `pytest tests/test_memory_tool.py -v` — all new + existing pass
- `pytest -x` — no regression (218 + 50 = 268 baseline)

---

## Phase 2: Backlinks

**Objective:** When `expand` returns an entry, also show which OTHER entries reference `[[]]` this entry.

### Changes

**MemoryStore maintains a backlink index:**
```python
self._backlinks: Dict[str, List[int]] = {}  # keyword → [entry_indexes that link to it]
```

**On add/replace:** Scan entry content for `[[]]` patterns, update `_backlinks`.

**On remove:** Clean up backlinks.

**expand()** returns:
```python
{
    "success": True,
    "matches": [
        {
            "index": 0,
            "content": "完整条目内容...",
            "char_count": 123,
            "referenced_by": [
                {"index": 2, "summary": "引用该条目的摘要..."},
            ]
        }
    ],
    "count": 1
}
```

### Files Modified:
- `tyagent/tools/memory_tool.py`
- `tests/test_memory_tool.py`

---

## Phase 3: Bidirectional Link Visibility — `links_to` + cross-store expand

**Objective:** When `expand` returns a matching entry, show both:
- **`links_to`** — which keywords this entry links *to* via `[[keyword]]` (forward links)
- **`referenced_by`** — which entries link *back* to this entry (backlinks, already implemented)

Also make `expand()` more flexible by allowing cross-store search (when `target` is omitted).

### Changes

#### `expand()` — richer response

Each match dict adds `links_to`:

```python
{
    "index": 0,
    "content": "Project [[tyagent]] integrates with [[飞书]]",
    "links_to": [
        {"keyword": "tyagent"},
        {"keyword": "飞书"},
    ],
    "referenced_by": [ ... ],
}
```

`links_to` is derived by scanning the entry's content with `_WIKI_LINK_RE` and extracting all `[[keyword]]` patterns. No lookup needed — just regex extraction.

#### `expand()` — optional `target` (cross-store search)

When `target` is not provided (or None/empty), search both `"memory"` and `"user"` stores. The response includes the `target` for each match.

#### Schema changes

Update `target` parameter to be optional (not in `required` array) — only for `expand` and `read`. Keep `target` required for `add`, `replace`, `remove`.

Actually, the simplest approach: make `target` optional in the schema (remove from `required`), and handle it per-action in the handler:
- `add/replace/remove` — error if target is None
- `expand` — search all stores if target is None
- `read` — return both stores if target is None (already implemented)

### Files Modified:
- `tyagent/tools/memory_tool.py` — `expand()` method, handler dispatch, schema
- `tests/test_memory_tool.py` — new tests for links_to and cross-store expand

### Verification:
- `pytest tests/test_memory_tool.py -v` — all 77+ tests pass
- `pytest -x` — no regression (329+ baseline)

---

## Phase 4 (Future): Suggestive `[[]]` Auto-Wrapping

- On `add`, scan entry content for words/phrases that match existing entry summaries
- Auto-wrap them in `[[]]` if they appear verbatim in existing entries
- `read()` → `expand()` → `read()` cycle for serendipitous discovery

---

## Test Strategy

Each phase adds tests alongside implementation:

| Phase | New Tests |
|-------|-----------|
| 1 | Test expand finds correct entries, test expand no match, test read returns index, test schema has new actions |
| 2 | Test backlinks populated on add, test backlinks updated on replace, test backlinks cleaned on remove, test expand returns backlinks |
| 3 | Test auto-detection of `[[]]` in content, test rebuild on save, test cross-entry links found |
