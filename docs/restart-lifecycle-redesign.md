# Architectural Redesign: Restart Tool-Call Lifecycle

## Current Architecture (for reference)

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│  _handle_terminal() in tools/core.py                        │
│                                                             │
│  1. Regex-match command against _RESTART_TRIGGERS          │
│  2. If match → write .gateway_interrupt/<uuid>.json        │
│  3. Popen(command)                                          │
│  4. Write .terminal_pending/<uuid>.json (ALL commands)      │
│  5. communicate(timeout)                                    │
│  6. Read output file → return tool_result                   │
│  7. Clean up markers                                        │
│  8. On ANY Exception → clean up gw_interrupt marker (!)     │
└─────────────────────────────────────────────────────────────┘
         │                          │
         │ (markers on disk)        │ (SIGUSR1 to gateway)
         ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│  _write_restart_marker() in gateway/lifecycle.py            │
│                                                             │
│  A. Collect .gateway_interrupt/ → reason="restart_trigger" │
│  B. Collect in-flight tool calls → reason="unknown_failure"│
│  C. Dedup: if same tc_id in both, keep restart_trigger     │
│  D. Write .restart_pending                                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  _handle_restart_marker_on_startup() in lifecycle.py        │
│                                                             │
│  Step 1: _collect_orphan_terminal_results()                │
│          → scans .terminal_pending/ for real output        │
│          → writes tool response to DB (NO dedup check!)    │
│  Step 2: Process .restart_pending                          │
│          → for each pending tc_id, check if response exists│
│          → if not, write synthetic response                │
└─────────────────────────────────────────────────────────────┘
```

### Three Classes of Bugs Being Patched

| Bug | Location | Root Cause |
|-----|----------|------------|
| **#1** Exception handler scope | `_handle_terminal` lines 879-891 | Broad `except Exception` cleans up `gw_interrupt_path` even when command already ran (e.g., `communicate` OSError). The restart still happens but the restart machinery can't tell this tc_id was a trigger. |
| **#2** Regex false positives/negatives | `_handle_terminal` lines 782-793 | `kill` pattern (`kill\s+-SIGUSR1\s+\S+`) matches `echo "kill -SIGUSR1 12345"` (FP) and misses `xargs kill -SIGUSR1` (FN). Anchored patterns miss piped variants. |
| **#3** Dedup removed from collector | `_collect_orphan_terminal_results` | Writes tool response without checking if one already exists → duplicate responses on crash-during-cleanup. |

---

## Proposed Redesign: "Make Bugs Impossible by Design"

### Principle 1: Split Exception Scopes by Phase (Bug #1)

**Problem**: The current try/except conflates pre-Popen failures (command never ran) with post-Popen failures (command ran, but gateway crashed or file-read failed). Both paths clean up the restart-intent marker, which is wrong for post-Popen failures.

**Fix**: Split `_handle_terminal` into three explicit phases. Only clean up markers in Phase 1.

```python
def _handle_terminal(args, parent_agent=None):
    # === Phase 0: Setup ===
    tool_call_id = getattr(parent_agent, "_current_tool_call_id", "")
    is_restart = _is_restart_command(command)  # module-level function

    # Write unified marker (see Principle 4)
    marker_path = _write_tool_pending_marker(
        home_dir, tool_call_id, session_key, session_id,
        command, reason="restart" if is_restart else "terminal"
    )

    # === Phase 1: Popen (critical section) ===
    # If this fails, the command NEVER ran. Safe to clean up everything.
    try:
        proc = subprocess.Popen(command, shell=True, stdout=out_f, ...)
    except Exception as exc:
        _cleanup_marker(marker_path, output_path)
        return tool_error(f"Failed to start command: {exc}")

    # === Phase 2: Wait (command IS running) ===
    # From this point forward, we MUST NOT clean up the marker.
    # If the gateway crashes during communicate(), the marker persists
    # and _collect_orphan_terminal_results() collects the real output.
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        # Timeout = command finished (abnormally), clean up normally
        _cleanup_marker(marker_path, output_path)
        return tool_error(f"Command timed out after {timeout}s")
    except Exception as exc:
        # Unexpected error during wait (e.g., OSError from broken pipe).
        # Command DID run. DO NOT clean up markers — let restart machinery
        # collect the output if gateway is restarting.
        return tool_error(f"Command wait failed: {exc}")

    # === Phase 3: Read output ===
    try:
        with open(output_path) as f:
            output = f.read()
    except OSError as exc:
        # Can't read output but command ran. DO NOT clean up markers.
        return tool_error(f"Failed to read output: {exc}")

    # Success — clean up normally
    _cleanup_marker(marker_path, output_path)
    return tool_result(...)
```

**Why this makes Bug #1 impossible**: There is NO code path that calls `_cleanup_marker` (or cleans up restart-intent state) after `Popen` succeeds except explicit success/timeout paths. The "catch-all exception → clean up" anti-pattern is eliminated.

**Trade-off**: Slightly more code (explicit paths instead of a single catch-all), but the control flow is explicit and auditable. The marker sometimes persists after a real error (e.g., command started but file-read failed), but that's harmless — the next startup's `_collect_orphan_terminal_results` will detect the stale marker (no output, process dead) and clean it up.

---

### Principle 2: Module-Level Restart Detection + Dedicated Tool (Bug #2)

**Problem**: Inline regex inside `_handle_terminal` is hard to test, fragile (anchors/word boundaries), and fundamentally unreliable — can't predict all ways to restart.

**Fix**: Two complementary changes:

#### 2a. Extract to Module-Level Constant (immediate fix)

```python
# In tools/core.py, at module level:

# Compiled once, testable independently, word-boundary safe.
_RESTART_PATTERNS: tuple[re.Pattern, ...] = (
    # Direct CLI calls — anchored to avoid echo/printf false positives.
    re.compile(r"^(?:sudo\s+)?(?:uv\s+run\s+)?(?:python3(?:\.\d+)?\s+)?tyagent(?:_cli\.py)?\s+gateway\s+restart\b"),
    # systemctl — anchored.
    re.compile(r"^(?:sudo\s+)?systemctl\s+(?:--user\s+)?restart\s+tyagent-gateway\b"),
    # kill via SIGUSR1 — word-boundary on both sides, no PID required (handles xargs).
    # \bkill\b matches 'kill' as a word, not inside 'echokill'.
    re.compile(r"\bkill\b.*\b-?SIGUSR1\b"),
)

def _is_restart_command(command: str) -> bool:
    """Check if a shell command is likely to trigger a gateway restart.

    This is intentionally broad for the 'kill' pattern to handle pipelines
    (pgrep | xargs kill -SIGUSR1). False positives (echo "kill -SIGUSR1")
    are harmless — they only cause a spurious restart_trigger marker that
    gets cleaned up when the command completes normally without causing
    a restart.
    """
    return any(p.search(command) for p in _RESTART_PATTERNS)
```

**Key improvements**:
- `\bkill\b` — word boundaries prevent matching inside other words
- `.*\b-?SIGUSR1\b` — matches `kill -SIGUSR1`, `kill -s SIGUSR1`, `xargs kill -SIGUSR1`
- Accepts false positives (echo "kill -SIGUSR1") because they're **harmless**: the gw_interrupt marker is written, but the command completes normally without triggering a restart. On the NEXT restart (whenever that happens), `_write_restart_marker` will find the stale marker and... actually this IS a problem.

**Wait — the false positive problem**: If echo "kill -SIGUSR1" incorrectly triggers a restart_trigger marker, and then some later /restart happens, `_write_restart_marker` will pick up the stale marker from the echo command and give it a synthetic "restart success" response. But the echo command already completed and got a real response. The dedup in `_handle_restart_marker_on_startup` step 2 checks for existing responses, so it would skip. But the stale marker accumulates on disk.

**Better approach for kill pattern**: Use a heuristic that kills with SIGUSR1 on the gateway PID. But we can't know the gateway PID inside a shell command.

**Alternative**: Accept the false positive and ensure stale markers don't cause harm:
- `_handle_restart_marker_on_startup` step 2 checks for existing tool responses before writing synthetic ones → no duplicate responses
- `_cleanup_gateway_interrupt_dir` is called on startup (in `check_recovery_on_startup`) and after processing restart markers
- Any stale marker lingering between restarts is harmless

But the real question: can we avoid regex entirely?

#### 2b. Add `restart_gateway` Tool (eliminates regex for the primary path)

```python
# New tool: restart_gateway
# The agent calls this INSTEAD of terminal("tyagent gateway restart")

def _handle_restart_gateway(args, parent_agent=None):
    """Request a graceful gateway restart.

    This is the preferred way to restart. It writes the proper markers
    so the tool chain is preserved across the restart without relying
    on regex-based command detection.
    """
    tool_call_id = getattr(parent_agent, "_current_tool_call_id", "")
    home_dir = getattr(parent_agent, "home_dir", None)
    session_key = getattr(parent_agent, "session_key", "")
    session_id = getattr(parent_agent, "current_session_id", "")

    # Write restart-intent marker with explicit reason
    _write_tool_pending_marker(
        home_dir, tool_call_id, session_key, session_id,
        command="<restart_gateway tool>", reason="restart"
    )

    # Send SIGUSR1 to own gateway process
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
    except OSError:
        return tool_error("Failed to send restart signal")

    # Return immediately — the gateway will restart, and the next
    # startup's _handle_restart_marker_on_startup will write a
    # synthetic "restart completed" response for this tool_call_id.
    return tool_result({
        "success": True,
        "message": "Gateway restart initiated. The response will be updated after restart.",
        "restart_pending": True,
    })
```

**Why this helps**: The agent now has a first-class tool for restarting. The tool explicitly sets `reason="restart"` — no regex needed. The pattern becomes: "call restart_gateway tool" → marker written → restart → synthetic response.

**Trade-off**: The terminal-based restart path still needs regex for backward compatibility (users typing `systemctl restart` directly). But:
1. The agent prefers `restart_gateway`
2. The regex is now only a fallback, and false positives are harmless due to dedup

**Is regex fundamentally avoidable?** Not completely, unless we remove the terminal-based restart path entirely. A possible middle ground: make `_is_restart_command` a **plugin point** — allow users to configure custom patterns, and provide a safe default that errs on the side of false positives (harmless) over false negatives (broken chain).

---

### Principle 3: Dedup at Collection Time (Bug #3)

**Problem**: `_collect_orphan_terminal_results` writes tool responses to DB without checking if one already exists. If the gateway crashes after the DB write but before cleaning the marker file, the next startup writes a duplicate.

**Fix**: Add a lightweight dedup check before writing each response:

```python
def _collect_orphan_terminal_results(self) -> set[str]:
    affected = set()
    for marker_path in markers:
        ...
        # Check if a tool response already exists for this tool_call_id
        try:
            existing = gw.session_store.get_messages(
                session_key, session_id=session_id
            )
        except Exception:
            existing = []

        has_response = any(
            m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id
            for m in existing
        )
        if has_response:
            logger.debug(
                "Skipping %s — tool response already exists for %s",
                marker_path.name, tool_call_id,
            )
            marker_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            continue

        # ... write response ...
```

**Why this makes Bug #3 impossible**: Every write is guarded by an existence check. Even in the crash-during-cleanup scenario, the second startup detects the existing response and skips.

**Trade-off**: One extra DB read per pending marker. This is negligible compared to the cost of a duplicate response breaking the message chain.

**Question from the task**: "Should dedup happen at collection time (_write_restart_marker) rather than at processing time?"

**Answer**: It should happen at BOTH times, for different reasons:
- **Collection time** (`_write_restart_marker`): Already does dedup between gateway_interrupt and in-flight (lines 487-497). This prevents the same tc_id from appearing twice in `.restart_pending` with different reasons. This is correct and should stay.
- **Processing time** (`_handle_restart_marker_on_startup` step 2): Already checks for existing responses (lines 617-623). This is correct and should stay.
- **Orphan collection time** (`_collect_orphan_terminal_results`): Currently MISSING the dedup check. This is Bug #3.

So the answer is: dedup should happen at **all three** points. Each prevents a different race condition.

---

### Principle 4: Unified Marker File Per Tool Call

**Problem**: Two separate marker directories (`.gateway_interrupt/` and `.terminal_pending/`) create inconsistency and require cross-referencing. UUID-based filenames allow multiple markers for the same tool_call_id.

**Fix**: Unify into a single `.tool_pending/` directory with tool_call_id-based filenames.

```python
# Old:
#   .gateway_interrupt/<uuid>.json  → reason="restart_trigger"
#   .terminal_pending/<uuid>.json   → all terminal commands

# New:
#   .tool_pending/<tool_call_id>.json  → reason="restart" or "terminal"

MARKER_SCHEMA = {
    "tool_call_id": "call_abc123",      # used as filename → natural dedup
    "session_key": "user:chat123",
    "session_id": "sess_xyz",
    "command": "tyagent gateway restart",
    "started_at": 1715100000.0,
    "output_path": "/tmp/tyagent_term_abc.out",
    "pid": 12345,
    "reason": "restart",                 # "restart" | "terminal"
    "version": 1,                        # schema version for future migration
}
```

**Benefits**:
- **Natural dedup**: Can't have two markers for the same tool_call_id (filename collision)
- **Single source of truth**: One place to look for all pending tool calls
- **Simpler restart machinery**: `_write_restart_marker` reads one directory instead of two
- **Simpler cleanup**: One directory to manage instead of two

**How it changes the flow**:

In `_handle_terminal`:
```python
# Write ONE marker for all terminal commands
marker_path = home_dir / ".tool_pending" / f"{tool_call_id}.json"
data = {
    "tool_call_id": tool_call_id,
    "reason": "restart" if _is_restart_command(command) else "terminal",
    ...
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
marker_path.write_text(json.dumps(data))

# ... execute command ...

# On success/timeout: clean up marker
marker_path.unlink(missing_ok=True)
```

In `_write_restart_marker`:
```python
# Collect ALL pending tool calls from a single directory
pending_dir = home_dir / ".tool_pending"
if pending_dir.exists():
    for marker_file in pending_dir.glob("*.json"):
        data = json.loads(marker_file.read_text())
        reason = data.get("reason", "terminal")
        marker["sessions"][sk]["pending_tool_calls"].append({
            "tool_call_id": data["tool_call_id"],
            "reason": reason,  # "restart" or "terminal"
            ...
        })
```

In `_collect_orphan_terminal_results`:
```python
# Same directory, filter by reason="terminal"
for marker_file in pending_dir.glob("*.json"):
    data = json.loads(marker_file.read_text())
    if data.get("reason") != "terminal":
        continue  # restart-trigger markers handled by synthetic path
    ...
```

**Trade-off**: The `reason` field reintroduces the need for classification. But it's now a simple field, not a separate directory. And for the `restart_gateway` tool (Principle 2b), the reason is set explicitly without regex.

---

### Principle 5: Is "Write Marker → Popen → Clean on Failure" Fundamentally Sound?

**Analysis**: The pattern itself is sound — it's a form of "intent logging" similar to write-ahead logging (WAL) in databases. The problem isn't the pattern, it's the scope of the cleanup.

The pattern works correctly when:
1. Write intent (marker) — if this fails, abort (can't track the command)
2. Execute — if this fails, clean up intent (command never ran)
3. Command runs — intent stays (serves as record)
4. Command completes normally — clean up intent
5. Gateway crashes — intent persists, collected on restart

The bug in the current code is that step 3 failures (command ran but communicate/file-read failed) incorrectly trigger step 2 cleanup. Principle 1 fixes this by making the phase boundaries explicit.

**Could we make it atomic?** Not easily — POSIX doesn't support "write file AND fork process" atomically. The best we can do is make the marker filename intrinsically tied to the command (using tool_call_id) so that retries don't create duplicates. Principle 4 achieves this.

---

## Summary of Changes

### File: `tyagent/tools/core.py`

| Change | Impact |
|--------|--------|
| Extract `_RESTART_PATTERNS` to module level | Testable, maintainable, uses `\b` for word boundaries |
| Add `_is_restart_command()` function | Single point of truth for restart detection |
| Split `_handle_terminal` try/except into 3 phases | Bug #1 eliminated: post-Popen errors never clean up markers |
| Unify markers: single `.tool_pending/<tc_id>.json` | Bug #3 natural dedup; simpler lifecycle |
| Add `restart_gateway` tool | Bug #2 eliminated for primary path (no regex needed) |

### File: `tyagent/gateway/lifecycle.py`

| Change | Impact |
|--------|--------|
| `_write_restart_marker`: read unified `.tool_pending/` dir | Single directory to manage |
| `_collect_orphan_terminal_results`: add dedup check before write | Bug #3 eliminated |
| Filter by `reason` field instead of directory | Cleaner separation of concerns |
| `_cleanup_gateway_interrupt_dir` → `_cleanup_tool_pending_dir` | Renamed for clarity |

### File: `tyagent/tools/registry.py`

| Change | Impact |
|--------|--------|
| Register `restart_gateway` tool | New first-class tool, eliminates regex for primary restart path |

### File: `tests/test_gateway.py`

| Change | Impact |
|--------|--------|
| `TestRestartTriggers`: test module-level `_RESTART_PATTERNS` directly | No need to reconstruct patterns from `_handle_terminal` |
| Add tests for: kill false positive (echo), kill false negative (xargs), phase-scoped exception handling, dedup in orphan collection | Regression prevention |
| Test unified marker with reason field | Verify correct behavior |

---

## Trade-offs Discussion

### 1. Regex vs. Explicit Tool

**Regex approach**: Covers all possible restart commands, including edge cases. But inherently fragile.
**Explicit tool approach**: Perfect correctness for the tool path. But doesn't cover user-typed terminal commands.
**Hybrid (recommended)**: `restart_gateway` tool for the agent (primary path), regex fallback for terminal commands. The regex is now a safety net, not the primary mechanism.

### 2. False Positives in Kill Pattern

The broader kill pattern (`\bkill\b.*\b-?SIGUSR1\b`) has more false positives but fewer false negatives. False positives are **harmless** because:
- The restart_trigger marker is written but the command completes normally
- On the next real restart, the marker is processed. But if a real response already exists, dedup skips it.
- The marker is cleaned up after processing.

False negatives are **harmful** because they break the message chain (missing tool response). So biasing toward false positives is the right trade-off.

### 3. Phase Splitting vs. Catch-All

Splitting the try/except means more code paths to audit. But the catch-all was the source of Bug #1. The explicit paths make the behavior **obvious by reading the code** — you can see exactly what happens in each failure mode.

### 4. Dedup at Multiple Points

Three dedup points seem redundant, but each prevents a different race:
- `_write_restart_marker` dedup: prevents same tc_id appearing with different reasons
- `_collect_orphan_terminal_results` dedup: prevents duplicate responses from crash-during-cleanup
- `_handle_restart_marker_on_startup` dedup: prevents synthetic response overwriting real output

These are defense-in-depth, not redundancy. Each is necessary for its specific scenario.

### 5. Tool-Call-ID as Marker Filename

**Pro**: Natural dedup — can't have two markers for the same call.
**Con**: If tool_call_id is empty (shouldn't happen), we can't write a marker. Fallback: use UUID for empty tool_call_id.
**Mitigation**: `tool_call_id` is always set by the agent before executing any tool (agent.py:269).

---

## Migration Path

1. **Phase 1** (low risk): Extract regex to module level, split exception handling, add dedup to `_collect_orphan_terminal_results`. These are surgical fixes to the existing architecture.
2. **Phase 2** (medium risk): Unify marker directories (`.tool_pending/` with `reason` field), maintaining backward compat by reading both old and new marker locations during a transition period.
3. **Phase 3** (new feature): Add `restart_gateway` tool. This is additive and doesn't break anything.
4. **Phase 4** (cleanup): Remove old marker directories after confirming no stale markers exist in production.
