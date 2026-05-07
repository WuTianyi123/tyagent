# Analysis: Alternatives to Regex-Based Restart Trigger Detection

**Context:** `/home/wtyopenclaw/project/tyagent/tyagent/tools/core.py` lines 782-793

## Current System Flow

1. AI agent calls `terminal(command="tyagent gateway restart")` (or systemctl/kill variant)
2. `_handle_terminal()` regex-checks the command string against `_RESTART_TRIGGERS`
3. If match: writes `.gateway_interrupt/<uuid>.json` marker (so the new gateway knows this was an intentional restart)
4. Always: writes `.terminal_pending/<uuid>.json` marker (for orphan output collection)
5. Subprocess runs detached (`preexec_fn=os.setpgrp`), sends SIGUSR1 to gateway
6. Gateway drains, calls `_write_restart_marker()` → reads `.gateway_interrupt/` markers + in-flight tool calls → writes `.restart_pending`
7. Gateway exits with code 75; systemd restarts it
8. New gateway startup: `_handle_restart_marker_on_startup()` reads `.restart_pending` + collects orphan terminal results → writes synthetic tool responses

### Problems with Current Regex Approach

| Problem | Detail |
|---------|--------|
| **False positives** | `^` anchors prevent `echo "tyagent gateway restart"` but can't prevent `printf '%s\n' 'tyagent gateway restart' \| sh` or `bash -c '...'` |
| **False negatives** | `xargs` pipelines where PID comes from stdin are handled by unanchored `kill -SIGUSR1 \S+`, but novel wrappers/aliases are missed |
| **Tests duplicate patterns** | `test_gateway.py:1430-1437` copy-pastes `_RESTART_TRIGGERS` instead of importing from source |
| **Patterns are private** | `_RESTART_TRIGGERS` defined locally inside `_handle_terminal`, not importable |
| **Fragile maintenance** | Every new restart method needs a new regex added (uv run, python3.x, systemctl variants, kill variants, future wrappers...) |
| **Root cause mismatch** | Regex infers *intent* from *command syntax* — fundamentally brittle because intent and syntax are only loosely coupled |

---

## Alternative Approaches Evaluated

### 1. Whitelist / Flag Approach (⭐⭐⭐⭐⭐ RECOMMENDED)

**Idea:** Don't detect restart *commands*; instead, mark the restart *action* explicitly. The `/restart` command handler (`_cmd_restart`) or a dedicated `restart_gateway` tool sets a flag/sentinel *before* the restart is triggered. The terminal tool checks this flag.

**Implementation:**
```python
# In terminal tool (_handle_terminal):
will_restart = getattr(parent_agent, "_restart_pending", False)
# OR check a sentinel file:
will_restart = (home_dir / ".restart_sentinel").exists()
```

The sentinel is set by:
- `_cmd_restart` (commands.py:126-141) — the `/restart` slash command
- A new `restart_gateway` tool — called by AI agent instead of `terminal(command="...restart...")`

The sentinel is cleared after the restart marker is written.

**Pros:**
- Zero false positives/negatives — intent is explicitly signalled
- No regex to maintain
- Works for ANY restart method (systemctl, kill, tyagent CLI, future wrappers)
- The sentinel can be written by `/restart` command, systemd service hooks, or a dedicated tool
- Simple to test: just check sentinel presence

**Cons:**
- Requires the AI agent to use `/restart` or a dedicated tool instead of `terminal(command="systemctl restart...")` — this is a **prompt engineering problem**, not a code problem
- Backward compatibility: existing AI agents that use `terminal(command="tyagent gateway restart")` would no longer be detected unless the sentinel is set by a pre-exec hook

**Mitigation for backward compat:** Keep a SIMPLIFIED regex fallback (module-level constant, importable by tests) while migrating to the flag approach. The regex becomes a secondary safety net, not the primary mechanism.

---

### 2. Process Tree Inspection (⭐⭐⭐)

**Idea:** After the terminal command completes, check if the gateway PID changed. If the gateway process died and a new one started, infer that a restart happened.

**Implementation:**
```python
# Before running command:
gateway_pid = os.getpid()

# After command completes (if we're still alive):
if not _pid_is_alive(gateway_pid):
    # Gateway restarted — this command was the trigger
    write_restart_marker()
```

**Pros:**
- No regex at all
- Detects ANY restart, regardless of how it was triggered

**Cons:**
- **Timing problem:** The terminal tool runs INSIDE the gateway process. When the gateway exits with code 75, the asyncio event loop is cancelled and `_handle_terminal` never reaches the "after command" check
- Even if it could, by that point the markers need to already be written (they're consumed by the NEW gateway process on startup)
- Requires the check to happen *before* the restart drains, but the restart might happen during drain (after tool completion)
- **Fundamentally incompatible with in-process tool execution during restart**

---

### 3. Signal-Based / PID Registration (⭐⭐)

**Idea:** The gateway writes its PID to a known file (e.g., `~/.tyagent/gateway.pid`). The terminal tool, after running the command, checks if the gateway PID is still alive. Alternatively, use a pre-exec hook that detects SIGUSR1 being sent to the gateway.

**Pros:**
- PID-based detection is reliable for "did the gateway die?"

**Cons:**
- Same timing problem as process tree inspection — the check needs to happen *before* `_handle_terminal` is interrupted
- The subprocess sends SIGUSR1 to the gateway, but the parent (terminal tool) can't easily trace which child sent which signal
- `os.kill(pid, 0)` only tells you if the process died, not WHY (restart vs crash)
- **Doesn't distinguish between intentional restart and crash** — both look the same post-hoc

---

### 4. Environment Variable / Sentinel File (⭐⭐⭐⭐)

**Idea:** Before any operation that triggers a restart (whether `/restart` command or systemctl), write a sentinel file. The terminal tool checks for this sentinel.

**Implementation:**
```python
# In _cmd_restart or before any restart-triggering operation:
(home_dir / ".restart_sentinel").touch()

# In _handle_terminal:
will_restart = (home_dir / ".restart_sentinel").exists()
```

**Pros:**
- Simple, atomic, filesystem-based
- Works across process boundaries
- No regex needed for detection
- Easy to test and debug (just check file existence)

**Cons:**
- **Race condition:** The sentinel must be written BEFORE the terminal command runs. If the AI agent calls `terminal(command="tyagent gateway restart")`, who writes the sentinel? The terminal tool itself? That's circular.
- Only works if the restart is initiated through a known path (e.g., `/restart` command) that writes the sentinel first
- Need to handle stale sentinel cleanup (if restart is aborted)

**Key insight:** This is essentially the same as approach #1 (whitelist/flag), but using a filesystem sentinel instead of an in-memory flag. The filesystem version survives process boundaries but introduces cleanup complexity.

---

### 5. Wrapper Script / Action-Level Interception (⭐⭐⭐⭐⭐)

**Idea:** Instead of letting the AI agent run arbitrary shell commands that trigger restarts, intercept restart operations at the tool/action level. Provide a dedicated `restart_gateway` tool, or have the terminal tool's schema include a `restart: true` parameter.

**Implementation A — Dedicated tool:**
```python
def _handle_restart_gateway(args):
    """Dedicated tool for gateway restart."""
    # Always writes gateway_interrupt marker
    # Sends SIGUSR1 to own process
    os.kill(os.getpid(), signal.SIGUSR1)
    return "Gateway restart initiated"
```

**Implementation B — Terminal tool parameter:**
```python
TERMINAL_SCHEMA = {
    ...
    "properties": {
        "command": {...},
        "restart_gateway": {
            "type": "boolean",
            "description": "Set to true if this command will restart the gateway",
        }
    }
}
```

**Pros:**
- **Zero regex, zero false positives/negatives** — intent is explicit
- The AI agent explicitly declares "I am restarting the gateway"
- Tool can set all necessary markers BEFORE running the command
- Clean separation of concerns
- The `/restart` command already exists — extending it is natural

**Cons:**
- Requires system prompt changes to instruct the AI agent to use the dedicated tool
- Backward compatibility: old AI agent sessions might still use `terminal(command="systemctl restart...")` — keep regex as fallback during migration
- The AI agent might still try `terminal(command="kill -SIGUSR1 ...")` — need to handle gracefully (treat as unknown failure, which already works)

---

### 6. Unified Marker File Approach (⭐⭐⭐⭐)

**Idea:** Merge `.gateway_interrupt/` and `.terminal_pending/` into a single marker directory with a `reason` field. This simplifies the marker lifecycle and reduces disk I/O.

**Current state:** Three marker types:
- `.gateway_interrupt/<uuid>.json` — intent: "this was a restart trigger"
- `.terminal_pending/<uuid>.json` — intent: "this command was running"
- `.restart_pending` — intent: "here are the pending calls from the last restart"

**Proposed unified:**
```json
// .terminal_pending/<uuid>.json
{
    "tool_call_id": "...",
    "command": "...",
    "reason": "restart_trigger" | "normal",  // NEW FIELD
    "output_path": "...",
    "pid": 12345
}
```

**Pros:**
- One less directory to manage
- Simpler cleanup logic (fewer edge cases)
- `reason` field is explicit and can be set by regex OR sentinel OR dedicated tool
- Backward compatible: add `reason` field, keep existing logic

**Cons:**
- Doesn't solve the *detection* problem — still need to determine `reason`
- Refactoring risk: tight coupling between terminal tool and restart machinery

---

## Recommendation: Hybrid Approach (1 + 5 + 6)

The most elegant and robust solution combines three approaches:

### Primary: Dedicated `restart_gateway` tool (Approach 5)

```python
# New tool registration in core.py or a new file
RESTART_GATEWAY_SCHEMA = {
    "name": "restart_gateway",
    "description": "Restart the tyagent gateway gracefully.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

def _handle_restart_gateway(args, parent_agent=None):
    """Gracefully restart the gateway."""
    # Write gateway_interrupt marker (always — this IS a restart)
    # Send SIGUSR1 to trigger the restart
    # Return immediately; the restart machinery handles the response
    ...
```

The AI agent's system prompt is updated to instruct: "To restart the gateway, use the `restart_gateway` tool. Do NOT use terminal to run restart commands."

### Secondary: Module-level regex fallback with `reason` field (Approach 6)

Move `_RESTART_TRIGGERS` to module level for test import. Add a `reason` field to `.terminal_pending` markers. The terminal tool sets `reason="restart_trigger"` if regex matches (backward compat) OR if a sentinel is present. This handles the migration period.

```python
# Module-level constant, importable by tests
RESTART_TRIGGER_PATTERNS = (
    r"^tyagent\s+gateway\s+restart",
    r"^(?:sudo\s+)?systemctl\s+(--user\s+)?restart\s+tyagent-gateway",
    r"^uv\s+run\s+python3(?:\.\d+)?\s+tyagent_cli\.py\s+gateway\s+restart",
    r"^python3(?:\.\d+)?\s+tyagent_cli\.py\s+gateway\s+restart",
    r"^uv\s+run\s+tyagent\s+gateway\s+restart",
    r"kill\s+-SIGUSR1\s+\S+",
)
```

### Tertiary: Unified markers (Approach 6 — full)

Merge `.gateway_interrupt/` into `.terminal_pending/` by adding a `reason` field. During `_write_restart_marker`, iterate ALL `.terminal_pending/` markers and classify by `reason`. This eliminates the separate `.gateway_interrupt/` directory and its cleanup logic.

### Migration Path

1. **Phase 1 (now):** Move `_RESTART_TRIGGERS` to module level. Update tests to import it. Add `reason` field to `.terminal_pending`.
2. **Phase 2 (next):** Add `restart_gateway` tool. Update system prompt. Keep regex as fallback.
3. **Phase 3 (later):** Deprecate regex detection. Rely on `restart_gateway` tool and `/restart` command exclusively. Remove `_RESTART_TRIGGERS`.
4. **Phase 4 (cleanup):** Merge `.gateway_interrupt/` into `.terminal_pending/`. Remove `_cleanup_gateway_interrupt_dir`.

### Trade-Off Summary

| Approach | Correctness | Simplicity | Backward Compat | Effort |
|----------|-------------|------------|-----------------|--------|
| Regex (current) | ⭐⭐ | ⭐⭐⭐ | ✅ | Low |
| Whitelist/Flag (1) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ needs fallback | Medium |
| Process Tree (2) | ⭐ | ⭐⭐ | ❌ timing issue | High |
| Signal-Based (3) | ⭐⭐ | ⭐⭐ | ❌ can't distinguish restart vs crash | Medium |
| Sentinel File (4) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ race condition | Low |
| Wrapper/Dedicated Tool (5) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ needs prompt update | Medium |
| Unified Markers (6) | N/A (orthogonal) | ⭐⭐⭐⭐⭐ | ✅ additive | Low |
| **Hybrid (1+5+6)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ phased | Medium |
