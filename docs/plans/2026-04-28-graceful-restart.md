# Graceful Restart + Crash Recovery Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add graceful drain + self-restart to ty-agent gateway, so restarting doesn't lose tool call state or corrupt message chains. Based on Hermes Agent's proven pattern (SIGUSR1 → drain → mark resume_pending → .clean_shutdown → exit 75 → systemd restarts).

**Architecture:** Add SIGUSR1 handler to gateway. On signal: stop accepting new messages, wait for in-flight agents, mark interrupted sessions as `resume_pending` in metadata, write `.clean_shutdown` marker, exit with code 75. On startup: check marker to distinguish clean vs crash restart; if crash, suspend recently-active sessions so they get a clean slate.

**Tech Stack:** Python 3.11, asyncio, systemd user services, SQLite (sessions already persisted)

**Approach reference:** Hermes Agent (`gateway/run.py`, `gateway/session.py`, `gateway/restart.py`, `gateway/config.py`)

---

### Task 1: Add `resume_pending`/`suspended` fields + restore methods to SessionStore

**Objective:** Add session recovery metadata and query methods — no schema migration needed, use `metadata` JSON field.

**Files:**
- Modify: `tyagent/session.py` — add methods to SessionStore
- Test: `tests/test_session.py` — add tests for new methods

**Changes to `tyagent/session.py`:**

Add to SessionStore class:

```python
def mark_resume_pending(self, session_key: str, reason: str = "restart_timeout") -> bool:
    """Mark a session as resumable after restart interruption.
    The next get() call returns the same session (no auto-reset).
    Returns True if the session existed and was marked."""
    session_dict, _ = self._db.get_or_create_session(session_key)
    metadata = json.loads(session_dict.get("metadata", "{}"))
    if metadata.get("suspended"):
        return False  # Never override explicit suspension
    metadata["resume_pending"] = True
    metadata["resume_reason"] = reason
    metadata["resume_marked_at"] = time.time()
    self._db._update_session_metadata(session_key, metadata)
    return True

def clear_resume_pending(self, session_key: str) -> bool:
    """Clear the resume-pending flag after a successful resumed turn."""
    session_dict, _ = self._db.get_or_create_session(session_key)
    metadata = json.loads(session_dict.get("metadata", "{}"))
    if not metadata.pop("resume_pending", False):
        return False
    metadata.pop("resume_reason", None)
    metadata.pop("resume_marked_at", None)
    self._db._update_session_metadata(session_key, metadata)
    return True

def suspend_session(self, session_key: str, reason: str = "crash_recovery") -> bool:
    """Mark a session as suspended (will auto-reset on next message).
    Returns True if the session existed and was marked."""
    session_dict, _ = self._db.get_or_create_session(session_key)
    metadata = json.loads(session_dict.get("metadata", "{}"))
    metadata["suspended"] = True
    metadata["suspend_reason"] = reason
    metadata["suspend_at"] = time.time()
    self._db._update_session_metadata(session_key, metadata)
    return True

def suspend_recently_active(self, max_age_seconds: int = 120) -> int:
    """Mark recently-active sessions as suspended.
    Called on gateway startup after unexpected exit.
    Skips sessions that have resume_pending=True.
    Returns count of sessions suspended."""
    cutoff = time.time() - max_age_seconds
    count = 0
    for key in self._db.get_all_session_keys():
        session_dict, _ = self._db.get_or_create_session(key)
        metadata = json.loads(session_dict.get("metadata", "{}"))
        if metadata.get("resume_pending"):
            continue
        if metadata.get("suspended"):
            continue
        updated_at = session_dict.get("updated_at", 0)
        if updated_at >= cutoff:
            self.suspend_session(key, "crash_recovery")
            count += 1
    return count

def is_suspended(self, session_key: str) -> bool:
    """Check if a session is suspended."""
    session_dict, _ = self._db.get_or_create_session(session_key)
    metadata = json.loads(session_dict.get("metadata", "{}"))
    return bool(metadata.get("suspended"))

def is_resume_pending(self, session_key: str) -> bool:
    """Check if a session has recovery pending."""
    session_dict, _ = self._db.get_or_create_session(session_key)
    metadata = json.loads(session_dict.get("metadata", "{}"))
    return bool(metadata.get("resume_pending"))
```

Also add to `Database` class in `tyagent/db.py`:

```python
def _update_session_metadata(self, session_key: str, metadata: dict) -> None:
    """Update the metadata JSON field of a session."""
    with self._lock:
        self._conn.execute(
            "UPDATE sessions SET metadata = ?, updated_at = ? WHERE session_key = ?",
            (json.dumps(metadata, ensure_ascii=False), time.time(), session_key),
        )
        self._conn.commit()

def get_all_session_dicts(self) -> List[Dict[str, Any]]:
    """Return all session rows as dicts."""
    with self._lock:
        cur = self._conn.execute(
            "SELECT session_key, created_at, updated_at, metadata FROM sessions ORDER BY updated_at DESC"
        )
        return [_row_to_session(row) for row in cur.fetchall()]
```

**Step 1: Write failing tests** — Write tests in `tests/test_session.py`:
- test_mark_resume_pending: mark a session, verify metadata contains resume_pending
- test_clear_resume_pending: clear it, verify cleared
- test_suspend_session: suspend, verify metadata
- test_suspend_recently_active: create a session updated <120s ago, call suspend_recently_active(), verify suspended
- test_suspend_recently_active_skips_resume_pending: create session with resume_pending=True, verify not suspended
- test_is_suspended / test_is_resume_pending

**Step 2: Run tests** — verify FAIL

**Step 3: Implement** — Add the code as described above to `session.py` and `db.py`

**Step 4: Run tests** — verify PASS

**Step 5: Commit**
```bash
git add -A && git commit -m "feat: add session recovery metadata (resume_pending/suspended)"
```

---

### Task 2: Add graceful drain + SIGUSR1 handler to Gateway

**Objective:** Add graceful restart mechanism to Gateway class. On SIGUSR1: drain in-flight agents, mark interrupted sessions, write clean_shutdown marker, exit(75).

**Files:**
- Modify: `tyagent/gateway/gateway.py`
- Test: `tests/test_gateway.py`

**Changes to `tyagent/gateway/gateway.py`:**

1. Add imports at top: `import signal`, `import json`, `from pathlib import Path`

2. Add to `__init__`:
```python
self._restart_requested = False
self._draining = False
self._active_sessions: set[str] = set()
self._restart_drain_timeout: float = 60.0  # default, configurable later
```

3. Add after `__init__` (or in `start()`):
```python
def _setup_signal_handlers(self) -> None:
    """Register signal handlers for graceful shutdown and restart."""
    if hasattr(signal, "SIGUSR1"):
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGUSR1, self._on_sigusr1)
            logger.info("Registered SIGUSR1 handler for graceful restart")
        except Exception as e:
            logger.warning("Failed to register SIGUSR1 handler: %s", e)
```

4. Add signal handler methods:
```python
def _on_sigusr1(self) -> None:
    """Called when SIGUSR1 is received. Triggers graceful restart."""
    if self._restart_requested or self._draining:
        logger.info("SIGUSR1 ignored — restart already in progress")
        return
    logger.info("SIGUSR1 received — initiating graceful restart")
    self._restart_requested = True
    self._draining = True
    asyncio.create_task(self._do_graceful_restart())

async def _do_graceful_restart(self) -> None:
    """Graceful restart: drain agents, mark sessions, exit cleanly."""
    try:
        # Notify active sessions
        await self._notify_active_sessions_of_restart()
        
        # Drain: wait for in-flight agents
        timeout = self._restart_drain_timeout
        timed_out = await self._drain_active_agents(timeout)
        
        if timed_out:
            # Mark forcibly-interrupted sessions for recovery
            logger.warning(
                "Drain timed out after %.1fs with %d active session(s); marking for resume",
                timeout, len(self._active_sessions),
            )
            for key in list(self._active_sessions):
                try:
                    self.session_store.mark_resume_pending(key, "restart_timeout")
                except Exception as e:
                    logger.debug("mark_resume_pending failed for %s: %s", key, e)
        
        # Write clean-shutdown marker so next startup knows it wasn't a crash
        clean_marker = Path.home() / ".tyagent" / ".clean_shutdown"
        if not timed_out:
            clean_marker.parent.mkdir(parents=True, exist_ok=True)
            clean_marker.touch()
            logger.info("Written clean_shutdown marker")
        else:
            logger.warning(
                "Skipping clean_shutdown marker — drain timed out; "
                "next startup will suspend recently active sessions"
            )
        
        logger.info("Graceful restart complete — exiting with code 75")
        os._exit(75)
    except Exception as e:
        logger.exception("Graceful restart failed: %s", e)
        os._exit(75)

async def _drain_active_agents(self, timeout: float) -> bool:
    """Wait up to `timeout` seconds for active agents to finish.
    Returns True if any sessions were still active after timeout."""
    from asyncio import get_running_loop, sleep
    if not self._active_sessions:
        return False
    deadline = get_running_loop().time() + timeout
    while self._active_sessions and get_running_loop().time() < deadline:
        await sleep(0.1)
    return bool(self._active_sessions)

async def _notify_active_sessions_of_restart(self) -> None:
    """Send a notification to every chat with an active agent."""
    action = "restarting"
    hint = (
        "Your current task will be interrupted. "
        "Send any message after restart to resume."
    )
    msg = f"⚠️ Gateway {action} — {hint}"
    for session_key in list(self._active_sessions):
        for name, adapter in self.adapters.items():
            try:
                await adapter.send_message("", msg)
            except Exception:
                pass

def _check_recovery_on_startup(self) -> None:
    """On startup, check if previous exit was clean or crash."""
    clean_marker = Path.home() / ".tyagent" / ".clean_shutdown"
    if clean_marker.exists():
        logger.info("Previous gateway exited cleanly — skipping session suspension")
        try:
            clean_marker.unlink()
        except Exception:
            pass
    else:
        try:
            suspended = self.session_store.suspend_recently_active()
            if suspended:
                logger.info(
                    "Suspended %d in-flight session(s) from previous (unclean) run",
                    suspended,
                )
        except Exception as e:
            logger.warning("Session suspension on startup failed: %s", e)
```

5. Modify `_on_message` to check drain + recover suspended sessions:

In `_on_message`, after `session_key = adapter.build_session_key(event)` and before `session = self.session_store.get(session_key)`:

```python
# Check if gateway is draining
if self._draining:
    await adapter.send_message(
        event.chat_id or "",
        "⏳ Gateway is restarting — please wait a moment and try again.",
        reply_to_message_id=event.message_id,
    )
    return None

# Check if session was suspended (crash recovery)
session = self.session_store.get(session_key)
if self.session_store.is_suspended(session_key):
    logger.info("Session %s was suspended — auto-resetting", session_key)
    self.session_store.archive(session_key)
    session = self.session_store.get_or_create_after_archive(session_key)
    # Send notification
    await adapter.send_message(
        event.chat_id or "",
        "◐ Session automatically reset (previous session was interrupted). "
        "Send any message to start fresh.",
        reply_to_message_id=event.message_id,
    )
    return None
```

Wait, actually the session handling flow in _on_message is:
```python
session_key = adapter.build_session_key(event)
session = self.session_store.get(session_key)
```

I need to insert the suspended check here. But `get()` always returns a session (creates if none). So I need to check after get().

Actually, let me rethink. The `get()` method calls `self._db.get_or_create_session()` which always returns a session entry. The suspended flag is in metadata. So:

```python
session_key = adapter.build_session_key(event)
if not self._draining and self.session_store.is_suspended(session_key):
    # Auto-reset suspended session
    self.session_store.archive(session_key)
    session = self.session_store.get_or_create_after_archive(session_key)
    # Notify user
    ...
    return None
session = self.session_store.get(session_key)
```

Also track active sessions in `_on_message`:

```python
self._active_sessions.add(session_key)
try:
    ... agent.chat(...) ...
finally:
    self._active_sessions.discard(session_key)
```

6. Modify `start()` to call `_setup_signal_handlers()` and `_check_recovery_on_startup()`:

In `start()`, add after `self._load_adapters()`:
```python
self._setup_signal_handlers()
self._check_recovery_on_startup()
```

Also modify `_on_message` to track active sessions:
Wrap the agent.chat() call region with:
```python
self._active_sessions.add(session_key)
try:
    # ... (existing agent.chat code)
finally:
    self._active_sessions.discard(session_key)
```

**Step 1: Write failing tests** — Write tests:
- test_drain_no_active_sessions: drain with empty active set → returns immediately
- test_signal_handler: mock SIGUSR1, verify restart triggered
- test_clean_shutdown_marker: after graceful restart, marker file created
- test_suspended_session_auto_reset: session marked suspended → on message → archived + fresh created

**Step 2: Run tests** — verify FAIL

**Step 3: Implement** — Add code as described

**Step 4: Run tests** — verify PASS

**Step 5: Commit**
```bash
git add -A && git commit -m "feat: add graceful drain + SIGUSR1 restart to gateway"
```

---

### Task 3: Update systemd service unit + CLI restart command

**Objective:** Update systemd unit with `RestartForceExitStatus=75` so it restarts on graceful exit. Make `tyagent gateway restart` send SIGUSR1 instead of `systemctl restart`.

**Files:**
- Modify: `tyagent/service_manager.py`
- Modify: `tyagent_cli.py`

**Changes:**

In `service_manager.py`:
- Add `RestartForceExitStatus=75` to generated unit
- Keep `TimeoutStopSec=60` or adjust to allow drain timeout + headroom

In `tyagent_cli.py`:
- `restart` command: find gateway PID, send SIGUSR1, wait briefly, verify new process starts
- Update help text

**Step 1: Implement changes directly** (no new tests needed — existing tests cover service management)

**Step 2: Verify unit generation output**

**Step 3: Commit**
```bash
git add -A && git commit -m "feat: systemd RestartForceExitStatus=75 + SIGUSR1 restart CLI"
```

---

### Task 4: Final integration — verify tests pass, run full suite

**Objective:** Verify all tests pass, all components work together, no regressions.

**Steps:**
1. Run full test suite: `python -m pytest tests/ -q -o 'addopts='`
2. Check git status for any unexpected files
3. Commit any final adjustments

**Deliverables:** All 370+ tests pass, git working tree clean.
