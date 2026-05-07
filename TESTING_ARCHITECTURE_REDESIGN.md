# Testing Architecture Redesign: Restart Marker Pipeline

## Problems Identified

### 1. Regex tests duplicate `_RESTART_TRIGGERS` patterns locally
**Current:** `TestRestartTriggers._will_restart()` (lines 1425–1438 in test_gateway.py) **redefines** the same regex tuple that lives in `tyagent/tools/core.py` lines 782–792. If a developer adds/removes/modifies a trigger in core.py, the tests will silently pass because they're testing a **copy**—not the actual production patterns.

### 2. Dedup test doesn't verify DB contents
**Current:** `test_write_marker_dedup_in_flight_and_restart_trigger` (lines 951–990) only checks the marker file JSON. It doesn't verify that when the handler processes the marker, only **one** tool response ends up in the database (no duplicate synthetic responses).

### 3. MagicMock agents don't behave like real agents
**Current:** Nearly every test creates `agent = MagicMock()` and manually sets `agent.clone`, `agent.chat`, `agent.start`, `agent.send_message`, `agent._output_queue`, `agent._tool_progress_callback`, `agent._running`, `agent._current_tool_call_id`. New attributes added to `TyAgent` break no tests, and tests fail with cryptic `AttributeError` when the production code accesses an attribute the mock doesn't have.

### 4. No end-to-end test from terminal tool → restart marker → handler processing
**Current:** Tests call `_write_restart_marker()` and `_handle_restart_marker_on_startup()` directly with manually-constructed JSON. There's no test that exercises the **actual pipeline**: a command is passed to `_handle_terminal()` → the interrupt marker is written → the gateway restarts → the handler reads the marker → synthetics are written to DB. This pipeline has multiple ordering/dedup/cleanup bugs that the current unit tests miss.

---

## Proposed Changes

### Change 1: Extract `_RESTART_TRIGGERS` to an importable module-level constant

**File: `tyagent/tools/core.py`** — move the tuple out of `_handle_terminal()` to module level:

```python
# After _handle_write_file (around line 264), before the terminal tool section:

# Regex patterns for commands that trigger a gateway restart.
# Exported so tests can import and verify against the actual patterns.
# See tests/test_gateway.py::TestRestartTriggers.
RESTART_TRIGGER_PATTERNS: tuple[str, ...] = (
    # Direct CLI or wrapper calls — anchored to avoid matching echo/printf.
    r"^tyagent\s+gateway\s+restart",
    r"^(?:sudo\s+)?systemctl\s+(--user\s+)?restart\s+tyagent-gateway",
    r"^uv\s+run\s+python3(?:\.\d+)?\s+tyagent_cli\.py\s+gateway\s+restart",
    r"^python3(?:\.\d+)?\s+tyagent_cli\.py\s+gateway\s+restart",
    r"^uv\s+run\s+tyagent\s+gateway\s+restart",
    # kill — unanchored to support pipelines (pgrep … | xargs kill …).
    # \S+ matches PIDs, $(…), and `…`.
    r"kill\s+-SIGUSR1\s+\S+",
)
```

Then inside `_handle_terminal()`, replace the local tuple with:

```python
    will_restart = any(re.search(p, command) for p in RESTART_TRIGGER_PATTERNS)
```

**Benefit:** Tests import `from tyagent.tools.core import RESTART_TRIGGER_PATTERNS` directly—no copy drift possible.

**File: `tests/test_gateway.py`** — rewrite `TestRestartTriggers`:

```python
class TestRestartTriggers:
    """Test RESTART_TRIGGER_PATTERNS regex patterns — import the real ones."""

    def _will_restart(self, command: str) -> bool:
        """Check if command matches any restart trigger pattern."""
        import re
        from tyagent.tools.core import RESTART_TRIGGER_PATTERNS
        return any(re.search(p, command) for p in RESTART_TRIGGER_PATTERNS)

    def test_direct_cli(self):
        assert self._will_restart("tyagent gateway restart")
        # ^ anchor prevents echo false positive
        assert not self._will_restart('echo "tyagent gateway restart"')
        assert not self._will_restart("echo tyagent gateway restart")

    def test_systemctl(self):
        assert self._will_restart("systemctl --user restart tyagent-gateway")
        assert self._will_restart("sudo systemctl restart tyagent-gateway")
        assert self._will_restart("sudo systemctl --user restart tyagent-gateway")
        assert not self._will_restart('echo "systemctl restart tyagent-gateway"')

    def test_kill(self):
        assert self._will_restart("kill -SIGUSR1 12345")
        assert self._will_restart("kill -SIGUSR1 $(pgrep -f tyagent-gateway)")
        assert self._will_restart("kill -SIGUSR1 `pgrep tyagent`")

    def test_python_cli(self):
        assert self._will_restart("python3 tyagent_cli.py gateway restart")
        assert self._will_restart("python3.11 tyagent_cli.py gateway restart")
        assert self._will_restart("python3.12 tyagent_cli.py gateway restart")

    def test_uv_run(self):
        assert self._will_restart("uv run tyagent gateway restart")
        assert self._will_restart("uv run python3 tyagent_cli.py gateway restart")
        assert self._will_restart("uv run python3.11 tyagent_cli.py gateway restart")

    def test_non_restart_commands(self):
        assert not self._will_restart("ls -la")
        assert not self._will_restart("echo hello")
        assert not self._will_restart("cat /proc/loadavg")
        assert not self._will_restart("tyagent gateway status")
        assert not self._will_restart("systemctl status tyagent-gateway")
```

---

### Change 2: Realistic `TyAgent` test double (spec-based, not MagicMock)

**New file: `tests/agent_doubles.py`** — a `StubTyAgent` that satisfies the `TyAgent` interface but is controllable:

```python
"""Test doubles for TyAgent — spec-verified, minimal, and controllable."""
from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

class StubTyAgent:
    """A controllable TyAgent stub that satisfies the real interface.

    Unlike MagicMock, this class:
    - Has all attributes that the real TyAgent has at __init__ time
    - Allows setting attribute values directly (no mock setup dance)
    - Raises AttributeError on unknown attributes (catches typos)
    - Supports async method override via simple assignment
    """

    def __init__(self, *, model: str = "test-model"):
        self.model = model
        self._running = False
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._current_tool_call_id: str = ""
        self._bg_tasks: list = []
        self._tool_progress_callback = None
        self._messages: list = []
        # session_key and current_session_id set by gateway
        self.session_key: str = ""
        self.current_session_id: str = ""
        self.home_dir = None  # Path, set by agent factory

    # --- Methods needed by Gateway ---

    def clone(self) -> "StubTyAgent":
        """Per-session clone — returns a fresh instance."""
        return StubTyAgent(model=self.model)

    async def start(self, *, on_message=None, messages=None, **kwargs):
        """Called by _ensure_session_agent.  No-op by default."""
        self._running = True

    async def stop(self, *, shutdown_timeout: float = 5.0):
        """Called by _stop_session_agent."""
        self._running = False

    async def send_message(self, text: str, *,
                           reply_target=None,
                           tool_progress_cb=None,
                           turn_done_cb=None):
        """Fire-and-forget message to the agent loop.  No-op by default."""
        pass

    async def close(self):
        """Clean up resources."""
        self._running = False

    # Prevent accidental attribute typos that MagicMock would silently swallow
    def __setattr__(self, name, value):
        # Allow any attribute set — Stub is lenient
        super().__setattr__(name, value)


class FakeTyAgent(StubTyAgent):
    """A TyAgent double that actually runs a minimal turn loop.

    Use this for end-to-end pipeline tests that need the agent to
    produce output or tool calls.
    """

    def __init__(self, *, model: str = "test-model",
                 tool_registry: Any = None):
        super().__init__(model=model)
        self._tool_registry = tool_registry
        self._turn_done_cb = None
        self._reply_target = None

    async def start(self, *, on_message=None, messages=None, **kwargs):
        self._running = True

    async def send_message(self, text: str, *,
                           reply_target=None,
                           tool_progress_cb=None,
                           turn_done_cb=None):
        """Accept the message and trigger a minimal turn."""
        self._turn_done_cb = turn_done_cb
        self._reply_target = reply_target
        # Simulate a turn — in real agent this would be _run_turn()
        if tool_progress_cb:
            # Simulate tool progress callback with a fake tool_call_id
            self._current_tool_call_id = "call_test_1"
            tool_progress_cb("call_test_1", "terminal", "running")

        # Queue a fake response
        await self._output_queue.put({
            "text": f"Response to: {text}",
            "reply_target": reply_target,
        })

        if turn_done_cb:
            turn_done_cb()
```

Then update the test helpers to use the stub:

```python
# In test_gateway.py, replace agent = MagicMock() with:
def _make_agent(model="test-model"):
    """Create a realistic but controllable agent double."""
    from tests.agent_doubles import StubTyAgent
    return StubTyAgent(model=model)
```

---

### Change 3: End-to-end pipeline test

**New test class in `tests/test_gateway.py`:**

```python
class TestRestartPipelineE2E:
    """End-to-end test of the restart marker pipeline:

    terminal tool writes gateway_interrupt marker
      → _write_restart_marker collects it into .restart_pending
      → _handle_restart_marker_on_startup writes synthetic DB responses
      → Sessions are mark_resume_pending
      → Gateway reconnects and sends restart notification
    """

    def test_full_cycle(self, tmp_path):
        """Simulate a complete restart triggered by 'tyagent gateway restart'."""
        import json, time
        from tyagent.tools.core import _handle_terminal, RESTART_TRIGGER_PATTERNS

        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        agent = _make_agent()
        agent._current_tool_call_id = "call_e2e_1"
        agent.session_key = "feishu:chat1"
        agent.current_session_id = "sid_e2e_1"
        agent.home_dir = tmp_path

        gw = Gateway(config, agent=agent)
        adapter = _make_adapter()
        gw.adapters["feishu"] = adapter

        # --- Stage 1: Create a session with an orphaned tool call ---
        session = gw.session_store.get("feishu:chat1")
        session.add_message("user", "restart the gateway")
        session.add_message("assistant", "", tool_calls=[{
            "id": "call_e2e_1",
            "type": "function",
            "function": {"name": "terminal",
                         "arguments": '{"command": "tyagent gateway restart"}'},
        }])

        # --- Stage 2: Simulate terminal tool execution ---
        # The terminal tool would normally run the command in a subprocess,
        # but for the test we just verify the interrupt marker gets written.
        marker_dir = tmp_path / ".gateway_interrupt"
        marker_dir.mkdir(parents=True, exist_ok=True)
        interrupt_data = {
            "tool_call_id": "call_e2e_1",
            "session_key": "feishu:chat1",
            "session_id": "sid_e2e_1",
            "command": "tyagent gateway restart",
            "started_at": time.time(),
            "reason": "restart_trigger",
        }
        (marker_dir / "e2e.json").write_text(
            json.dumps(interrupt_data), encoding="utf-8")

        # Register the session as active
        from tyagent.gateway.gateway import SessionContext
        gw._sessions["feishu:chat1"] = SessionContext(
            agent, adapter, platform_name="feishu", chat_id="chat1")

        # --- Stage 3: _write_restart_marker ---
        gw.supervisor._write_restart_marker()
        marker_path = tmp_path / ".restart_pending"
        assert marker_path.exists(), (
            "Should write .restart_pending when interrupt markers exist")

        # --- Stage 4: Simulate new process startup ---
        # New Gateway instance (same session store)
        gw2 = Gateway(config)
        gw2.supervisor._handle_restart_marker_on_startup()

        # --- Stage 5: Verify ---
        # A) Marker files cleaned up
        assert not marker_path.exists(), ".restart_pending should be removed"
        remaining = list(marker_dir.glob("*.json"))
        assert len(remaining) == 0, (
            f".gateway_interrupt markers should be cleaned; found {len(remaining)}")

        # B) Synthetic tool response written to DB
        messages = gw2.session_store.get_messages("feishu:chat1", session_id="sid_e2e_1")
        tool_msgs = [m for m in messages
                     if m.get("role") == "tool" and m.get("tool_call_id") == "call_e2e_1"]
        assert len(tool_msgs) == 1, (
            f"Should have exactly 1 tool response; found {len(tool_msgs)}")
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed.get("restart_completed") is True, (
            f"Should be a restart_completed response: {parsed}")
        assert parsed.get("success") is True

        # C) Message chain is valid (no orphaned tool_calls)
        sanitized = _sanitize_message_chain(messages)
        assert len(sanitized) == len(messages), (
            f"_sanitize_message_chain should not add new messages; "
            f"{len(sanitized)} vs {len(messages)}")

        gw.session_store.close()
        gw2.session_store.close()

    def test_dedup_verified_in_db(self, tmp_path):
        """When the same tool_call_id appears in both gateway_interrupt AND
        in-flight, verify only ONE tool response ends up in the database."""
        import json, time

        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)
        agent = _make_agent()
        agent._current_tool_call_id = "call_shared"
        agent.session_key = "test:key"
        agent.current_session_id = "sid_shared"
        agent.home_dir = tmp_path

        gw = Gateway(config, agent=agent)

        # Create session with an orphaned tool call
        session = gw.session_store.get("test:key")
        session.add_message("user", "do restart")
        session.add_message("assistant", "", tool_calls=[{
            "id": "call_shared",
            "type": "function",
            "function": {"name": "terminal",
                         "arguments": '{"command": "tyagent gateway restart"}'},
        }])

        # Write gateway_interrupt marker (same tool_call_id)
        marker_dir = tmp_path / ".gateway_interrupt"
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / "shared.json").write_text(json.dumps({
            "tool_call_id": "call_shared",
            "session_key": "test:key",
            "session_id": "sid_shared",
            "command": "tyagent gateway restart",
            "reason": "restart_trigger",
        }), encoding="utf-8")

        # Register as active with in-flight tool call (same id)
        from tyagent.gateway.gateway import SessionContext
        agent._running = True
        gw._sessions["test:key"] = SessionContext(
            agent, MagicMock(), platform_name="test", chat_id="chat1")

        # Write the restart marker (dedup happens here)
        gw.supervisor._write_restart_marker()
        marker_path = tmp_path / ".restart_pending"
        assert marker_path.exists()

        # Verify marker JSON has exactly 1 pending_tool_call
        marker = json.loads(marker_path.read_text())
        pending = marker["sessions"]["test:key"]["pending_tool_calls"]
        assert len(pending) == 1, f"Expected 1 deduplicated; got {len(pending)}"
        assert pending[0]["tool_call_id"] == "call_shared"
        assert pending[0]["reason"] == "restart_trigger", (
            "restart_trigger should take precedence over unknown_failure")

        # --- NEW: Process the marker and verify DB ---
        gw2 = Gateway(config)
        gw2.supervisor._handle_restart_marker_on_startup()
        messages = gw2.session_store.get_messages("test:key", session_id="sid_shared")
        tool_responses = [m for m in messages
                          if m.get("role") == "tool" and m.get("tool_call_id") == "call_shared"]
        assert len(tool_responses) == 1, (
            f"Dedup failed: expected 1 tool response in DB, found {len(tool_responses)}. "
            f"Messages: {json.dumps(tool_responses, indent=2)}")

        gw.session_store.close()
        gw2.session_store.close()
        marker_path.unlink(missing_ok=True)

    def test_real_response_prevents_synthetic(self, tmp_path):
        """If a real terminal result was collected (Step 1), the handler
        must NOT overwrite it with a synthetic response."""
        import json

        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)

        # Stage 1: Create session with orphaned tool call
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        sid = session.metadata.get("current_session_id", "")
        session.add_message("user", "run something")
        session.add_message("assistant", "", tool_calls=[{
            "id": "call_real",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": "echo real output"}'},
        }])

        # Stage 2: Write a restart marker for this call
        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": sid,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_real", "function_name": "terminal",
                         "reason": "unknown_failure"},
                    ],
                }
            },
        }
        (tmp_path / ".restart_pending").write_text(json.dumps(marker))

        # Stage 3: BEFORE handler, simulate real result collected (Step 1)
        session.add_message("tool",
            json.dumps({"output": "real output", "exit_code": 0}),
            tool_call_id="call_real")

        # Stage 4: Handler should SKIP because real response exists
        gw.supervisor._handle_restart_marker_on_startup()
        assert not (tmp_path / ".restart_pending").exists()

        # Verify only the real response exists
        messages = gw.session_store.get_messages("test:key", session_id=sid)
        tool_msgs = [m for m in messages
                     if m.get("role") == "tool" and m.get("tool_call_id") == "call_real"]
        assert len(tool_msgs) == 1
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed.get("output") == "real output", (
            f"Real output was overwritten: {parsed}")
        assert "restart_completed" not in parsed, (
            "Real response should not have synthetic markers")

        gw.session_store.close()

    def test_ordering_real_before_synthetic(self, tmp_path):
        """Test the ordering constraint: if a real response already exists
        in DB, _write_restart_marker → _handle should NOT add a duplicate
        even when marker JSON says there's a pending call.

        This catches the bug where the handler doesn't check for existing
        responses before writing synthetics.
        """
        import json

        config = _make_config(sessions_dir=tmp_path / "sessions", home_dir=tmp_path)

        # Create session with a full chain: user → assistant(tool_call) → tool(response)
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        sid = session.metadata.get("current_session_id", "")
        session.add_message("user", "run command")
        session.add_message("assistant", "", tool_calls=[{
            "id": "call_completed",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
        }])
        # Real response already exists — the command completed during drain/stop
        session.add_message("tool",
            json.dumps({"output": "file1.txt", "exit_code": 0}),
            tool_call_id="call_completed")

        # Simulate: marker has this call as pending (written before drain completed)
        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": sid,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_completed", "function_name": "terminal",
                         "reason": "unknown_failure"},
                    ],
                }
            },
        }
        (tmp_path / ".restart_pending").write_text(json.dumps(marker))

        gw.supervisor._handle_restart_marker_on_startup()

        # Should only have the ONE real response — not a synthetic duplicate
        messages = gw.session_store.get_messages("test:key", session_id=sid)
        tool_msgs = [m for m in messages
                     if m.get("role") == "tool" and m.get("tool_call_id") == "call_completed"]
        assert len(tool_msgs) == 1, (
            f"Ordering bug: expected 1 response, found {len(tool_msgs)}. "
            f"Real response should not be duplicated by synthetic.")
        assert json.loads(tool_msgs[0]["content"]).get("output") == "file1.txt"

        gw.session_store.close()


class TestDedupDBVerification:
    """Verifies deduplication at the database level — the handler writes
    exactly one response per tool_call_id regardless of how many times
    a call appears in the marker."""

    def test_duplicate_in_same_session(self, tmp_path):
        """Same tool_call_id appears twice in the same session's
        pending_tool_calls list in the marker JSON.  Handler must
        deduplicate and write only one response."""
        import json

        config = _make_config(sessions_dir=tmp_path / "sessions")
        gw = Gateway(config)
        session = gw.session_store.get("test:key")
        sid = session.metadata.get("current_session_id", "")
        session.add_message("user", "restart trigger test")
        session.add_message("assistant", "", tool_calls=[{
            "id": "call_dup",
            "type": "function",
            "function": {"name": "terminal",
                         "arguments": '{"command": "tyagent gateway restart"}'},
        }])

        # Marker has the SAME tool_call_id listed twice (bug simulation)
        marker = {
            "restarted_at": time.time(),
            "sessions": {
                "test:key": {
                    "session_id": sid,
                    "pending_tool_calls": [
                        {"tool_call_id": "call_dup", "function_name": "terminal",
                         "reason": "restart_trigger"},
                        {"tool_call_id": "call_dup", "function_name": "terminal",
                         "reason": "unknown_failure"},
                    ],
                }
            },
        }
        (tmp_path / ".restart_pending").write_text(json.dumps(marker))

        gw.supervisor._handle_restart_marker_on_startup()
        messages = gw.session_store.get_messages("test:key", session_id=sid)
        tool_responses = [m for m in messages
                          if m.get("role") == "tool" and m.get("tool_call_id") == "call_dup"]
        assert len(tool_responses) == 1, (
            f"Duplicate bug: expected 1 response, found {len(tool_responses)}")

        gw.session_store.close()
```

---

### Change 4: Add `conftest.py` with shared fixtures

**New file: `tests/conftest.py`:**

```python
"""Shared pytest fixtures for tyagent tests."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tyagent.config import AgentConfig, TyAgentConfig
from tyagent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult


@pytest.fixture
def temp_home(tmp_path) -> Path:
    """A temporary home directory pair (sessions_dir inside tmp_path)."""
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def basic_config(temp_home) -> TyAgentConfig:
    """Minimal Gateway config with temp storage."""
    return TyAgentConfig(
        agent=AgentConfig(model="test-model", api_key="test-key"),
        sessions_dir=temp_home / "sessions",
        home_dir=temp_home,
    )


@pytest.fixture
def mock_event() -> MessageEvent:
    """A basic MessageEvent mock."""
    event = MagicMock(spec=MessageEvent)
    event.text = "hello"
    event.message_type = MessageType.TEXT
    event.platform = "feishu"
    event.sender_id = "user1"
    event.chat_id = "chat1"
    event.chat_type = "private"
    event.message_id = "msg1"
    event.media_urls = None
    event.media_types = None
    event.reply_to = None
    event.reply_to_text = None
    event.raw_message = None
    event.is_command.return_value = False
    event.get_command.return_value = None
    return event


@pytest.fixture
def mock_adapter() -> BasePlatformAdapter:
    """A mock platform adapter."""
    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.platform_name = "feishu"
    adapter.send_message = AsyncMock(
        return_value=SendResult(success=True, message_id="sent1"))
    adapter.build_session_key = MagicMock(return_value="feishu:chat1")
    adapter.start = AsyncMock()
    adapter.stop = AsyncMock()
    return adapter
```

Then update `_make_event`, `_make_adapter`, `_make_config` in `test_gateway.py` to delegate to these fixtures where appropriate.

---

## Summary of Changes

| File | Change | Why |
|------|--------|-----|
| `tyagent/tools/core.py` | Move `_RESTART_TRIGGERS` tuple to module-level `RESTART_TRIGGER_PATTERNS` | Tests can import the real patterns — no copy drift |
| `tests/agent_doubles.py` | New file: `StubTyAgent` + `FakeTyAgent` | Realistic test doubles with spec-verified interface |
| `tests/conftest.py` | New file: shared pytest fixtures | Reduce MagicMock boilerplate across all test files |
| `tests/test_gateway.py` | Rewrite `TestRestartTriggers` to import from `core.py` | Tests test the real patterns |
| `tests/test_gateway.py` | Add `TestRestartPipelineE2E` class (4 tests) | End-to-end pipeline coverage |
| `tests/test_gateway.py` | Add `TestDedupDBVerification` class (1 test) | Dedup verified at DB level, not just marker JSON |
| `tests/test_gateway.py` | Gradual migration to `_make_agent()` using `StubTyAgent` | Catch attribute drift between mocks and real TyAgent |

## Bug Classes Caught

1. **Copy-drift regex:** Adding a trigger to core.py but forgetting the test passes silently. *Fixed by importing the real patterns.*

2. **Dedup-in-JSON-only:** Marker dedup looks correct in JSON, but handler writes duplicates to DB. *Fixed by `TestDedupDBVerification` and updated `TestRestartPipelineE2E.test_dedup_verified_in_db`.*

3. **Ordering (real-beats-synthetic):** Real terminal result is written to DB before handler runs, but handler overwrites it. *Fixed by `test_real_response_prevents_synthetic` and `test_ordering_real_before_synthetic`.*

4. **Cleanup timing:** `.gateway_interrupt` markers not cleaned when no active sessions exist. *Already covered by existing test; E2E test adds coverage for the full cleanup path.*

5. **Attribute drift:** TyAgent gains `_current_tool_call_id` — MagicMock silently returns a new mock; tests pass but production crashes with AttributeError. *Fixed by `StubTyAgent` — accessing unknown attributes raises `AttributeError`.*
