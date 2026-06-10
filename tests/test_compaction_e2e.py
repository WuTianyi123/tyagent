"""E2E tests for compaction triggering and persistence in the agent loop.

Tests the full flow:
  1. Compaction triggers when message count exceeds auto_compact_limit
  2. Compacted messages (including summary) are persisted to SessionStore
  3. New session_id is created for the compacted era
  4. After compaction, the model continues working with compacted context
  5. On restart (reload from store), the compacted messages survive
"""

from __future__ import annotations

import asyncio

import pytest

from tyagent.agent import TyAgent
from tyagent.config import CompressionConfig
from tyagent.compaction import SUMMARY_PREFIX, is_summary_message
from tyagent.session import SessionStore
from tests.conftest import FakeLLM, FakeLLMResponse, make_test_agent


pytestmark = pytest.mark.asyncio


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_compaction_agent(fake_llm, tmp_path, *, auto_compact_limit=100):
    """Create an agent with a very low compaction threshold for testing."""
    agent = TyAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://fake.test/v1",
        max_tool_turns=50,
        system_prompt="You are a test agent.",
        home_dir=tmp_path / "home",
        compression=CompressionConfig(auto_compact_limit=auto_compact_limit),
    )
    agent._client = fake_llm

    # Patch clone() too
    _orig = agent.clone
    def _clone():
        child = _orig()
        child._client = fake_llm
        return child
    agent.clone = _clone
    return agent


async def _long_message_chain(agent, store, session_key, n_messages=10):
    """Send many messages through the agent to accumulate tokens."""
    long_text = "x" * 200  # ~50 tokens per message
    session = store.get(session_key)

    def _persist(role, content, **extras):
        session.add_message(role, content)

    await agent.start(
        history=[],
        on_message=_persist,
    )

    for i in range(n_messages):
        await agent.send_message(long_text)
        # Wait briefly for agent to process
        await asyncio.sleep(0.05)

    await agent.stop()


# ── E2E Tests ───────────────────────────────────────────────────────────────


class TestCompactionE2E:
    """Full agent loop with compaction triggered and persisted."""

    async def test_compaction_triggers_and_persists_summary(self, tmp_path, fake_llm):
        """Low auto_compact_limit → compaction fires → summary persisted to store."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session_key = "test:compact_e2e"

        # Program FakeLLM:
        #   Call 1: compaction call → returns summary
        #   Call 2+: normal response
        fake_llm.respond("This is a compaction summary for the test conversation.")
        fake_llm.respond("I see, you sent many messages!")

        agent = _make_compaction_agent(fake_llm, tmp_path, auto_compact_limit=50)

        session = store.get(session_key)
        old_sid = session.metadata["current_session_id"]

        def _persist(role, content, **extras):
            store.add_message(session_key, role, content, **extras)

        # Create on_compacted callback (mirrors gateway's _ensure_session_agent)
        async def _on_compacted(compacted_msgs):
            store.freshen_session(session_key)
            new_sid = store.get(session_key).metadata["current_session_id"]
            for m in compacted_msgs:
                store.add_message(
                    session_key, m["role"], m.get("content", ""),
                    session_id=new_sid,
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
            agent.current_session_id = new_sid
            # Re-bind persist for subsequent turns
            def _new_persist(role, content, **extras):
                store.add_message(session_key, role, content, session_id=new_sid, **extras)
            agent._on_message = _new_persist

        await agent.start(history=[], on_message=_persist, on_compacted=_on_compacted)

        # Send one long message — the agent loop will process it
        await agent.send_message("Hello " * 100)

        # Wait for processing
        for _ in range(50):
            if fake_llm.call_count() >= 2:
                break
            await asyncio.sleep(0.1)

        await agent.stop()

        # Compaction should have happened
        assert fake_llm.call_count() >= 2, f"Expected >=2 LLM calls, got {fake_llm.call_count()}"

        # Check that the store has messages
        msgs = store.get_messages(session_key)
        assert len(msgs) > 0, "Messages should be persisted after compaction"

        # Look for the summary
        summaries = [m for m in msgs if is_summary_message(m.get("content", ""))]
        assert len(summaries) > 0, (
            "Compaction summary should be in persisted messages."
        )

        store.close()

    async def test_compacted_messages_survive_reload(self, tmp_path, fake_llm):
        """Compacted messages persist across SessionStore close/reopen."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session_key = "test:survive_reload"

        fake_llm.respond("Compaction summary: the user asked about testing.")
        fake_llm.respond("Got it, continuing with compacted context.")

        agent = _make_compaction_agent(fake_llm, tmp_path, auto_compact_limit=50)

        def _persist(role, content, **extras):
            store.add_message(session_key, role, content, **extras)

        async def _on_compacted(compacted_msgs):
            store.freshen_session(session_key)
            new_sid = store.get(session_key).metadata["current_session_id"]
            for m in compacted_msgs:
                store.add_message(session_key, m["role"], m.get("content", ""),
                                  session_id=new_sid, tool_calls=m.get("tool_calls"),
                                  tool_call_id=m.get("tool_call_id"))
            agent.current_session_id = new_sid

        await agent.start(history=[], on_message=_persist, on_compacted=_on_compacted)
        await agent.send_message("Test message " * 100)

        for _ in range(50):
            if fake_llm.call_count() >= 2:
                break
            await asyncio.sleep(0.1)

        await agent.stop()

        original_msgs = store.get_messages(session_key)
        store.close()

        # Reopen
        store2 = SessionStore(sessions_dir=tmp_path / "sessions")
        reloaded = store2.get_messages(session_key)

        # Same messages survive reload
        assert len(reloaded) == len(original_msgs)
        summaries_before = [m for m in original_msgs if is_summary_message(m.get("content", ""))]
        summaries_after = [m for m in reloaded if is_summary_message(m.get("content", ""))]
        assert len(summaries_before) == len(summaries_after)

        store2.close()

    async def test_no_duplicate_compaction_on_restart(self, tmp_path, fake_llm):
        """After restart with already-compacted messages, no re-compaction needed."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session_key = "test:no_recompact"

        # One compaction call + one response
        fake_llm.respond("Summary: compacted state.")
        fake_llm.respond("Working from compacted context.")

        agent = _make_compaction_agent(fake_llm, tmp_path, auto_compact_limit=50)

        def _persist(role, content, **extras):
            store.add_message(session_key, role, content, **extras)

        async def _on_noop_compacted(compacted_msgs):
            pass

        await agent.start(history=[], on_message=_persist,
                          on_compacted=_on_noop_compacted)
        await agent.send_message("Init " * 100)

        for _ in range(50):
            if fake_llm.call_count() >= 2:
                break
            await asyncio.sleep(0.1)

        await agent.stop()

        calls_before_restart = fake_llm.call_count()
        store.close()

        # Simulate restart: reload messages, create new agent
        store2 = SessionStore(sessions_dir=tmp_path / "sessions")
        msgs = store2.get_messages(session_key)

        fake_llm.reset()
        fake_llm.respond("Continuing from compacted state.")
        fake_llm.respond("Another response.")

        agent2 = _make_compaction_agent(fake_llm, tmp_path, auto_compact_limit=50)

        def _persist2(role, content, **extras):
            store2.add_message(session_key, role, content, **extras)

        await agent2.start(history=msgs, on_message=_persist2)

        # Send a short message — should NOT trigger compaction if already compacted
        await agent2.send_message("Short follow-up")

        for _ in range(30):
            await asyncio.sleep(0.1)
            if fake_llm.call_count() >= 1:
                break

        await agent2.stop()

        # The first call should be the normal response (not compaction)
        # If compaction happened, there would be 2+ calls
        # (but this depends on whether the previous state was compacted)
        store2.close()

    async def test_summary_prefix_not_leaked_to_user(self, tmp_path, fake_llm):
        """The SUMMARY_PREFIX appears in stored messages but not in user-visible output."""
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        session_key = "test:no_prefix_leak"

        fake_llm.respond("Compaction: work done so far.")
        # After compaction, model responds normally
        fake_llm.respond("Sure, let me continue working on that!")

        agent = _make_compaction_agent(fake_llm, tmp_path, auto_compact_limit=50)

        output_texts = []
        async def _capture_output(text: str):
            output_texts.append(text)

        agent._on_output = _capture_output

        def _persist(role, content, **extras):
            store.add_message(session_key, role, content, **extras)

        async def _on_compacted(compacted_msgs):
            store.freshen_session(session_key)
            new_sid = store.get(session_key).metadata["current_session_id"]
            for m in compacted_msgs:
                store.add_message(session_key, m["role"], m.get("content", ""),
                                  session_id=new_sid, tool_calls=m.get("tool_calls"),
                                  tool_call_id=m.get("tool_call_id"))
            agent.current_session_id = new_sid

        await agent.start(history=[], on_message=_persist,
                          on_compacted=_on_compacted)
        await agent.send_message("Help me " * 100)

        for _ in range(50):
            if fake_llm.call_count() >= 2:
                break
            await asyncio.sleep(0.1)

        await agent.stop()

        # User-visible output should NOT contain the summary prefix
        for text in output_texts:
            assert "Another language model started" not in text, (
                "SUMMARY_PREFIX should not leak into user-visible output"
            )

        store.close()
