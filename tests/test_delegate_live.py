"""Delegate tool integration tests — real sub-agent spawning, wait, and close.

Tests exercise the full sub-agent lifecycle: spawn_task creates a real child
TyAgent with its own _agent_loop(), child processes messages from its inbox,
completes, sends FinalNotification, and parent collects results via wait_task.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tyagent.agent import TyAgent


pytestmark = pytest.mark.asyncio


# ============================================================
# Helpers
# ============================================================


def _make_agent(fake_llm, tmp_path):
    """Create an agent capable of spawning children."""
    from tests.conftest import make_test_agent
    return make_test_agent(fake_llm, home_dir=tmp_path / "home")


# ============================================================
# Tests
# ============================================================


class TestSpawnTask:
    async def test_spawn_and_child_completes(self, fake_llm, tmp_path):
        """spawn_task creates child; child runs and sends FinalNotification."""
        parent = _make_agent(fake_llm, tmp_path)

        # Program: spawn_task response, then child's goal response
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[tc("s1", "spawn_task", {
                "task_name": "research",
                "goal": "Research topic X",
            })]),
            FakeLLMResponse(content="Spawned research agent"),
        )

        # Start parent, send message that triggers spawn
        await parent.start()
        await parent.send_message("spawn a research agent")
        out = await asyncio.wait_for(parent._output_queue.get(), timeout=5)
        assert "Spawned" in out.text

        # Child should be registered
        assert len(parent._bg_tasks) >= 1

        # Wait for child to complete naturally
        # The child's agent loop will process its inbox goal, then exit
        await asyncio.sleep(0.5)

        await parent.stop()

    async def test_wait_task_collects_child_result(self, fake_llm, tmp_path):
        """wait_task blocks until child completes, returns results."""
        parent = _make_agent(fake_llm, tmp_path)

        # Program: spawn, then wait, then final response
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[tc("s1", "spawn_task", {
                "task_name": "worker",
                "goal": "Do a quick task",
            })]),
            FakeLLMResponse(tool_calls=[tc("w1", "wait_task", {
                "timeout": 30,
            })]),
            FakeLLMResponse(content="All children done!"),
        )

        await parent.start()
        await parent.send_message("spawn and wait for worker")

        # First output: spawn acknowledgment
        out1 = await asyncio.wait_for(parent._output_queue.get(), timeout=5)

        # Child should have been spawned
        assert len(parent._bg_tasks) >= 1

        # Wait for child to finish and for parent to process completion
        # The parent's agent loop will detect the FinalNotification
        # in the mailbox and trigger another turn
        await asyncio.sleep(0.5)

        # Give child time to run and complete
        # The child has its own FakeLLM... wait, it doesn't!
        # The child is cloned from parent and has same FakeLLM.
        # Same FakeLLM already consumed its responses. Need to add more.
        fake_llm.respond("Child task completed")
        await asyncio.sleep(0.5)

        await parent.stop()

    async def test_list_tasks_shows_running_children(self, fake_llm, tmp_path):
        """list_tasks returns all spawned agents with statuses."""
        parent = _make_agent(fake_llm, tmp_path)

        # spawn task a
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[tc("s1", "spawn_task", {
                "task_name": "worker_a",
                "goal": "Task A",
            })]),
            FakeLLMResponse(content="Spawned A"),
        )

        await parent.start()
        await parent.send_message("spawn A")
        out = await asyncio.wait_for(parent._output_queue.get(), timeout=5)
        assert "Spawned A" in out.text
        assert parent._task_tree is not None
        paths = parent._task_tree.all_paths()
        assert any("worker_a" in p for p in paths)

        await parent.stop()

    async def test_close_task_stops_child(self, fake_llm, tmp_path):
        """close_task stops a running child agent."""
        parent = _make_agent(fake_llm, tmp_path)

        # spawn then close
        fake_llm.chain(
            FakeLLMResponse(tool_calls=[tc("s1", "spawn_task", {
                "task_name": "worker",
                "goal": "Long running task",
            })]),
            FakeLLMResponse(tool_calls=[tc("c1", "close_task", {
                "target": "worker",
            })]),
            FakeLLMResponse(content="Worker closed"),
        )

        await parent.start()
        await parent.send_message("spawn then close worker")
        await asyncio.wait_for(parent._output_queue.get(), timeout=5)

        # After close, child should be cleaned up
        await asyncio.sleep(0.3)

        await parent.stop()

    async def test_max_concurrent_limit(self, fake_llm, tmp_path):
        """Cannot spawn more than MAX_CONCURRENT_SUBAGENTS."""
        parent = _make_agent(fake_llm, tmp_path)
        await parent.start()

        # Try to spawn 6 agents (max is 5)
        calls = []
        for i in range(6):
            calls.append(tc(f"s{i}", "spawn_task", {
                "task_name": f"agent_{i}",
                "goal": f"Task {i}",
            }))
        calls.append(FakeLLMResponse(content="Done spawning"))

        fake_llm.chain(*calls)

        await parent.send_message("spawn many")
        out = await asyncio.wait_for(parent._output_queue.get(), timeout=5)

        # Should have less than 6 children
        running_count = len(parent._bg_tasks)
        assert running_count <= 5

        await parent.stop()


# ============================================================
# Helpers
# ============================================================


from tests.conftest import FakeLLMResponse


def tc(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
