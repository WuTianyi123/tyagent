"""Tests for v2 sub-agent tools — spawn, wait, close, list, send_input,
send_message, followup_task, resume_agent."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tyagent.agent import TyAgent
from tyagent.tools import delegate_tool
from tyagent.tools.delegate_tool import (
    CHILD_BLOCKED_TOOLS,
    DEFAULT_SUBAGENT_MAX_TOOL_TURNS,
    ROOT_PATH,
    _handle_close_task,
    _handle_followup_task,
    _handle_list_tasks,
    _handle_resume_agent,
    _handle_send_input,
    _handle_send_message,
    _handle_spawn_task,
    _handle_wait_task,
)
from tyagent.tools.registry import registry

# Only async tests need this — sync registration/schema tests work without
pytestmark = pytest.mark.asyncio

# ── Helpers ────────────────────────────────────────────────


def _make_agent(**overrides) -> MagicMock:
    """Create a minimal TyAgent mock for testing."""
    agent = MagicMock(spec=TyAgent)
    agent.model = overrides.get("model", "test-model")
    agent.api_key = overrides.get("api_key", "test-key")
    agent.base_url = overrides.get("base_url", "https://test.api/v1")
    agent.system_prompt = overrides.get("system_prompt", "You are a test agent.")
    agent.reasoning_effort = overrides.get("reasoning_effort", None)
    agent._compression_config = overrides.get("compression", None)
    agent._task_path = ROOT_PATH
    agent._task_tree = delegate_tool.TaskTree()
    agent._mailbox = delegate_tool.Mailbox(owner_path=ROOT_PATH)
    agent._child_agents = {}
    agent._bg_tasks = {}
    agent._tool_progress_callback = None
    agent.home_dir = None
    agent.context_length = None
    pass
    agent._http_timeout = 120.0
    agent._shutdown_timeout = 5.0
    agent.send_message = AsyncMock()
    agent.close = AsyncMock()
    return agent


# ═══════════════════════════════════════════════════════════
# Registration & schema
# ═══════════════════════════════════════════════════════════


class TestV2ToolsRegistration:
    def test_all_eight_tools_registered(self):
        names = registry.get_all_names()
        for tool in ("spawn_task", "wait_task", "close_task", "list_tasks",
                     "send_input", "send_message", "followup_task", "resume_agent"):
            assert tool in names, f"Missing tool: {tool}"

    def test_spawn_schema_requires_task_name_and_goal(self):
        schemas = registry.get_definitions(names=["spawn_task"])
        fn = schemas[0]["function"]
        required = fn["parameters"]["required"]
        assert "task_name" in required
        assert "goal" in required

    def test_spawn_schema_has_fork_turns(self):
        schemas = registry.get_definitions(names=["spawn_task"])
        fn = schemas[0]["function"]
        props = fn["parameters"]["properties"]
        assert "fork_turns" in props

    def test_wait_v2_schema_no_targets(self):
        schemas = registry.get_definitions(names=["wait_task"])
        fn = schemas[0]["function"]
        # v2: no "targets" or "task_ids" — only timeout
        assert "targets" not in fn["parameters"].get("required", [])
        assert "task_ids" not in fn["parameters"].get("required", [])
        assert "timeout" in fn["parameters"]["properties"]

    def test_close_schema_has_target_required(self):
        schemas = registry.get_definitions(names=["close_task"])
        fn = schemas[0]["function"]
        assert "target" in fn["parameters"]["required"]

    def test_list_tasks_schema_has_path_prefix(self):
        schemas = registry.get_definitions(names=["list_tasks"])
        fn = schemas[0]["function"]
        assert "path_prefix" in fn["parameters"]["properties"]

    def test_send_message_registered(self):
        schemas = registry.get_definitions(names=["send_message"])
        fn = schemas[0]["function"]
        assert "target" in fn["parameters"]["required"]
        assert "message" in fn["parameters"]["required"]

    def test_followup_task_registered(self):
        schemas = registry.get_definitions(names=["followup_task"])
        fn = schemas[0]["function"]
        assert "target" in fn["parameters"]["required"]
        assert "message" in fn["parameters"]["required"]

    def test_resume_agent_registered(self):
        schemas = registry.get_definitions(names=["resume_agent"])
        fn = schemas[0]["function"]
        assert "id" in fn["parameters"]["required"]


# ═══════════════════════════════════════════════════════════
# Blocked tools (only memory for children)
# ═══════════════════════════════════════════════════════════


class TestChildBlockedTools:
    def test_only_memory_is_blocked(self):
        assert CHILD_BLOCKED_TOOLS == frozenset(["memory"])

    def test_spawn_is_not_blocked_for_children(self):
        assert "spawn_task" not in CHILD_BLOCKED_TOOLS

    def test_wait_is_not_blocked_for_children(self):
        assert "wait_task" not in CHILD_BLOCKED_TOOLS


# ═══════════════════════════════════════════════════════════
# spawn_task handler
# ═══════════════════════════════════════════════════════════


class TestSpawnTask:
    async def test_requires_task_name(self):
        result = await _handle_spawn_task({"goal": "do X"}, parent_agent=None)
        data = json.loads(result)
        assert data["error"]

    async def test_requires_goal(self):
        ag = _make_agent()
        result = await _handle_spawn_task({"task_name": "do_x"}, parent_agent=ag)
        data = json.loads(result)
        assert data["error"]

    async def test_returns_task_path(self):
        ag = _make_agent()
        result = await _handle_spawn_task(
            {"task_name": "test_task", "goal": "do something"},
            parent_agent=ag,
        )
        data = json.loads(result)
        assert "task_path" in data
        assert data["task_path"] == "/root/test_task"
        assert data["status"] == "running"

    async def test_rejects_duplicate_task_name(self):
        ag = _make_agent()
        await _handle_spawn_task(
            {"task_name": "dup", "goal": "first"}, parent_agent=ag,
        )
        result = await _handle_spawn_task(
            {"task_name": "dup", "goal": "second"}, parent_agent=ag,
        )
        data = json.loads(result)
        assert data["error"]

    async def test_requires_parent_agent(self):
        result = await _handle_spawn_task(
            {"task_name": "orphan", "goal": "help"}, parent_agent=None,
        )
        data = json.loads(result)
        assert data["error"]
        assert "session agent" in str(data["error"])


# ═══════════════════════════════════════════════════════════
# wait_task handler
# ═══════════════════════════════════════════════════════════


class TestWaitTask:
    async def test_waits_for_mailbox(self):
        ag = _make_agent()
        # Pre-fill mailbox with a notification
        ag._mailbox.send(delegate_tool.FinalNotification(
            task_path="/root/test", success=True,
            summary="all done", error=None, duration_seconds=1.0,
        ))
        result = await _handle_wait_task({"timeout": 1}, parent_agent=ag)
        data = json.loads(result)
        assert data["timed_out"] is False
        assert "completions" in data
        assert any(c["task_path"] == "/root/test" for c in data["completions"])

    async def test_requires_parent_agent(self):
        result = await _handle_wait_task({"timeout": 1}, parent_agent=None)
        data = json.loads(result)
        assert data["error"]


# ═══════════════════════════════════════════════════════════
# close_task handler
# ═══════════════════════════════════════════════════════════


class TestCloseTask:
    async def test_close_existing(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "child", agent=None)
        result = await _handle_close_task({"target": "child"}, parent_agent=ag)
        data = json.loads(result)
        assert "previous_status" in data

    async def test_close_not_found(self):
        ag = _make_agent()
        result = await _handle_close_task({"target": "ghost"}, parent_agent=ag)
        data = json.loads(result)
        assert not data["success"]

    async def test_requires_parent_agent(self):
        result = await _handle_close_task({"target": "x"}, parent_agent=None)
        data = json.loads(result)
        assert data["error"]


# ═══════════════════════════════════════════════════════════
# list_tasks handler
# ═══════════════════════════════════════════════════════════


class TestListTasks:
    async def test_list_empty(self):
        ag = _make_agent()
        result = await _handle_list_tasks({}, parent_agent=ag)
        data = json.loads(result)
        assert data["agents"] == []

    async def test_list_with_agents(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "a", agent=None)
        ag._task_tree.register(ROOT_PATH, "b", agent=None)
        result = await _handle_list_tasks({}, parent_agent=ag)
        data = json.loads(result)
        assert len(data["agents"]) == 2
        names = {a["agent_name"] for a in data["agents"]}
        assert "/root/a" in names
        assert "/root/b" in names

    async def test_list_with_path_prefix(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "main", agent=None)
        ag._task_tree.register("/root/main", "sub", agent=None)
        ag._task_tree.register(ROOT_PATH, "side", agent=None)

        result = await _handle_list_tasks(
            {"path_prefix": "/root/main"}, parent_agent=ag,
        )
        data = json.loads(result)
        names = {a["agent_name"] for a in data["agents"]}
        assert "/root/main" in names
        assert "/root/main/sub" in names
        assert "/root/side" not in names


# ═══════════════════════════════════════════════════════════
# send_input handler
# ═══════════════════════════════════════════════════════════


class TestSendInput:
    async def test_send_to_running_child(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "child", agent=None)

        child_mock = MagicMock(spec=TyAgent)
        child_mock.send_message = AsyncMock()
        ag._child_agents["/root/child"] = child_mock

        result = await _handle_send_input(
            {"target": "child", "message": "rethink!"},
            parent_agent=ag,
        )
        data = json.loads(result)
        assert data["success"] is True
        child_mock.send_message.assert_called_once()

    async def test_child_not_found(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "child", agent=None)
        result = await _handle_send_input(
            {"target": "ghost", "message": "hi"},
            parent_agent=ag,
        )
        data = json.loads(result)
        assert not data["success"]


# ═══════════════════════════════════════════════════════════
# send_message handler
# ═══════════════════════════════════════════════════════════


class TestSendMessage:
    async def test_send_message_no_trigger(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "child", agent=None)
        child_mock = MagicMock(spec=TyAgent)
        child_mock.send_message = AsyncMock()
        ag._child_agents["/root/child"] = child_mock

        result = await _handle_send_message(
            {"target": "child", "message": "fyi"},
            parent_agent=ag,
        )
        data = json.loads(result)
        assert data["success"] is True
        assert data["status"] == "delivered"


# ═══════════════════════════════════════════════════════════
# followup_task handler
# ═══════════════════════════════════════════════════════════


class TestFollowupTask:
    async def test_followup_triggers_turn(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "child", agent=None)
        child_mock = MagicMock(spec=TyAgent)
        child_mock._mailbox = delegate_tool.Mailbox()
        child_mock.send_message = AsyncMock()
        ag._child_agents["/root/child"] = child_mock

        result = await _handle_followup_task(
            {"target": "child", "message": "new task"},
            parent_agent=ag,
        )
        data = json.loads(result)
        assert data["success"] is True
        # Mailbox should have received an InterAgentMessage with trigger_turn=True
        assert child_mock._mailbox.peek() is True


# ═══════════════════════════════════════════════════════════
# resume_agent handler
# ═══════════════════════════════════════════════════════════


class TestResumeAgent:
    async def test_resume_not_supported_yet(self):
        ag = _make_agent()
        ag._task_tree.register(ROOT_PATH, "old", agent=None)
        ag._task_tree.set_status("/root/old", "completed")
        result = await _handle_resume_agent({"id": "/root/old"}, parent_agent=ag)
        data = json.loads(result)
        assert not data["success"]
        assert "not yet supported" in str(data["error"])
