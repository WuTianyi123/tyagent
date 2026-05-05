"""Tests for async sub-agent tools — spawn, wait, close, list."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tyagent.agent import TyAgent
from tyagent.tools import delegate_tool
from tyagent.tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DEFAULT_SUBAGENT_MAX_TOOL_TURNS,
    _handle_close_task,
    _handle_list_tasks,
    _handle_spawn_task,
    _handle_wait_task,
    _run_child_async,
)
from tyagent.tools.registry import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(**overrides) -> MagicMock:
    """Create a minimal TyAgent mock for testing."""
    agent = MagicMock(spec=TyAgent)
    agent.model = overrides.get("model", "test-model")
    agent.api_key = overrides.get("api_key", "test-key")
    agent.base_url = overrides.get("base_url", "https://test.api/v1")
    agent.system_prompt = overrides.get("system_prompt", "You are a test agent.")
    agent.reasoning_effort = overrides.get("reasoning_effort", None)
    agent._compression_config = overrides.get("compression", None)
    return agent


# ---------------------------------------------------------------------------
# Registration & schema
# ---------------------------------------------------------------------------


class TestAsyncToolsRegistration:
    def test_spawn_task_registered(self):
        assert "spawn_task" in registry.get_all_names()

    def test_wait_task_registered(self):
        assert "wait_task" in registry.get_all_names()

    def test_close_task_registered(self):
        assert "close_task" in registry.get_all_names()

    def test_list_tasks_registered(self):
        assert "list_tasks" in registry.get_all_names()

    def test_spawn_schema_has_goal_required(self):
        schemas = registry.get_definitions(names=["spawn_task"])
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert "goal" in fn["parameters"]["required"]
        assert "goal" in fn["parameters"]["properties"]

    def test_spawn_schema_has_max_tool_turns_with_bounds(self):
        schemas = registry.get_definitions(names=["spawn_task"])
        fn = schemas[0]["function"]
        mt = fn["parameters"]["properties"]["max_tool_turns"]
        assert mt["minimum"] == 1
        assert mt["maximum"] == 200

    def test_wait_schema_has_task_ids_required(self):
        schemas = registry.get_definitions(names=["wait_task"])
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert "task_ids" in fn["parameters"]["required"]

    def test_close_schema_has_task_id_required(self):
        schemas = registry.get_definitions(names=["close_task"])
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert "task_id" in fn["parameters"]["required"]


# ---------------------------------------------------------------------------
# Blocked tools
# ---------------------------------------------------------------------------


class TestBlockedTools:
    def test_memory_is_blocked(self):
        assert "memory" in DELEGATE_BLOCKED_TOOLS

    def test_spawn_task_is_blocked(self):
        assert "spawn_task" in DELEGATE_BLOCKED_TOOLS

    def test_wait_task_is_blocked(self):
        assert "wait_task" in DELEGATE_BLOCKED_TOOLS

    def test_close_task_is_blocked(self):
        assert "close_task" in DELEGATE_BLOCKED_TOOLS

    def test_list_tasks_is_blocked(self):
        assert "list_tasks" in DELEGATE_BLOCKED_TOOLS

    def test_send_input_is_blocked(self):
        assert "send_input" in DELEGATE_BLOCKED_TOOLS

    def test_blocked_tools_is_immutable(self):
        with pytest.raises(Exception):
            DELEGATE_BLOCKED_TOOLS.remove("memory")


# ---------------------------------------------------------------------------
# send_input — mid-turn messaging to child agents
# ---------------------------------------------------------------------------


class TestSendInput:
    """Tests for _handle_send_input — sending messages to running children."""

    @pytest.mark.asyncio
    async def test_valid_send(self):
        """send_input puts a message in the child's inbox."""
        child = MagicMock(spec=TyAgent)
        child.send_message = AsyncMock()
        parent = _make_agent()
        parent._child_agents = {"abc": child}

        from tyagent.tools.delegate_tool import _handle_send_input
        result = json.loads(await _handle_send_input(
            {"target": "abc", "message": "Hello"}, parent_agent=parent,
        ))
        assert result["success"] is True
        assert result["target"] == "abc"
        child.send_message.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_interrupt_flag(self):
        """interrupt=true is accepted and reflected in status."""
        child = MagicMock(spec=TyAgent)
        child.send_message = AsyncMock()
        parent = _make_agent()
        parent._child_agents = {"abc": child}

        from tyagent.tools.delegate_tool import _handle_send_input
        result = json.loads(await _handle_send_input(
            {"target": "abc", "message": "H", "interrupt": True},
            parent_agent=parent,
        ))
        assert result["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_missing_target(self):
        """Missing target yields an error."""
        from tyagent.tools.delegate_tool import _handle_send_input
        result = json.loads(await _handle_send_input(
            {"message": "Hi"}, parent_agent=_make_agent(),
        ))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_message(self):
        """Missing message yields an error."""
        from tyagent.tools.delegate_tool import _handle_send_input
        result = json.loads(await _handle_send_input(
            {"target": "abc"}, parent_agent=_make_agent(),
        ))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_target(self):
        """Non-existent task_id yields an error."""
        parent = _make_agent()
        parent._child_agents = {}
        from tyagent.tools.delegate_tool import _handle_send_input
        result = json.loads(await _handle_send_input(
            {"target": "nonexistent", "message": "Hi"},
            parent_agent=parent,
        ))
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_no_parent_agent(self):
        """Missing parent_agent yields an error."""
        from tyagent.tools.delegate_tool import _handle_send_input
        result = json.loads(await _handle_send_input(
            {"target": "abc", "message": "Hi"},
        ))
        assert "error" in result

    def test_send_input_registered(self):
        """send_input is registered as a tool."""
        from tyagent.tools.registry import registry
        assert "send_input" in registry.get_all_names()

    def test_send_input_has_required_params(self):
        """send_input schema has target and message as required."""
        from tyagent.tools.registry import registry
        schemas = registry.get_definitions(names=["send_input"])
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert "target" in fn["parameters"]["required"]
        assert "message" in fn["parameters"]["required"]


# ---------------------------------------------------------------------------
# _run_child_async — async child agent runner
# ---------------------------------------------------------------------------


class TestRunChildAsync:
    """Tests for the async child agent runner."""

    @pytest.mark.asyncio
    async def test_successful_child_run(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="Child completed task successfully.")
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            result = await _run_child_async(
                task_id="t1",
                model="test", api_key="k", base_url="http://x",
                system_prompt="Test system prompt", reasoning_effort=None,
                goal="Do a thing", tool_names=["read_file", "write_file"],
                max_tool_turns=30, context=None,
            )

        assert result["success"] is True
        assert result["summary"] == "Child completed task successfully."
        assert result["error"] is None
        assert "duration_seconds" in result

    @pytest.mark.asyncio
    async def test_child_run_with_context(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="done")
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="Base prompt.", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30,
                context="Extra context here",
            )

            _, kwargs = mock_cls.call_args
            system = kwargs["system_prompt"]
            assert "Base prompt." in system
            assert "Extra context here" in system

    @pytest.mark.asyncio
    async def test_child_run_without_context_uses_pure_system_prompt(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="done")
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="Pure prompt.", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30, context=None,
            )

            _, kwargs = mock_cls.call_args
            assert kwargs["system_prompt"] == "Pure prompt."

    @pytest.mark.asyncio
    async def test_child_failure_captured(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(side_effect=RuntimeError("Boom"))
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            result = await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="p", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30, context=None,
            )

        assert result["success"] is False
        assert "Boom" in result["error"]

    @pytest.mark.asyncio
    async def test_child_close_failure_is_silent(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="done")
            mock_child.close = AsyncMock(side_effect=RuntimeError("Close failed"))
            mock_cls.return_value = mock_child

            result = await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="p", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30, context=None,
            )

        assert result["success"] is True
        assert result["summary"] == "done"

    @pytest.mark.asyncio
    async def test_empty_summary_stripped(self):
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="   ")
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            result = await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="p", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30, context=None,
            )

        assert result["summary"] == ""

    @pytest.mark.asyncio
    async def test_notifies_collector(self):
        collector = MagicMock()
        with patch("tyagent.tools.delegate_tool.registry.get_definitions", return_value=[]), \
             patch("tyagent.tools.delegate_tool.TyAgent") as mock_cls:
            mock_child = MagicMock()
            mock_child.chat = AsyncMock(return_value="done")
            mock_child.close = AsyncMock()
            mock_cls.return_value = mock_child

            await _run_child_async(
                task_id="t1", model="test", api_key="k", base_url="http://x",
                system_prompt="p", reasoning_effort=None,
                goal="task", tool_names=[], max_tool_turns=30,
                context=None, collector=collector,
            )

        collector.notify_child_done.assert_called_once()
        args, _ = collector.notify_child_done.call_args
        assert args[0] == "t1"
