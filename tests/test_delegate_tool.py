"""Tests for the delegate_task tool — child agent spawning."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tyagent.agent import TyAgent
from tyagent.tools import delegate_tool
from tyagent.tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DEFAULT_SUBAGENT_MAX_TOOL_TURNS,
    _handle_delegate_task,
    _handle_spawn_task,
    _handle_wait_task,
    _handle_close_task,
    _handle_list_tasks,
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


async def _call_handler(args: dict, parent_agent=None) -> dict:
    """Call _handle_delegate_task and parse JSON result."""
    result = await _handle_delegate_task(args, parent_agent=parent_agent)
    return json.loads(result)


# ---------------------------------------------------------------------------
# Registration & schema
# ---------------------------------------------------------------------------


class TestDelegateTaskRegistration:
    def test_tool_registered(self):
        assert "delegate_task" in registry.get_all_names()

    def test_schema_has_goal_required(self):
        schemas = registry.get_definitions(names=["delegate_task"])
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert "goal" in fn["parameters"]["required"]
        assert "goal" in fn["parameters"]["properties"]

    def test_schema_has_max_tool_turns_with_bounds(self):
        schemas = registry.get_definitions(names=["delegate_task"])
        fn = schemas[0]["function"]
        mt = fn["parameters"]["properties"]["max_tool_turns"]
        assert mt["minimum"] == 1
        assert mt["maximum"] == 200


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestDelegateTaskValidation:
    @pytest.mark.asyncio
    async def test_missing_goal(self):
        result = await _call_handler({}, parent_agent=_make_agent())
        assert "error" in result
        assert "goal" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_goal(self):
        result = await _call_handler({"goal": "  "}, parent_agent=_make_agent())
        assert "error" in result
        assert "goal" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_parent_agent(self):
        result = await _call_handler({"goal": "do something"})
        assert "error" in result
        assert "parent" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_max_tool_turns_not_integer(self):
        result = await _call_handler(
            {"goal": "test", "max_tool_turns": "abc"},
            parent_agent=_make_agent(),
        )
        assert "error" in result
        assert "integer" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_max_tool_turns_zero(self):
        result = await _call_handler(
            {"goal": "test", "max_tool_turns": 0},
            parent_agent=_make_agent(),
        )
        assert "error" in result
        assert "at least 1" in result["error"]

    @pytest.mark.asyncio
    async def test_max_tool_turns_negative(self):
        result = await _call_handler(
            {"goal": "test", "max_tool_turns": -5},
            parent_agent=_make_agent(),
        )
        assert "error" in result
        assert "at least 1" in result["error"]

    @pytest.mark.asyncio
    async def test_max_tool_turns_exceeds_max(self):
        result = await _call_handler(
            {"goal": "test", "max_tool_turns": 999},
            parent_agent=_make_agent(),
        )
        assert "error" in result
        assert "200" in result["error"]


# ---------------------------------------------------------------------------
# Toolsets filtering
# ---------------------------------------------------------------------------


class TestDelegateTaskToolsetFiltering:
    @pytest.mark.asyncio
    async def test_unknown_toolset_removed(self):
        """Toolsets not in registry are silently dropped."""
        agent = _make_agent()
        with patch.object(delegate_tool, "_handle_spawn_task") as mock_spawn, \
             patch.object(delegate_tool, "_handle_wait_task") as mock_wait:
            mock_spawn.return_value = json.dumps({"task_id": "abc", "status": "running"})
            mock_wait.return_value = json.dumps({
                "abc": {"success": True, "summary": "ok", "error": None, "duration_seconds": 1.0}
            })
            result = await _call_handler(
                {"goal": "test", "toolsets": ["nonexistent_tool", "read_file"]},
                parent_agent=agent,
            )
        # Should succeed — nonexistent_tool is just dropped, read_file stays
        assert "error" not in result or result.get("error") is None
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Blocked tools
# ---------------------------------------------------------------------------


class TestDelegateTaskBlockedTools:
    def test_delegate_task_is_blocked(self):
        assert "delegate_task" in DELEGATE_BLOCKED_TOOLS

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

    def test_blocked_tools_is_immutable(self):
        with pytest.raises(Exception):
            DELEGATE_BLOCKED_TOOLS.remove("memory")


# ---------------------------------------------------------------------------
# _handle_delegate_task — backward compat (spawn+wait wrapper)
# ---------------------------------------------------------------------------


class TestDelegateTaskBackwardCompat:
    @pytest.mark.asyncio
    async def test_delegate_task_spawns_and_waits(self):
        agent = _make_agent()
        with patch.object(delegate_tool, "_handle_spawn_task") as mock_spawn, \
             patch.object(delegate_tool, "_handle_wait_task") as mock_wait:
            mock_spawn.return_value = json.dumps({"task_id": "abc123", "status": "running"})
            mock_wait.return_value = json.dumps({
                "abc123": {"success": True, "summary": "ok", "error": None, "duration_seconds": 1}
            })

            result = json.loads(await _handle_delegate_task({"goal": "test"}, parent_agent=agent))

        assert result["success"] is True
        assert result["summary"] == "ok"

    @pytest.mark.asyncio
    async def test_delegate_task_error_on_spawn_failure(self):
        agent = _make_agent()
        result = json.loads(await _handle_delegate_task({"goal": ""}, parent_agent=agent))
        assert "error" in result


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


# ---------------------------------------------------------------------------
# Subagent progress relay via spawn_task
# ---------------------------------------------------------------------------


class TestSubagentProgressRelay:
    """Tests for child agent tool progress relay via spawn_task."""

    @pytest.mark.asyncio
    async def test_parent_callback_is_read(self):
        """Parent's _tool_progress_callback is available when spawning."""
        parent_cb = MagicMock()
        agent = _make_agent()
        agent._tool_progress_callback = parent_cb

        with patch.object(delegate_tool, "_handle_spawn_task") as mock_spawn, \
             patch.object(delegate_tool, "_handle_wait_task") as mock_wait:
            mock_spawn.return_value = json.dumps({"task_id": "abc", "status": "running"})
            mock_wait.return_value = json.dumps({
                "abc": {"success": True, "summary": "ok", "error": None, "duration_seconds": 1}
            })
            await _call_handler({"goal": "test"}, parent_agent=agent)

        # spawn_task is called with the parent agent; it reads the callback internally
        mock_spawn.assert_called_once()
        assert mock_spawn.call_args[1].get("parent_agent") is agent

    @pytest.mark.asyncio
    async def test_no_callback_when_parent_has_none(self):
        """When parent has no progress callback, spawn_task still works."""
        agent = _make_agent()

        with patch.object(delegate_tool, "_handle_spawn_task") as mock_spawn, \
             patch.object(delegate_tool, "_handle_wait_task") as mock_wait:
            mock_spawn.return_value = json.dumps({"task_id": "abc", "status": "running"})
            mock_wait.return_value = json.dumps({
                "abc": {"success": True, "summary": "ok", "error": None, "duration_seconds": 1}
            })
            await _call_handler({"goal": "test"}, parent_agent=agent)

        mock_spawn.assert_called_once()
        assert mock_spawn.call_args[1].get("parent_agent") is agent
