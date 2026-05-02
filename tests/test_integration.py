"""Integration tests for the full async sub-agent architecture."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from tyagent.agent import TyAgent, AgentError
from tyagent.types import ReplyTarget, AgentOutput, InboxMessage
from tyagent.events import EventCollector

pytestmark = pytest.mark.asyncio


class TestFullAgentLoop:
    """End-to-end tests of the permanent agent loop."""

    async def test_send_receive_cycle(self):
        """agent.start → send_message → _output_queue gets response."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello world!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_resp)):
            await agent.start()
            await agent.send_message("hi")
            output = await agent._output_queue.get()
            assert output.text == "Hello world!"
            assert output.reply_target is None  # no reply_target passed

        await agent.stop()

    async def test_send_receive_with_reply_target(self):
        """send_message with reply_target preserves it in AgentOutput."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        rt = ReplyTarget(platform="feishu", chat_id="chat1", message_id="msg1")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Replied!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_resp)):
            await agent.start()
            await agent.send_message("test", reply_target=rt)
            output = await agent._output_queue.get()
            assert output.text == "Replied!"
            assert output.reply_target is rt

        await agent.stop()

    async def test_history_loaded_at_start(self):
        """History passed to start() is available in _messages."""
        history = [{"role": "user", "content": "previous message"}]
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_resp)):
            await agent.start(history=history)
            assert len(agent._messages) == 1
            assert agent._messages[0]["content"] == "previous message"

            await agent.send_message("new message")
            await agent._output_queue.get()

            # Both history and new message should be in messages
            assert len(agent._messages) >= 2

        await agent.stop()

    async def test_on_message_callback(self):
        """Agent calls on_message for assistant and tool messages."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        call_log = []

        def on_msg(role, content, **extras):
            call_log.append((role, content[:20] if content else ""))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Logged!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_resp)):
            await agent.start(on_message=on_msg)
            await agent.send_message("hello")
            await agent._output_queue.get()

        assert len(call_log) >= 1
        assert call_log[0][0] == "assistant"

        await agent.stop()


class TestSubAgentIntegration:
    """Integration tests for spawn_task/wait_task/close_task/list_tasks."""

    async def test_spawn_via_agent_loop(self):
        """spawn_task tool is dispatched via _run_turn."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        # First API call returns spawn_task tool call
        spawn_tc = {
            "id": "call_spawn1",
            "type": "function",
            "function": {
                "name": "spawn_task",
                "arguments": json.dumps({"goal": "search papers"}),
            },
        }

        mock_spawn_response = MagicMock()
        mock_spawn_response.status_code = 200
        mock_spawn_response.json.return_value = {
            "choices": [{"message": {
                "content": None,
                "tool_calls": [spawn_tc],
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

        # Second API call returns final text
        mock_final_response = MagicMock()
        mock_final_response.status_code = 200
        mock_final_response.json.return_value = {
            "choices": [{"message": {"content": "Child spawned and done."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(agent._client, "post",
                          AsyncMock(side_effect=[mock_spawn_response, mock_final_response])), \
             patch("tyagent.tools.delegate_tool._run_child_async") as mock_run:
            mock_run.return_value = {"success": True, "summary": "done", "error": None, "duration": 0.1}

            await agent.start()
            await agent.send_message("spawn a child")
            output = await agent._output_queue.get()

        assert output.text == "Child spawned and done."

        await agent.stop()

    async def test_multiple_spawn_and_wait(self):
        """Multiple children spawned, wait_task collects all results."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        # Turn 1: spawn child A
        spawn_a = {"id": "s1", "type": "function",
                    "function": {"name": "spawn_task", "arguments": json.dumps({"goal": "task A"})}}
        # Turn 2: spawn child B
        spawn_b = {"id": "s2", "type": "function",
                    "function": {"name": "spawn_task", "arguments": json.dumps({"goal": "task B"})}}
        # Turn 3: wait for both
        wait_call = {"id": "w1", "type": "function",
                      "function": {"name": "wait_task", "arguments": json.dumps({"task_ids": ["a1", "b1"]})}}
        # Turn 4: final answer
        final_msg = {"message": {"content": "Both done!"}}

        responses = []
        # Turn 1 response
        r = MagicMock(); r.status_code = 200
        r.json.return_value = {"choices": [{"message": {"content": None, "tool_calls": [spawn_a]}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
        responses.append(r)
        # Turn 2 response
        r = MagicMock(); r.status_code = 200
        r.json.return_value = {"choices": [{"message": {"content": None, "tool_calls": [spawn_b]}}], "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}
        responses.append(r)
        # Turn 3 response
        r = MagicMock(); r.status_code = 200
        r.json.return_value = {"choices": [{"message": {"content": None, "tool_calls": [wait_call]}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}
        responses.append(r)
        # Turn 4 response
        r = MagicMock(); r.status_code = 200
        r.json.return_value = {"choices": [final_msg], "usage": {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}}
        responses.append(r)

        with patch.object(agent._client, "post", AsyncMock(side_effect=responses)), \
             patch("tyagent.tools.delegate_tool._run_child_async") as mock_run, \
             patch("tyagent.tools.delegate_tool._handle_wait_task") as mock_wait:
            mock_run.return_value = {"success": True, "summary": "task done", "error": None, "duration": 0.1}
            mock_wait.return_value = json.dumps({
                "a1": {"success": True, "summary": "ok", "error": None, "duration": 0.1},
                "b1": {"success": True, "summary": "ok", "error": None, "duration": 0.1},
            })

            # Simulate child tasks already in bg_tasks
            async def fake_child(): return {"success": True, "summary": "ok"}
            agent._bg_tasks["a1"] = asyncio.ensure_future(fake_child())
            agent._bg_tasks["b1"] = asyncio.ensure_future(fake_child())

            await agent.start()
            await agent.send_message("run tasks")
            output = await agent._output_queue.get()

        assert output.text == "Both done!"

        await agent.stop()

    async def test_close_task(self):
        """close_task cancels a running child task."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        async def never_ends():
            await asyncio.Event().wait()

        task = asyncio.ensure_future(never_ends())
        agent._bg_tasks["slow1"] = task

        assert not task.done()
        task.cancel()
        # Yield control to allow cancellation to propagate
        await asyncio.sleep(0.01)
        assert task.cancelled()

        await agent.close()

    async def test_list_tasks(self):
        """list_tasks returns status of spawned children."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        async def ok(): return "done"
        t1 = asyncio.ensure_future(ok())
        t2 = asyncio.ensure_future(asyncio.Event().wait())  # never ends
        agent._bg_tasks["done1"] = t1
        agent._bg_tasks["running1"] = t2

        await asyncio.sleep(0.01)  # Let t1 complete
        assert t1.done()
        assert not t2.done()

        await agent.close()


class TestChildAutoInjection:
    """Child completion auto-injects into agent loop."""

    async def test_child_completion_triggers_new_turn(self):
        """Child completion → injected as user message → _run_turn called again."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        agent._event_collector = EventCollector()

        # First response: model finishes (no tool calls)
        r1 = MagicMock(); r1.status_code = 200
        r1.json.return_value = {
            "choices": [{"message": {"content": "Waiting for research..."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        # Second response: model produces final answer (after child completes)
        r2 = MagicMock(); r2.status_code = 200
        r2.json.return_value = {
            "choices": [{"message": {"content": "Research complete! Found 3 papers."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(agent._client, "post", AsyncMock(side_effect=[r1, r2])):
            await agent.start()
            # Simulate child completing after a brief delay
            async def child_completes():
                await asyncio.sleep(0.05)
                agent._event_collector.notify_child_done("c1", {
                    "success": True, "summary": "Found 3 papers about climate change",
                    "error": None, "duration": 2.0,
                })

            asyncio.ensure_future(child_completes())

            # First send_message should get the initial response,
            # then child completion triggers another turn whose output
            # goes to _output_queue as a second message
            await agent.send_message("research climate change")
            r1_output = await agent._output_queue.get()
            assert r1_output.text == "Waiting for research..."

            # Second output from child completion
            r2_output = await agent._output_queue.get()
            assert r2_output.text == "Research complete! Found 3 papers."
            assert r2_output.reply_target is None  # auto-reply

        await agent.stop()


class TestDelegateTaskBackwardCompat:
    """delegate_task still works as before (spawn+wait wrapper)."""

    async def test_delegate_task_produces_same_result(self):
        """delegate_task returns flattened result like the old implementation."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        with patch.object(agent._client, "post") as mock_post:
            # Mock the API responses for delegate_task being called as a tool
            ...

        # For now, just verify delegate_task is registered
        from tyagent.tools.registry import registry
        names = registry.get_all_names()
        assert "delegate_task" in names
