"""Tests for the agent loop architecture."""
from __future__ import annotations
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from tyagent.agent import TyAgent, AgentOutput, ReplyTarget
from tyagent.events import EventCollector

pytestmark = pytest.mark.asyncio


class TestAgentLifecycle:
    async def test_start_stop(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        assert agent._running is False
        await agent.start()
        assert agent._running is True
        assert agent._loop_task is not None
        await agent.stop()
        assert agent._running is False

    async def test_double_start_is_noop(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        await agent.start()
        await agent.start()  # Should not crash
        await agent.stop()

    async def test_double_stop_is_noop(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        await agent.start()
        await agent.stop()
        await agent.stop()  # Should not crash

    async def test_send_without_start_raises(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        with pytest.raises(RuntimeError):
            await agent.send_message("hello")

    async def test_close_stops_running_agent(self):
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        await agent.start()
        await agent.close()
        assert agent._running is False


class TestAgentOutput:
    async def test_agent_output_dataclass(self):
        rt = ReplyTarget(platform="feishu", chat_id="chat1", message_id="msg1")
        output = AgentOutput(text="hello", reply_target=rt)
        assert output.text == "hello"
        assert output.reply_target is rt

    async def test_agent_output_no_reply_target(self):
        output = AgentOutput(text="auto reply")
        assert output.reply_target is None


class TestRunTurn:
    async def test_run_turn_with_mock(self):
        """_run_turn() calls API and returns content."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        agent._messages = [{"role": "user", "content": "hello"}]

        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hi there!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_response)):
            result = await agent._run_turn()

        assert result == "Hi there!"
        await agent.close()


class TestAgentLoop:
    async def test_inbox_to_output(self):
        """send_message → agent loop → _run_turn → output_queue."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello back!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_response)):
            await agent.start()
            await agent.send_message("hi")
            # Agent loop runs in background
            await asyncio.sleep(0.1)
            # Check output
            output = await agent._output_queue.get()
            assert output.text == "Hello back!"

        await agent.stop()

    async def test_agent_loop_child_completion_triggers_turn(self):
        """Child completion injected via collector triggers a turn."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")
        agent._event_collector = EventCollector()

        # Child completes immediately
        agent._event_collector.notify_child_done("c1", {
            "success": True, "summary": "research done",
            "error": None, "duration_seconds": 0.5,
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Based on research: ..."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch.object(agent._client, "post", AsyncMock(return_value=mock_response)):
            await agent.start()
            # Child completion triggers a turn even without user message
            output = await agent._output_queue.get()
            assert output.text == "Based on research: ..."
            # Verify notification was queued in _messages
            assert any("子代理完成" in m["content"] for m in agent._messages)

        await agent.stop()

    async def test_messages_accumulate_across_turns(self):
        """Messages list persists across turns in the loop."""
        agent = TyAgent(model="test", api_key="k", base_url="http://x")

        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "choices": [{"message": {"content": "First reply"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "choices": [{"message": {"content": "Second reply"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }

        with patch.object(agent._client, "post", AsyncMock(side_effect=[mock_response_1, mock_response_2])):
            await agent.start()

            await agent.send_message("first msg")
            r1 = await agent._output_queue.get()
            assert r1.text == "First reply"

            await agent.send_message("second msg")
            r2 = await agent._output_queue.get()
            assert r2.text == "Second reply"

        await agent.stop()
