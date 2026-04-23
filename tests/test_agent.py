"""Unit tests for ty_agent.agent — TyAgent LLM interface."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ty_agent.agent import AgentError, TyAgent


# ---------------------------------------------------------------------------
# TyAgent.__init__
# ---------------------------------------------------------------------------


class TestTyAgentInit:
    def test_defaults(self):
        agent = TyAgent.__new__(TyAgent)
        TyAgent.__init__(agent)
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.api_key == ""
        assert "openai.com" in agent.base_url
        assert agent.max_turns == 50
        assert agent.max_tool_turns == 30

    def test_env_var_fallback(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            agent = TyAgent()
            assert agent.api_key == "test-key-123"

    def test_explicit_params(self):
        agent = TyAgent(
            model="gpt-4",
            api_key="my-key",
            base_url="https://custom.api/v1",
            system_prompt="Custom prompt",
        )
        assert agent.model == "gpt-4"
        assert agent.api_key == "my-key"
        assert agent.base_url == "https://custom.api/v1"
        assert agent.system_prompt == "Custom prompt"


# ---------------------------------------------------------------------------
# TyAgent.from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config(self):
        cfg = MagicMock(
            model="test-model",
            api_key="cfg-key",
            base_url="https://cfg.api/v1",
            max_turns=100,
            system_prompt="Cfg prompt",
        )
        agent = TyAgent.from_config(cfg)
        assert agent.model == "test-model"
        assert agent.api_key == "cfg-key"
        assert agent.base_url == "https://cfg.api/v1"
        assert agent.max_turns == 100
        assert agent.system_prompt == "Cfg prompt"


# ---------------------------------------------------------------------------
# TyAgent.chat — basic text response
# ---------------------------------------------------------------------------


class TestChatBasic:
    @pytest.mark.asyncio
    async def test_simple_response(self):
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [{"role": "user", "content": "hi"}]
            result = await agent.chat(messages)

        assert result == "Hello world"
        assert len(messages) == 3  # system + user + assistant
        assert messages[0]["role"] == "system"
        assert messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_system_prompt_injected(self):
        agent = TyAgent(
            api_key="key", base_url="https://api.test/v1", system_prompt="Be helpful"
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [{"role": "user", "content": "hi"}]
            await agent.chat(messages)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_existing_system_prompt_not_duplicated(self):
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [
                {"role": "system", "content": "Existing"},
                {"role": "user", "content": "hi"},
            ]
            await agent.chat(messages)

        assert len(messages) == 3
        assert messages[0]["content"] == "Existing"


# ---------------------------------------------------------------------------
# TyAgent.chat — error handling
# ---------------------------------------------------------------------------


class TestChatErrors:
    @pytest.mark.asyncio
    async def test_http_error_raises_agent_error(self):
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limited"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=mock_resp
        )

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            with pytest.raises(AgentError, match="429"):
                await agent.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_error_raises_agent_error(self):
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(AgentError, match="ConnectError"):
                await agent.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# TyAgent.chat — tool calling loop
# ---------------------------------------------------------------------------


class TestChatToolLoop:
    @pytest.mark.asyncio
    async def test_tool_call_and_response(self):
        """Agent calls a tool, gets result, then responds."""
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")

        tool_resp = MagicMock()
        tool_resp.status_code = 200
        tool_resp.raise_for_status = MagicMock()
        tool_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/test.txt"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        final_resp = MagicMock()
        final_resp.status_code = 200
        final_resp.raise_for_status = MagicMock()
        final_resp.json.return_value = {
            "choices": [{"message": {"content": "File contents: hello"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [tool_resp, final_resp]
            with patch("ty_agent.tools.registry.registry") as mock_reg:
                mock_reg.dispatch.return_value = "hello"
                mock_reg.get_emoji.return_value = "📄"
                messages = [{"role": "user", "content": "read the file"}]
                result = await agent.chat(
                    messages,
                    tools=[{"type": "function", "function": {"name": "read_file"}}],
                )

        assert result == "File contents: hello"
        assert len(messages) == 5  # system, user, assistant(tool), tool, assistant
        assert messages[3]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_malformed_tool_call(self):
        """Malformed tool call returns error but loop continues."""
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")

        tool_resp = MagicMock()
        tool_resp.status_code = 200
        tool_resp.raise_for_status = MagicMock()
        tool_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_bad",
                                "type": "unknown_type",
                                "function": {"name": "", "arguments": ""},
                            }
                        ],
                    }
                }
            ]
        }

        final_resp = MagicMock()
        final_resp.status_code = 200
        final_resp.raise_for_status = MagicMock()
        final_resp.json.return_value = {
            "choices": [{"message": {"content": "done"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [tool_resp, final_resp]
            with patch("ty_agent.tools.registry.registry") as mock_reg:
                messages = [{"role": "user", "content": "test"}]
                result = await agent.chat(
                    messages, tools=[{"type": "function", "function": {}}]
                )

        assert result == "done"

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self):
        """Invalid JSON in tool arguments returns error."""
        agent = TyAgent(api_key="key", base_url="https://api.test/v1")

        tool_resp = MagicMock()
        tool_resp.status_code = 200
        tool_resp.raise_for_status = MagicMock()
        tool_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_json",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "not valid json{",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        final_resp = MagicMock()
        final_resp.status_code = 200
        final_resp.raise_for_status = MagicMock()
        final_resp.json.return_value = {
            "choices": [{"message": {"content": "handled"}}]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [tool_resp, final_resp]
            with patch("ty_agent.tools.registry.registry") as mock_reg:
                messages = [{"role": "user", "content": "test"}]
                result = await agent.chat(
                    messages, tools=[{"type": "function", "function": {}}]
                )

        assert result == "handled"
        tool_msg = messages[3]
        assert tool_msg["role"] == "tool"
        assert "Invalid JSON" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_max_tool_turns(self):
        """Agent stops after max_tool_turns."""
        agent = TyAgent(
            api_key="key", base_url="https://api.test/v1", max_tool_turns=2
        )

        tool_resp = MagicMock()
        tool_resp.status_code = 200
        tool_resp.raise_for_status = MagicMock()
        tool_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "partial",
                        "tool_calls": [
                            {
                                "id": "tc_loop",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/test"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        with patch.object(agent._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = tool_resp
            with patch("ty_agent.tools.registry.registry") as mock_reg:
                mock_reg.dispatch.return_value = "data"
                mock_reg.get_emoji.return_value = "📄"
                messages = [{"role": "user", "content": "loop"}]
                result = await agent.chat(
                    messages, tools=[{"type": "function", "function": {}}]
                )

        assert result == "partial"
        assert mock_post.call_count == 2  # initial + 1 tool turn (2nd hit max_tool_turns)


# ---------------------------------------------------------------------------
# TyAgent.close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self):
        agent = TyAgent(api_key="key")
        with patch.object(
            agent._client, "aclose", new_callable=AsyncMock
        ) as mock_close:
            await agent.close()
            mock_close.assert_called_once()
