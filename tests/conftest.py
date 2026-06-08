"""Shared test infrastructure for tyagent testing pyramid.

Provides FakeLLM (programmable LLM at HTTP boundary), FakeAdapter
(collects sent messages), and reusable fixtures for agent/gateway tests.
"""

from __future__ import annotations

import asyncio
import json as _json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

# Trigger tool registration via side-effect imports
import tyagent.tools  # noqa: E402
from tyagent.agent import TyAgent, AgentError
from tyagent.config import AgentConfig, TyAgentConfig
from tyagent.gateway import Gateway
from tyagent.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)


# ============================================================
# FakeLLM -- programmable LLM at HTTP boundary
# ============================================================

def _sse(data: dict) -> str:
    """Build a single SSE event string."""
    return "data: " + _json.dumps(data) + "\n\n"


class FakeLLMResponse:
    """A single turn's response from the fake LLM."""

    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content
        self.usage = usage or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def to_chat_response(self) -> dict:
        """Return an OpenAI-compatible chat completion response."""
        message: Dict[str, Any] = {}
        if self.content is not None:
            message["content"] = self.content
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return {
            "choices": [{"message": message, "index": 0}],
            "usage": self.usage,
        }


class FakeLLM:
    """Programmable LLM that lives at the HTTP boundary.

    Replaces ``httpx.AsyncClient`` so that TyAgent's entire agent loop
    runs against it, exercising _run_turn(), tool dispatch, message
    accumulation, and error recovery -- all without a real API.
    """

    def __init__(self):
        self._responses: List[FakeLLMResponse] = []
        self._call_count = 0
        self._response_idx = 0
        self._last_payload: Optional[dict] = None
        self._closed = False

    # -- Programming interface --

    def respond(self, text: str, *, usage=None) -> "FakeLLM":
        """Program the next response as a simple text answer."""
        self._responses.append(FakeLLMResponse(content=text, usage=usage))
        return self

    def tool_call(self, name: str, arguments: dict,
                  *, call_id="call_1", usage=None) -> "FakeLLM":
        """Program the next response as a tool call."""
        self._responses.append(FakeLLMResponse(
            tool_calls=[{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": _json.dumps(arguments)},
            }],
            usage=usage,
        ))
        return self

    def tool_calls(self, calls, *, usage=None) -> "FakeLLM":
        """Program next response with multiple tool calls.
        Args:
            calls: list of (name, arguments, call_id) tuples
        """
        tc_list = [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": _json.dumps(args)}}
            for name, args, cid in calls
        ]
        self._responses.append(FakeLLMResponse(tool_calls=tc_list, usage=usage))
        return self

    def chain(self, *responses: FakeLLMResponse) -> "FakeLLM":
        """Program a sequence of responses for multi-turn conversation."""
        self._responses.extend(responses)
        return self

    def reset(self) -> None:
        """Reset call counters, keep programmed responses."""
        self._call_count = 0
        self._response_idx = 0
        self._last_payload = None

    # -- Inspection --

    @property
    def last_payload(self) -> Optional[dict]:
        return self._last_payload

    @property
    def last_messages(self) -> Optional[List[dict]]:
        if self._last_payload:
            return self._last_payload.get("messages", [])
        return None

    def last_user_message(self) -> Optional[str]:
        msgs = self.last_messages
        if msgs:
            for m in msgs:
                if m.get("role") == "user":
                    return m.get("content", "")
        return None

    def call_count(self) -> int:
        return self._call_count

    # -- httpx.AsyncClient interface (post with payload= not json= to avoid shadowing) --

    async def post(self, url: str, *, headers=None, json=None, **kw) -> "FakeResponse":
        """Imitate httpx.AsyncClient.post()."""
        self._call_count += 1
        self._last_payload = json

        if self._response_idx >= len(self._responses):
            return FakeResponse(500, _json.dumps({
                "error": "No programmed response at call {}".format(self._call_count)
            }))

        resp = self._responses[self._response_idx]
        self._response_idx += 1

        if resp.content and resp.content.startswith("__ERROR__:"):
            parts = resp.content.split(":", 2)
            status = int(parts[1])
            msg = parts[2] if len(parts) > 2 else "error"
            return FakeResponse(status, msg)

        return FakeResponse(200, _json.dumps(resp.to_chat_response()))

    def stream(self, method, url, *, json=None, headers=None, **kw):
        """Imitate httpx.AsyncClient.stream()."""
        self._call_count += 1
        self._last_payload = json

        if self._response_idx >= len(self._responses):
            return _StreamCtx(500, [_sse({"error": "No programmed responses"})])

        resp = self._responses[self._response_idx]
        self._response_idx += 1

        if resp.content and resp.content.startswith("__ERROR__:"):
            parts = resp.content.split(":", 2)
            return _StreamCtx(int(parts[1]), [])

        # Build SSE events from the response
        sse_lines = []
        body = resp.to_chat_response()
        choice = body.get("choices", [{}])[0]
        delta = choice.get("message", {})

        if delta.get("reasoning_content"):
            sse_lines.append(_sse({
                "choices": [{"delta": {"reasoning_content": delta["reasoning_content"]}, "index": 0}]
            }))

        content = delta.get("content")
        if content:
            chunk_size = max(1, len(content) // 4)
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                sse_lines.append(_sse({
                    "choices": [{"delta": {"content": chunk}, "index": 0}]
                }))

        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                sse_lines.append(_sse({
                    "choices": [{"delta": {"tool_calls": [tc]}, "index": 0}]
                }))

        if body.get("usage"):
            sse_lines.append(_sse({"usage": body["usage"]}))
        sse_lines.append("data: [DONE]\n\n")

        return _StreamCtx(200, sse_lines)

    async def aclose(self) -> None:
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


class FakeResponse:
    """Minimal httpx.Response stand-in."""
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return _json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception("HTTP {}".format(self.status_code))


class _StreamCtx:
    """Async context manager for stream()."""
    def __init__(self, status: int, lines: List[str]):
        self._resp = _StreamResponse(status, lines)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass


class _StreamResponse:
    """Response object yielded by stream() context manager."""
    def __init__(self, status: int, lines: List[str]):
        self.status_code = status
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aiter_raw(self, chunk_size=4096):
        yield b""


# ============================================================
# FakeAdapter -- records sent messages
# ============================================================


class FakeAdapter(BasePlatformAdapter):
    """Test adapter that records sent messages without real platform connection."""

    def __init__(
        self, platform_name="feishu",
        config=None, *, home_dir=None,
    ):
        if config is None:
            config = MagicMock()
            config.app_id = "test_app"
            config.app_secret = "test_secret"
            config.encrypt_key = None
            config.verification_token = None
            config.longevity = None
            config.extra = {}
        super().__init__(config, platform_name=platform_name, home_dir=home_dir)
        self.sent_messages: List[Tuple[str, str, dict]] = []
        self.message_handler: Optional[Callable] = None

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_message(self, target, text, **kwargs) -> SendResult:
        self.sent_messages.append((target, text, kwargs))
        return SendResult(success=True, message_id="fake_msg_{}".format(len(self.sent_messages)))

    def build_session_key(self, event: MessageEvent) -> str:
        return "test:{}".format(event.chat_id)

    def set_message_handler(self, handler) -> None:
        self.message_handler = handler

    def clear(self) -> None:
        self.sent_messages.clear()


# ============================================================
# Helpers
# ============================================================


def make_event(
    text="hello",
    platform="feishu",
    chat_id="chat1",
    sender_id="user1",
    message_id="msg1",
    chat_type="private",
    media_urls=None,
    media_types=None,
) -> MessageEvent:
    """Create a MessageEvent for testing."""
    return MessageEvent(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id=sender_id,
        message_id=message_id,
        text=text,
        message_type=MessageType.TEXT,
        media_urls=media_urls or [],
        media_types=media_types or [],
        raw_message={"test": True},
    )


def make_test_agent(
    fake_llm: FakeLLM,
    *,
    model="test-model",
    api_key="test-key",
    base_url="http://fake.test/v1",
    max_tool_turns=50,
    system_prompt="You are a helpful test assistant.",
    home_dir=None,
) -> TyAgent:
    """Create a TyAgent wired to a FakeLLM at the HTTP boundary.

    Also monkey-patches clone() so that gateway per-session cloning
    preserves the FakeLLM client.
    """
    agent = TyAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tool_turns=max_tool_turns,
        system_prompt=system_prompt,
        home_dir=home_dir,
    )
    agent._client = fake_llm  # type: ignore[assignment]

    # Monkey-patch clone() so gateway per-session agent creation
    # also gets the FakeLLM instead of a real httpx client.
    _orig_clone = agent.clone
    def _cloning_clone() -> TyAgent:
        child = _orig_clone()
        child._client = fake_llm  # type: ignore[assignment]
        return child
    agent.clone = _cloning_clone  # type: ignore[method-assign]

    return agent


# ============================================================
# Pytest fixtures
# ============================================================


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Fresh FakeLLM for each test."""
    return FakeLLM()


@pytest.fixture
def test_agent(fake_llm: FakeLLM, tmp_path: Path) -> TyAgent:
    """TyAgent wired to FakeLLM with isolated home_dir."""
    return make_test_agent(fake_llm, home_dir=tmp_path / "home")


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    """Fresh FakeAdapter for each test."""
    return FakeAdapter("feishu")


@pytest.fixture
def test_agent_config(tmp_path: Path) -> AgentConfig:
    """Minimal AgentConfig for gateway tests."""
    return AgentConfig(
        model="test-model",
        api_key="test-key",
        base_url="http://fake.test/v1",
        max_tool_turns=50,
        system_prompt="You are a test agent.",
    )
