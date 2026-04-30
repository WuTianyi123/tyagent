"""AI Agent interface for tyagent.

Provides a simplified adapter layer for LLM interactions.
Supports OpenAI-compatible APIs, model routing, and function calling (tools).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

# Type alias for the on_message callback
OnMessageCallback = Callable[..., Any]

import httpx

from tyagent.config import CompressionConfig
from tyagent.context import compress_context

logger = logging.getLogger(__name__)


class ContextOverflow(Exception):
    """LLM API returned 400 indicating the context is too long.
    Caught by chat() to trigger single-pass compression and retry.
    """
    pass


# Patterns used to detect context overflow errors in API responses.
# Borrowed from Hermes error_classifier.py.
_CONTEXT_OVERFLOW_PATTERNS = [
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    "max_tokens",
    "maximum number of tokens",
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length",
    "input is too long",
    "maximum model length",
    "context length exceeded",
    "超过最大长度",
    "上下文长度",
    "max input token",
    "exceeds the maximum number of input tokens",
]


def _is_context_overflow(status_code: int, body: str) -> bool:
    """Return True if the API error indicates a context overflow."""
    if status_code != 400:
        return False
    body_lower = body.lower()
    return any(p in body_lower for p in _CONTEXT_OVERFLOW_PATTERNS)


class TyAgent:
    """Simplified AI agent for tyagent.

    Uses OpenAI-compatible chat completions API with optional tool calling.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tool_turns: Optional[int] = 200,
        system_prompt: str = "You are a helpful assistant.",
        reasoning_effort: Optional[str] = "high",
        compression: Optional[CompressionConfig] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_tool_turns = max_tool_turns
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        c = compression or CompressionConfig()
        self.compress_model = c.model
        self.compress_api_key = c.api_key or self.api_key
        self.compress_base_url = c.base_url or self.base_url
        self.compress_context_window = c.context_window
        self.compress_cut_ratio = c.cut_ratio
        self._client = httpx.AsyncClient(timeout=120.0)
        # Real token usage from the last API response
        self.last_usage: Optional[Dict[str, int]] = None
        # Cached system message dict for prompt caching (built once per session)
        self._system_msg: Optional[Dict[str, Any]] = None
        # Boundary index for append-only api_messages mode
        self._prev_msg_count: int = 0
        # Token tracking: (message_count, cumulative_prompt_tokens) per API call.
        # Used by compression to find precise cut points without a tokenizer.
        self._token_history: List[tuple] = []

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        on_message: Optional[OnMessageCallback] = None,
        stream_delta_callback: Optional[Callable[[str], None]] = None,
        on_segment_break: Optional[Callable[[], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send messages to the LLM and return the response text.

        If *tools* are provided, the agent runs a tool-calling loop.
        Context overflow (400 'context too long') triggers single-pass
        compression: a cut point is found at ~cut_ratio of the context
        window using precise token counts, and pre-cut messages are
        summarized by an LLM.
        """
        from tyagent.tools.registry import registry

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "KimiCLI/1.30.0",
        }

        # Inject cached system prompt (built once per session)
        if self._system_msg is None:
            self._system_msg = {"role": "system", "content": self.system_prompt}
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, self._system_msg)

        # Build the message list to send. No proactive compression —
        # we only compress when the API returns 400 (context overflow).
        self._prev_msg_count = 0
        self._token_history = []
        api_messages = list(messages)
        self._prev_msg_count = len(messages)

        payload_base: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        if self.reasoning_effort:
            payload_base["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload_base["tools"] = tools
            payload_base["tool_choice"] = "auto"

        tool_turn = 0
        content: Optional[str] = None
        reasoning_content: Optional[str] = None
        tool_calls = None
        while True:
            if self.max_tool_turns is not None and self.max_tool_turns > 0 and tool_turn >= self.max_tool_turns:
                logger.warning("Max tool turns (%d) reached, returning last content", self.max_tool_turns)
                break

            # Append-only: on subsequent turns, only add new messages
            if tool_turn > 0:
                api_messages.extend(messages[self._prev_msg_count:])
            self._prev_msg_count = len(messages)
            payload_base["messages"] = api_messages

            # Context overflow retry loop — single-pass compression
            _compressed = False
            while True:
                self.last_usage = None  # reset before each API call
                try:
                    if stream:
                        # --- Streaming path ---
                        payload = {
                            **payload_base,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        }
                        async with self._client.stream(
                            "POST", f"{self.base_url}/chat/completions",
                            json=payload, headers=headers,
                        ) as resp:
                            if resp.status_code >= 400:
                                error_body = b""
                                async for chunk in resp.aiter_raw(chunk_size=4096):
                                    error_body += chunk
                                body_str = error_body.decode("utf-8", errors="replace")[:2000]
                                logger.error("LLM API error: %s - %s", resp.status_code, body_str)
                                if _is_context_overflow(resp.status_code, body_str):
                                    raise ContextOverflow(body_str)
                                raise AgentError(f"LLM API returned {resp.status_code}: {body_str}")

                            content_parts: List[str] = []
                            tool_calls_acc: Dict[int, Dict[str, Any]] = {}
                            reasoning_parts: List[str] = []
                            usage_obj: Optional[Dict[str, Any]] = None

                            async for line in resp.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                if not data_str:
                                    continue

                                chunk = json.loads(data_str)
                                if not chunk.get("choices"):
                                    if chunk.get("usage"):
                                        usage_obj = chunk["usage"]
                                    continue

                                delta = chunk["choices"][0].get("delta", {})

                                if delta.get("content"):
                                    content_parts.append(delta["content"])
                                    if not tool_calls_acc and stream_delta_callback:
                                        stream_delta_callback(delta["content"])

                                if delta.get("tool_calls"):
                                    for tc_delta in delta["tool_calls"]:
                                        idx = tc_delta.get("index", 0)
                                        if idx not in tool_calls_acc:
                                            tool_calls_acc[idx] = {
                                                "id": tc_delta.get("id", ""),
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""},
                                            }
                                        acc = tool_calls_acc[idx]
                                        if tc_delta.get("id"):
                                            acc["id"] = tc_delta["id"]
                                        if tc_delta.get("function", {}).get("name"):
                                            acc["function"]["name"] = tc_delta["function"]["name"]
                                        if tc_delta.get("function", {}).get("arguments"):
                                            acc["function"]["arguments"] += tc_delta["function"]["arguments"]

                                if delta.get("reasoning_content"):
                                    reasoning_parts.append(delta["reasoning_content"])
                                    if reasoning_callback:
                                        reasoning_callback(delta["reasoning_content"])

                            content = "".join(content_parts) if content_parts else None
                            reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
                            tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else None

                            if usage_obj:
                                self.last_usage = {
                                    "prompt_tokens": usage_obj.get("prompt_tokens", 0) if isinstance(usage_obj, dict) else getattr(usage_obj, "prompt_tokens", 0),
                                    "completion_tokens": usage_obj.get("completion_tokens", 0) if isinstance(usage_obj, dict) else getattr(usage_obj, "completion_tokens", 0),
                                    "total_tokens": usage_obj.get("total_tokens", 0) if isinstance(usage_obj, dict) else getattr(usage_obj, "total_tokens", 0),
                                }
                    else:
                        # --- Non-streaming path ---
                        resp = await self._client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload_base,
                        )
                        # Check for 400 overflow before raise_for_status so
                        # ContextOverflow is raised inside the try block
                        # (sibling except handlers don't catch re-raised exceptions).
                        if resp.status_code >= 400:
                            body_str = resp.text[:2000]
                            logger.error("LLM API error: %s - %s", resp.status_code, body_str)
                            if _is_context_overflow(resp.status_code, body_str):
                                raise ContextOverflow(body_str)
                            raise AgentError(f"LLM API returned {resp.status_code}: {body_str}")
                        data = resp.json()
                        usage = data.get("usage")
                        if usage:
                            self.last_usage = {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            }

                        choice = data.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        content = message.get("content")
                        tool_calls = message.get("tool_calls")
                        reasoning_content = message.get("reasoning_content")

                    # Record token usage for later cut-point calculation.
                    # Uses api_messages count — maps 1:1 to prompt_tokens.
                    if self.last_usage and self.last_usage.get("prompt_tokens"):
                        self._token_history.append(
                            (len(api_messages), self.last_usage["prompt_tokens"])
                        )

                    break  # API call succeeded

                except ContextOverflow:
                    if not _compressed:
                        _compressed = True
                        logger.info("Context overflow — applying single-pass compression")
                        compressed = await compress_context(
                            api_messages, self._client,
                            model=self.compress_model or self.model,
                            api_key=self.compress_api_key,
                            base_url=self.compress_base_url,
                            token_history=self._token_history,
                            context_window=self.compress_context_window,
                            cut_ratio=self.compress_cut_ratio,
                        )
                        if compressed is not None:
                            api_messages = compressed
                            self._prev_msg_count = len(messages)
                            self._token_history = []
                            payload_base["messages"] = api_messages
                            continue
                    raise AgentError(
                        "Context too long even after compression."
                    )
                except AgentError:
                    raise  # re-raise directly — already an AgentError
                except Exception as exc:
                    logger.exception("LLM request failed")
                    raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc

            # --- Shared assistant message building ---
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content is not None:
                assistant_msg["content"] = content
            if reasoning_content is not None:
                assistant_msg["reasoning_content"] = reasoning_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Persist via callback if provided
            if on_message:
                msg_kwargs: Dict[str, Any] = {}
                if tool_calls:
                    msg_kwargs["tool_calls"] = tool_calls
                if reasoning_content:
                    msg_kwargs["reasoning"] = reasoning_content
                on_message("assistant", content or "", **msg_kwargs)

            # No tool calls -> final answer
            if not tool_calls:
                return (content or reasoning_content or "")

            # Tool boundary: fire segment break (streaming only)
            if stream and on_segment_break:
                try:
                    on_segment_break()
                except Exception:
                    pass

            # Execute tool calls and append results
            tool_turn += 1
            logger.info("Tool turn %d/%s: executing %d tool call(s)",
                        tool_turn, self.max_tool_turns or "∞", len(tool_calls))

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_type = tc.get("type", "")
                func = tc.get("function", {}) or {}
                func_name = func.get("name", "")
                func_args_str = func.get("arguments", "")

                if tc_type != "function" or not func_name:
                    tool_result = json.dumps({"error": "Malformed tool call"})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})
                    if on_message:
                        on_message("tool", tool_result, tool_call_id=tc_id)
                    continue

                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except json.JSONDecodeError:
                    tool_result = json.dumps({"error": f"Invalid JSON arguments: {func_args_str}"})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})
                    if on_message:
                        on_message("tool", tool_result, tool_call_id=tc_id)
                    continue

                logger.info("  ⚡ %s(%s)", func_name, ", ".join(f"{k}={v!r}" for k, v in list(func_args.items())[:3]))
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, registry.dispatch, func_name, func_args)
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                if on_message:
                    on_message("tool", result, tool_call_id=tc_id)

        return (content or reasoning_content or "")

    async def close(self) -> None:
        await self._client.aclose()

    @classmethod
    def from_config(cls, config: Any) -> "TyAgent":
        """Create a TyAgent from an AgentConfig."""
        compression = getattr(config, "compression", None)
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tool_turns=getattr(config, "max_tool_turns", 200),
            system_prompt=config.system_prompt,
            reasoning_effort=getattr(config, "reasoning_effort", "high"),
            compression=compression,
        )


class AgentError(Exception):
    """Raised when the AI agent encounters an error."""
    pass
