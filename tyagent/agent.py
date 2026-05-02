"""AI Agent interface for tyagent.

Provides a simplified adapter layer for LLM interactions.
Supports OpenAI-compatible APIs, model routing, and function calling (tools).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Type alias for the on_message callback
OnMessageCallback = Callable[..., Any]

import httpx

from tyagent.config import CompressionConfig
from tyagent.context import compress_context
from tyagent.events import EventCollector
from tyagent.types import AgentOutput, InboxMessage, ReplyTarget

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
    if status_code not in (400, 413):
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
        self._compression_config = compression  # stored for child agent cloning
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
        # ── Actor model lifecycle ─────────────────────────────────
        self._inbox: asyncio.Queue["InboxMessage"] = asyncio.Queue()
        self._output_queue: asyncio.Queue[AgentOutput] = asyncio.Queue()
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._messages: List[Dict[str, Any]] = []
        self._on_message: Optional[OnMessageCallback] = None
        self._tool_progress_callback: Optional[Callable] = None
        # ── Child agent management ────────────────────────────────
        self._bg_tasks: Dict[str, asyncio.Task] = {}
        self._event_collector: Optional[EventCollector] = None

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
        tool_progress_callback: Optional[Callable[..., Any]] = None,
    ) -> str:
        """Backward-compatible one-shot chat. Delegates to _run_turn() for non-streaming.

        For streaming calls, runs the original streaming logic inline.
        Used by sub-agents and existing tests.
        """
        self._tool_progress_callback = tool_progress_callback
        self._on_message = on_message

        if stream:
            # Keep original streaming path for backward compat
            # (gateway/consumer.py uses this)
            import copy
            from tyagent.tools.registry import registry

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "KimiCLI/1.30.0",
            }
            if self._system_msg is None:
                self._system_msg = {"role": "system", "content": self.system_prompt}
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, self._system_msg)
            self._prev_msg_count = 0
            self._token_history = []
            api_messages = list(messages)
            self._prev_msg_count = len(messages)
            payload_base = {
                "model": self.model, "max_tokens": 4096, "temperature": 0.7,
            }
            if self.reasoning_effort:
                payload_base["reasoning_effort"] = self.reasoning_effort
            if tools:
                payload_base["tools"] = tools
                payload_base["tool_choice"] = "auto"

            tool_turn = 0
            content = None
            reasoning_content = None
            tool_calls = None
            while True:
                if self.max_tool_turns and tool_turn >= self.max_tool_turns:
                    break
                if tool_turn > 0:
                    api_messages.extend(messages[self._prev_msg_count:])
                self._prev_msg_count = len(messages)
                payload = {**payload_base, "messages": api_messages,
                           "stream": True, "stream_options": {"include_usage": True}}

                _compressed = False
                while True:
                    self.last_usage = None
                    try:
                        async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                                       json=payload, headers=headers) as resp:
                            if resp.status_code >= 400:
                                error_body = b""
                                async for chunk in resp.aiter_raw(chunk_size=4096):
                                    error_body += chunk
                                body_str = error_body.decode("utf-8", errors="replace")[:2000]
                                if _is_context_overflow(resp.status_code, body_str):
                                    raise ContextOverflow(body_str)
                                raise AgentError(f"LLM API returned {resp.status_code}: {body_str}")
                            content_parts = []
                            tool_calls_acc = {}
                            reasoning_parts = []
                            usage_obj = None
                            async for line in resp.aiter_lines():
                                if not line.startswith("data: "): continue
                                data_str = line[6:].strip()
                                if data_str == "[DONE]": break
                                if not data_str: continue
                                chunk = json.loads(data_str)
                                if not chunk.get("choices"):
                                    if chunk.get("usage"): usage_obj = chunk["usage"]
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
                                        if tc_delta.get("id"): acc["id"] = tc_delta["id"]
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
                        break
                    except ContextOverflow:
                        if not _compressed:
                            _compressed = True
                            from tyagent.context import compress_context
                            compressed = await compress_context(
                                api_messages, self._client,
                                model=self.compress_model or self.model,
                                api_key=self.compress_api_key, base_url=self.compress_base_url,
                                token_history=self._token_history,
                                context_window=self.compress_context_window,
                                cut_ratio=self.compress_cut_ratio,
                            )
                            if compressed is not None:
                                api_messages = compressed
                                self._prev_msg_count = len(messages)
                                self._token_history = []
                                payload["messages"] = api_messages
                                continue
                        raise AgentError("Context too long even after compression.")
                    except AgentError: raise
                    except Exception as exc:
                        raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc

                if self.last_usage and self.last_usage.get("prompt_tokens"):
                    self._token_history.append((len(api_messages), self.last_usage["prompt_tokens"]))

                assistant_msg = {"role": "assistant"}
                if content is not None: assistant_msg["content"] = content
                if reasoning_content is not None: assistant_msg["reasoning_content"] = reasoning_content
                if tool_calls: assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
                if on_message:
                    msg_kwargs = {}
                    if tool_calls: msg_kwargs["tool_calls"] = tool_calls
                    if reasoning_content: msg_kwargs["reasoning"] = reasoning_content
                    on_message("assistant", content or "", **msg_kwargs)
                if not tool_calls:
                    return (content or reasoning_content or "")
                if stream and on_segment_break:
                    try: on_segment_break()
                    except Exception: pass
                tool_turn += 1
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    func_name = tc.get("function", {}).get("name", "")
                    func_args_str = tc.get("function", {}).get("arguments", "")
                    if not func_name:
                        messages.append({"role": "tool", "tool_call_id": tc_id,
                                         "content": json.dumps({"error": "Malformed"})})
                        continue
                    try:
                        func_args = json.loads(func_args_str) if func_args_str else {}
                    except json.JSONDecodeError:
                        messages.append({"role": "tool", "tool_call_id": tc_id,
                                         "content": json.dumps({"error": f"Invalid JSON: {func_args_str}"})})
                        continue
                    if tool_progress_callback:
                        try: tool_progress_callback(func_name, func_args)
                        except Exception: pass
                    # Check if handler is async (sub-agent tools)
                    entry = registry._tools.get(func_name)
                    if entry and asyncio.iscoroutinefunction(entry.handler):
                        result = await entry.handler(func_args, parent_agent=self)
                    else:
                        _loop = asyncio.get_running_loop()
                        result = await _loop.run_in_executor(
                            None, registry.dispatch, func_name, func_args, self,
                        )
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                    if on_message:
                        on_message("tool", result, tool_call_id=tc_id)
        else:
            # Non-streaming: delegate to _run_turn()
            self._messages = messages
            return await self._run_turn(tools=tools)

    async def _run_turn(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Run one LLM+tool-call turn to completion. Mutates self._messages in place."""
        from tyagent.tools.registry import registry

        messages = self._messages

        # System prompt injection
        if self._system_msg is None:
            self._system_msg = {"role": "system", "content": self.system_prompt}
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, self._system_msg)

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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "KimiCLI/1.30.0",
        }

        tool_turn = 0
        content: Optional[str] = None
        reasoning_content: Optional[str] = None
        tool_calls = None

        while True:
            if self.max_tool_turns is not None and self.max_tool_turns > 0 and tool_turn >= self.max_tool_turns:
                break

            if tool_turn > 0:
                api_messages.extend(messages[self._prev_msg_count:])
            self._prev_msg_count = len(messages)
            payload_base["messages"] = api_messages

            _compressed = False
            while True:
                self.last_usage = None
                try:
                    # Non-streaming path only
                    resp = await self._client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload_base,
                    )
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
                    break

                except ContextOverflow:
                    if not _compressed:
                        _compressed = True
                        logger.info("Context overflow — applying single-pass compression")
                        from tyagent.context import compress_context
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
                    raise AgentError("Context too long even after compression.")
                except AgentError:
                    raise
                except Exception as exc:
                    raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc

            if self.last_usage and self.last_usage.get("prompt_tokens"):
                self._token_history.append((len(api_messages), self.last_usage["prompt_tokens"]))

            # Build assistant message
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content is not None:
                assistant_msg["content"] = content
            if reasoning_content is not None:
                assistant_msg["reasoning_content"] = reasoning_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Persist via callback
            if self._on_message:
                msg_kwargs: Dict = {}
                if tool_calls:
                    msg_kwargs["tool_calls"] = tool_calls
                if reasoning_content:
                    msg_kwargs["reasoning"] = reasoning_content
                self._on_message("assistant", content or "", **msg_kwargs)

            if not tool_calls:
                return (content or reasoning_content or "")

            # Execute tool calls
            tool_turn += 1
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func_name = tc.get("function", {}).get("name", "")
                func_args_str = tc.get("function", {}).get("arguments", "")

                if not func_name:
                    messages.append({"role": "tool", "tool_call_id": tc_id,
                                     "content": json.dumps({"error": "Malformed tool call"})})
                    continue
                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc_id,
                                     "content": json.dumps({"error": f"Invalid JSON: {func_args_str}"})})
                    continue

                if self._tool_progress_callback:
                    try: self._tool_progress_callback(func_name, func_args)
                    except Exception: pass

                entry = registry._tools.get(func_name)
                if entry and asyncio.iscoroutinefunction(entry.handler):
                    result = await entry.handler(func_args, parent_agent=self)
                else:
                    _loop = asyncio.get_running_loop()
                    result = await _loop.run_in_executor(
                        None, registry.dispatch, func_name, func_args, self,
                    )
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                if self._on_message:
                    self._on_message("tool", result, tool_call_id=tc_id)

        return (content or reasoning_content or "")

    async def start(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        on_message: Optional[OnMessageCallback] = None,
    ) -> None:
        """Start the permanent agent event loop."""
        if self._running:
            return
        self._messages = list(history) if history else []
        self._on_message = on_message
        self._running = True
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._agent_loop())

    async def stop(self) -> None:
        """Stop the agent loop and clean up."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        # Cancel running children
        for tid, task in list(self._bg_tasks.items()):
            if not task.done():
                task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks.values(), return_exceptions=True)
        self._bg_tasks.clear()
        # Wait for loop to finish
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._loop_task.cancel()
                try: await self._loop_task
                except asyncio.CancelledError: pass
            self._loop_task = None

    async def send_message(self, text: str, reply_target: Optional[ReplyTarget] = None) -> None:
        """Send a message to the agent loop (fire-and-forget)."""
        if not self._running:
            raise RuntimeError("Agent loop not running. Call start() first.")
        if not text:
            return
        await self._inbox.put(InboxMessage(text=text, reply_target=reply_target))

    async def _agent_loop(self) -> None:
        """Permanent agent event loop — select! over inbox, collector, stop."""
        loop = asyncio.get_event_loop()
        from tyagent.tools.registry import registry

        while self._running:
            inbox_task = loop.create_task(self._inbox.get())
            child_task = loop.create_task(
                self._event_collector.wait_next() if self._event_collector
                else asyncio.sleep(999999)  # never completes if no collector
            )
            stop_task = loop.create_task(self._stop_event.wait())

            done, pending = await asyncio.wait(
                [inbox_task, child_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            if stop_task in done:
                break

            # Inject child completions (auto-replies have no reply_target)
            current_reply = None
            if self._event_collector is not None:
                child_events = self._event_collector.drain_completed()
                for event in child_events:
                    summary = event["result"].get("summary", "")
                    if summary:
                        inject = f"## Subagent `{event['task_id']}` completed\n\n{summary}"
                    elif event["result"].get("success"):
                        inject = f"## Subagent `{event['task_id']}` completed successfully"
                    else:
                        inject = f"## Subagent `{event['task_id']}` failed\n\n{event['result'].get('error', 'Unknown')}"
                    self._messages.append({"role": "user", "content": inject})

            # Process user message
            if inbox_task in done:
                msg = inbox_task.result()
                self._messages.append({"role": "user", "content": msg.text})
                current_reply = msg.reply_target

            # Run turn with error handling
            tools = registry.get_definitions()
            try:
                content = await self._run_turn(tools=tools)
                if content.strip():
                    await self._output_queue.put(AgentOutput(
                        text=content,
                        reply_target=current_reply,
                    ))
                elif current_reply is not None:
                    # Non-empty response required but got empty — put placeholder
                    await self._output_queue.put(AgentOutput(
                        text="(no response)",
                        reply_target=current_reply,
                    ))
            except AgentError as exc:
                logger.error("Agent loop error: %s", exc)
                await self._output_queue.put(AgentOutput(
                    text=f"❌ 错误: {exc}",
                    reply_target=current_reply,
                ))
            except Exception as exc:
                logger.exception("Unexpected error in agent loop")
                await self._output_queue.put(AgentOutput(
                    text=f"❌ 内部错误: {exc}",
                    reply_target=current_reply,
                ))

    async def close(self) -> None:
        """Close agent and clean up."""
        if self._running:
            await self.stop()
        self._bg_tasks.clear()
        self._event_collector = None
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
