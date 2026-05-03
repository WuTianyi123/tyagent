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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Type alias for the on_message callback
OnMessageCallback = Callable[..., Any]

import functools
import httpx

from tyagent.config import CompressionConfig
from tyagent.context import compress_context
from tyagent.events import EventCollector
from tyagent.model_metadata import get_model_context_length
from tyagent.prompt_builder import build_system_prompt
from tyagent.types import AgentOutput, InboxMessage, ReplyTarget

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
USER_AGENT = "tyagent/1.0"


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
        home_dir: Optional[Path] = None,
        context_length: Optional[int] = None,
    ):
        self.model = model
        self.home_dir = home_dir
        self.context_length = context_length
        self.api_key = api_key or os.environ.get("TYAGENT_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_tool_turns = max_tool_turns
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        c = compression or CompressionConfig()
        self.compress_model = c.model
        self.compress_api_key = c.api_key or self.api_key
        self.compress_base_url = c.base_url or self.base_url
        self.compress_cut_ratio = c.cut_ratio
        self._compression_config = compression  # stored for child agent cloning
        self._effective_context_length = get_model_context_length(
            model, context_length=context_length,
        )
        self._client = httpx.AsyncClient(timeout=120.0)
        # Real token usage from the last API response
        self.last_usage: Optional[Dict[str, int]] = None
        # Cached full system prompt string (built once per session, invalidated
        # after context compression).
        self._cached_system_prompt: Optional[str] = None
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

    # ── System prompt ──────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        """Return standard HTTP headers for LLM API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _build_payload_base(self, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Return the base payload for a chat completion request."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _build_assistant_msg(
        content: Optional[str],
        reasoning_content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Build an assistant message dict for the conversation history."""
        msg: Dict[str, Any] = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _dispatch_on_message(
        self, content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]],
        reasoning_content: Optional[str],
    ) -> None:
        """Notify the on_message callback about an assistant response."""
        if self._on_message:
            kwargs: Dict[str, Any] = {}
            if tool_calls:
                kwargs["tool_calls"] = tool_calls
            if reasoning_content:
                kwargs["reasoning"] = reasoning_content
            self._on_message("assistant", content or "", **kwargs)

    def _record_token_history(self, api_messages: List[Dict[str, Any]]) -> None:
        """Record a (message_count, prompt_tokens) data point from last_usage."""
        if self.last_usage and self.last_usage.get("prompt_tokens"):
            self._token_history.append(
                (len(api_messages), self.last_usage["prompt_tokens"])
            )

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        registry: Any,
    ) -> None:
        """Execute all tool calls, appending results to *messages*."""
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            func_name = tc.get("function", {}).get("name", "")
            func_args_str = tc.get("function", {}).get("arguments", "")

            if not func_name:
                messages.append({
                    "role": "tool", "tool_call_id": tc_id,
                    "content": json.dumps({"error": "Malformed tool call"}),
                })
                continue
            try:
                func_args = json.loads(func_args_str) if func_args_str else {}
            except json.JSONDecodeError:
                messages.append({
                    "role": "tool", "tool_call_id": tc_id,
                    "content": json.dumps({"error": f"Invalid JSON: {func_args_str}"}),
                })
                continue

            if self._tool_progress_callback:
                try:
                    self._tool_progress_callback(func_name, func_args)
                except Exception:
                    pass

            entry = registry._tools.get(func_name)
            if entry and asyncio.iscoroutinefunction(entry.handler):
                result = await entry.handler(func_args, parent_agent=self)
            else:
                _loop = asyncio.get_running_loop()
                result = await _loop.run_in_executor(
                    None, registry.dispatch, func_name, func_args, self,
                )

            messages.append({
                "role": "tool", "tool_call_id": tc_id, "content": result,
            })
            if self._on_message:
                self._on_message("tool", result, tool_call_id=tc_id)

    async def _call_api_nonstreaming(
        self, payload: Dict[str, Any], headers: Dict[str, str],
    ) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Make a non-streaming API call. Raises ContextOverflow on 400/413."""
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload,
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
        return (message.get("content"), message.get("reasoning_content"), message.get("tool_calls"))

    async def _call_api_streaming(
        self, payload: Dict[str, Any], headers: Dict[str, str], *,
        stream_delta_callback: Optional[Callable[[str], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Make a streaming API call. Raises ContextOverflow on 400/413."""
        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            if resp.status_code >= 400:
                error_body = b""
                async for chunk in resp.aiter_raw(chunk_size=4096):
                    error_body += chunk
                body_str = error_body.decode("utf-8", errors="replace")[:2000]
                if _is_context_overflow(resp.status_code, body_str):
                    raise ContextOverflow(body_str)
                raise AgentError(f"LLM API returned {resp.status_code}: {body_str}")

            content_parts: List[str] = []
            tool_calls_acc: Dict[int, Dict[str, Any]] = {}
            reasoning_parts: List[str] = []
            usage_obj = None

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
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
            tc_list = list(tool_calls_acc.values()) if tool_calls_acc else None

            if usage_obj:
                self.last_usage = {
                    "prompt_tokens": usage_obj.get("prompt_tokens", 0)
                    if isinstance(usage_obj, dict)
                    else getattr(usage_obj, "prompt_tokens", 0),
                    "completion_tokens": usage_obj.get("completion_tokens", 0)
                    if isinstance(usage_obj, dict)
                    else getattr(usage_obj, "completion_tokens", 0),
                    "total_tokens": usage_obj.get("total_tokens", 0)
                    if isinstance(usage_obj, dict)
                    else getattr(usage_obj, "total_tokens", 0),
                }

            return content, reasoning, tc_list

    async def _api_call_with_compression_retry(
        self,
        messages: List[Dict[str, Any]],
        api_messages: List[Dict[str, Any]],
        payload: Dict[str, Any],
        headers: Dict[str, str],
        *,
        api_caller: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Call the LLM API with single-pass compression retry.

        *api_caller* is an async callable ``(payload, headers) -> (content,
        reasoning, tool_calls)``.  On ContextOverflow, compresses once
        and retries.  *api_messages* is mutated in place and *payload*
        updated on compression.
        """
        _compressed = False
        while True:
            self.last_usage = None
            try:
                return await api_caller(payload, headers)
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
                        context_window=self._effective_context_length,
                        cut_ratio=self.compress_cut_ratio,
                    )
                    if compressed is not None:
                        api_messages[:] = compressed
                        self._prev_msg_count = len(messages)
                        self._token_history = []
                        self._cached_system_prompt = None
                        payload["messages"] = api_messages
                        continue
                raise AgentError("Context too long even after compression.")
            except AgentError:
                raise
            except Exception as exc:
                raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc

    def _ensure_system_prompt(self) -> str:
        """Return the system prompt, building and caching it on first call."""
        if self._cached_system_prompt is None:
            self._cached_system_prompt = build_system_prompt(
                model=self.model,
                user_prompt=self.system_prompt,
                home_dir=self.home_dir,
            )
        return self._cached_system_prompt

    def _insert_system_prompt(self, messages: List[Dict[str, Any]]) -> None:
        """Insert the system message at messages[0] if not already present."""
        prompt = self._ensure_system_prompt()
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": prompt})

    # ── Chat (backward-compat) ─────────────────────────────────

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
            from tyagent.tools.registry import registry

            headers = self._build_headers()
            self._insert_system_prompt(messages)
            self._prev_msg_count = 0
            self._token_history = []
            api_messages = list(messages)
            self._prev_msg_count = len(messages)
            payload_base = self._build_payload_base(tools)

            tool_turn = 0
            content = None
            reasoning_content = None
            tool_calls = None
            while True:
                if self.max_tool_turns is not None and self.max_tool_turns > 0 and tool_turn >= self.max_tool_turns:
                    break
                if tool_turn > 0:
                    api_messages.extend(messages[self._prev_msg_count:])
                self._prev_msg_count = len(messages)
                payload = {**payload_base, "messages": api_messages,
                           "stream": True, "stream_options": {"include_usage": True}}

                _stream_caller = functools.partial(
                    self._call_api_streaming,
                    stream_delta_callback=stream_delta_callback,
                    reasoning_callback=reasoning_callback,
                )
                content, reasoning_content, tool_calls = \
                    await self._api_call_with_compression_retry(
                        messages, api_messages, payload, headers,
                        api_caller=_stream_caller,
                    )

                self._record_token_history(api_messages)

                assistant_msg = self._build_assistant_msg(content, reasoning_content, tool_calls)
                messages.append(assistant_msg)
                self._dispatch_on_message(content, tool_calls, reasoning_content)

                if not tool_calls:
                    return (content or reasoning_content or "")

                if stream and on_segment_break:
                    try:
                        on_segment_break()
                    except Exception:
                        pass

                tool_turn += 1
                await self._execute_tool_calls(tool_calls, messages, registry)
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
        self._insert_system_prompt(messages)

        self._prev_msg_count = 0
        self._token_history = []
        api_messages = list(messages)
        self._prev_msg_count = len(messages)

        payload_base = self._build_payload_base(tools)

        headers = self._build_headers()

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

            content, reasoning_content, tool_calls = \
                await self._api_call_with_compression_retry(
                    messages, api_messages, payload_base, headers,
                    api_caller=self._call_api_nonstreaming,
                )

            self._record_token_history(api_messages)

            assistant_msg = self._build_assistant_msg(content, reasoning_content, tool_calls)
            messages.append(assistant_msg)
            self._dispatch_on_message(content, tool_calls, reasoning_content)

            if not tool_calls:
                return (content or reasoning_content or "")

            tool_turn += 1
            await self._execute_tool_calls(tool_calls, messages, registry)

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

    async def send_message(
        self, text: str,
        reply_target: Optional[ReplyTarget] = None,
        tool_progress_cb: Any = None,
        turn_done_cb: Any = None,
    ) -> None:
        """Send a message to the agent loop (fire-and-forget)."""
        if not self._running:
            raise RuntimeError("Agent loop not running. Call start() first.")
        if not text:
            return
        await self._inbox.put(InboxMessage(
            text=text, reply_target=reply_target,
            tool_progress_cb=tool_progress_cb,
            turn_done_cb=turn_done_cb,
        ))

    async def _agent_loop(self) -> None:
        """Permanent agent event loop — select! over inbox, collector, stop."""
        loop = asyncio.get_running_loop()
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
                # Handle the message consumed by inbox_task (if both
                # stop_task and inbox_task completed in the same tick).
                if inbox_task in done:
                    msg = inbox_task.result()
                    if msg.turn_done_cb:
                        try:
                            msg.turn_done_cb()
                        except Exception:
                            pass
                # Drain any remaining inbox messages — their turn_done
                # callbacks must be fired so ProgressSenders are finished.
                while not self._inbox.empty():
                    try:
                        msg = self._inbox.get_nowait()
                        if msg.turn_done_cb:
                            try:
                                msg.turn_done_cb()
                            except Exception:
                                pass
                    except asyncio.QueueEmpty:
                        break
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
            current_reply = None
            current_tool_cb = None
            current_turn_done = None
            if inbox_task in done:
                msg = inbox_task.result()
                self._messages.append({"role": "user", "content": msg.text})
                current_reply = msg.reply_target
                current_tool_cb = msg.tool_progress_cb
                current_turn_done = msg.turn_done_cb

            # Run turn with error handling
            tools = registry.get_definitions()
            try:
                # Activate per-message tool progress callback
                prev_tool_cb = self._tool_progress_callback
                self._tool_progress_callback = current_tool_cb
                content = await self._run_turn(tools=tools)
                self._tool_progress_callback = prev_tool_cb
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
                self._tool_progress_callback = prev_tool_cb
                await self._output_queue.put(AgentOutput(
                    text=f"❌ 错误: {exc}",
                    reply_target=current_reply,
                ))
            except Exception as exc:
                logger.exception("Unexpected error in agent loop")
                self._tool_progress_callback = prev_tool_cb
                await self._output_queue.put(AgentOutput(
                    text=f"❌ 内部错误: {exc}",
                    reply_target=current_reply,
                ))
            except asyncio.CancelledError:
                # CancelledError is BaseException in 3.11+, not caught above.
                # Restore callback before letting cancellation propagate.
                self._tool_progress_callback = prev_tool_cb
                raise
            finally:
                # Signal that this message's turn is done
                if current_turn_done:
                    try:
                        current_turn_done()
                    except Exception:
                        pass

    async def close(self) -> None:
        """Close agent and clean up."""
        if self._running:
            await self.stop()
        self._bg_tasks.clear()
        self._event_collector = None
        await self._client.aclose()

    @classmethod
    def from_config(cls, config: Any, *, home_dir: Optional[Path] = None) -> "TyAgent":
        """Create a TyAgent from an AgentConfig."""
        compression = getattr(config, "compression", None)
        return cls(
            model=config.model,
            api_key=getattr(config, "api_key", None),
            base_url=config.base_url,
            max_tool_turns=getattr(config, "max_tool_turns", 200),
            system_prompt=config.system_prompt,
            reasoning_effort=getattr(config, "reasoning_effort", "high"),
            compression=compression,
            home_dir=home_dir,
            context_length=getattr(config, "context_length", None),
        )


class AgentError(Exception):
    """Raised when the AI agent encounters an error."""
    pass
