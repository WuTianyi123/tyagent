"""AI Agent interface for tyagent.

Provides a simplified adapter layer for LLM interactions.
Supports OpenAI-compatible APIs, model routing, and function calling (tools).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Type alias for the on_message callback
OnMessageCallback = Callable[..., Any]

import httpx

from tyagent.compaction import run_compact, total_token_estimate
from tyagent.config import CompressionConfig
from tyagent.events import EventCollector
from tyagent.model_metadata import get_model_context_length
from tyagent.prompt_builder import build_system_prompt
from tyagent.types import AgentOutput, InboxMessage, ReplyTarget

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
USER_AGENT = "tyagent/1.0"


# No ContextOverflow exception — compaction is PROACTIVE (pre-turn + mid-turn
# threshold checks), not reactive (catch 400 then retry).  Architected to
# match Codex CLI's compact.rs design philosophy.


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
        max_tokens: int = 4096,
        temperature: float = 0.7,
        http_timeout: float = 120.0,
        shutdown_timeout: float = 5.0,
    ):
        self.model = model
        self.home_dir = home_dir
        self.context_length = context_length
        self.api_key = api_key or os.environ.get("TYAGENT_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_tool_turns = max_tool_turns
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http_timeout = http_timeout
        self._shutdown_timeout = shutdown_timeout
        c = compression or CompressionConfig()
        self._compression_config = compression  # stored for child agent cloning
        self._effective_context_length = get_model_context_length(
            model, context_length=context_length,
        )
        # Proactive compaction threshold.  When total estimated tokens
        # exceed this, compaction runs before the next API call.  Maps to
        # Codex CLI's ``model_auto_compact_token_limit`` config.
        auto_limit = c.auto_compact_limit
        try:
            if auto_limit is None or auto_limit <= 0:
                auto_limit = int(self._effective_context_length * 0.90)
        except TypeError:
            # Non-integer value (e.g. MagicMock in tests)
            auto_limit = int(self._effective_context_length * 0.90)
        self.auto_compact_limit: int = auto_limit
        # Which model/client to use for the compaction stand-alone turn.
        # Falls back to the main model when not configured separately.
        self.compact_model: str = c.model or self.model
        self.compact_api_key: str = c.api_key or self.api_key
        self.compact_base_url: str = c.base_url or self.base_url
        self._client = httpx.AsyncClient(timeout=self._http_timeout)
        # Real token usage from the last API response
        self.last_usage: Optional[Dict[str, int]] = None
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
        # System prompt built once at init, then cached for prefix-cache
        # stability across turns.  Rebuilt after compaction so mid-session
        # memory writes are picked up — the cache-miss cost is negligible
        # since compaction already changes the messages prefix.
        self._system_prompt: str = build_system_prompt(
            model=self.model,
            user_prompt=self.system_prompt,
            home_dir=self.home_dir,
        )

    # ── System prompt refresh ──────────────────────────────────

    def _refresh_memory_and_prompt(self) -> None:
        """Rebuild MemoryStore snapshot and regenerate system prompt.

        Called after compaction so mid-session memory writes are picked
        up.  Safe to call multiple times — regenerates system prompt only
        when the MemoryStore is available (gateway profile), no-op in
        isolated contexts (tests, child agents).
        """
        try:
            from tyagent.tools.memory_tool import get_store
            store = get_store()
            if store is None:
                return
            store._rebuild_snapshot()
            self._system_prompt = build_system_prompt(
                model=self.model,
                user_prompt=self.system_prompt,
                home_dir=self.home_dir,
            )
        except Exception:
            pass

    # ── API helpers ─────────────────────────────────────────────

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
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
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
        """Make a non-streaming API call."""
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload,
            )
        except Exception as exc:
            raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            body_str = resp.text[:2000]
            raise AgentError(f"LLM API returned {resp.status_code}: {body_str}")
        try:
            data = resp.json()
        except Exception as exc:
            raise AgentError(f"Invalid JSON response: {exc}") from exc
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
        """Make a streaming API call."""
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=payload, headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    error_body = b""
                    async for chunk in resp.aiter_raw(chunk_size=4096):
                        error_body += chunk
                    body_str = error_body.decode("utf-8", errors="replace")[:2000]
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
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc

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
        """Backward-compatible one-shot chat. Delegates to :meth:`_run_turn`.

        Sets instance callbacks (*_on_message*, *_tool_progress_callback*,
        *_messages*) so that :meth:`_run_turn` finds them — safe because
        sub-agents use a fresh ``TyAgent`` instance per task.
        """
        self._tool_progress_callback = tool_progress_callback
        self._on_message = on_message
        # Strip any system messages from incoming list — the new
        # architecture builds the system prompt fresh per turn.
        self._messages = messages
        # Remove any existing system messages in-place (mutation preserves
        # caller's reference for backward-compatible message inspection).
        while self._messages and self._messages[0].get("role") == "system":
            self._messages.pop(0)
        return await self._run_turn(
            tools=tools,
            stream=stream,
            stream_delta_callback=stream_delta_callback,
            reasoning_callback=reasoning_callback,
            on_segment_break=on_segment_break if stream else None,
        )

    async def _run_turn(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        stream_delta_callback: Optional[Callable[[str], None]] = None,
        reasoning_callback: Optional[Callable[[str], None]] = None,
        on_segment_break: Optional[Callable[[], None]] = None,
    ) -> str:
        """Run one LLM+tool-call turn to completion. Mutates self._messages in place.

        Architecture (Codex CLI style):
          - ``self._messages`` NEVER contains a system message
          - System prompt is built fresh each iteration
          - Pre-turn compaction: before first API call, check total tokens
          - Mid-turn compaction: after tool calls, if still over limit, compact
          - No reactive overflow handling — proactive estimates are authoritative

        When *stream* is True, uses SSE streaming and invokes
        *stream_delta_callback*, *reasoning_callback*, and
        *on_segment_break* at the appropriate points.
        """
        from tyagent.tools.registry import registry

        messages = self._messages

        # System prompt is cached from init for prefix-cache stability.
        # Stored separately from messages; injected at API call time.
        system_prompt = self._system_prompt

        # ── Pre-turn compaction ─────────────────────────────────
        # Use exact prompt_tokens from the last API response when available,
        # falling back to byte-based estimate only for the first-ever call.
        if self.last_usage is not None:
            est = self.last_usage["prompt_tokens"]
        else:
            est = total_token_estimate(messages, system_prompt=system_prompt)
        if est >= self.auto_compact_limit:
            logger.info(
                "Pre-turn compaction: %d prompt_tokens >= %d limit",
                est,
                self.auto_compact_limit,
            )
            compacted = await run_compact(
                messages,
                model=self.compact_model,
                api_key=self.compact_api_key,
                base_url=self.compact_base_url,
                http_client=self._client,
            )
            if compacted is not None:
                messages[:] = compacted
                self._refresh_memory_and_prompt()

        payload_base = self._build_payload_base(tools)
        headers = self._build_headers()

        tool_turn = 0
        content: Optional[str] = None
        reasoning_content: Optional[str] = None
        tool_calls = None

        while True:
            if self.max_tool_turns is not None and self.max_tool_turns > 0 and tool_turn >= self.max_tool_turns:
                break

            # Build api_messages: system prompt prepended at call time only.
            # messages itself NEVER contains system — compaction never touches it.
            api_messages = [{"role": "system", "content": system_prompt}] + messages

            if stream:
                payload: Dict[str, Any] = {
                    **payload_base, "messages": api_messages,
                    "stream": True, "stream_options": {"include_usage": True},
                }
                content, reasoning_content, tool_calls = await self._call_api_streaming(
                    payload, headers,
                    stream_delta_callback=stream_delta_callback,
                    reasoning_callback=reasoning_callback,
                )
            else:
                payload = {**payload_base, "messages": api_messages}
                content, reasoning_content, tool_calls = await self._call_api_nonstreaming(
                    payload, headers,
                )

            assistant_msg = self._build_assistant_msg(content, reasoning_content, tool_calls)
            messages.append(assistant_msg)
            self._dispatch_on_message(content, tool_calls, reasoning_content)

            if not tool_calls:
                return (content or reasoning_content or "")

            if on_segment_break:
                try:
                    on_segment_break()
                except Exception:
                    pass

            # ── Mid-turn compaction (Codex style, pre-exec) ──────
            # Before executing tool calls, check if we're over the limit.
            # If yes, compact and continue so the model re-decides in a
            # freed-up context.  Equivalent to turn.rs L485-501.
            if self.last_usage is not None:
                est = self.last_usage["prompt_tokens"]
            else:
                est = total_token_estimate(messages, system_prompt=system_prompt)
            if est >= self.auto_compact_limit:
                logger.info(
                    "Mid-turn compaction (pre-exec): %d prompt_tokens >= %d limit",
                    est,
                    self.auto_compact_limit,
                )
                compacted = await run_compact(
                    messages,
                    model=self.compact_model,
                    api_key=self.compact_api_key,
                    base_url=self.compact_base_url,
                    http_client=self._client,
                )
                if compacted is not None:
                    messages[:] = compacted
                    self._refresh_memory_and_prompt()
                # Continue — model sees compacted context and re-decides
                continue

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
                await asyncio.wait_for(self._loop_task, timeout=self._shutdown_timeout)
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
            stop_task = loop.create_task(self._stop_event.wait())

            tasks = [inbox_task, stop_task]
            child_task = None
            if self._event_collector:
                child_task = loop.create_task(self._event_collector.wait_next())
                tasks.append(child_task)

            done, pending = await asyncio.wait(
                tasks,
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
            prev_tool_cb = self._tool_progress_callback
            self._tool_progress_callback = current_tool_cb
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
            finally:
                self._tool_progress_callback = prev_tool_cb
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
            max_tokens=getattr(config, "max_tokens", 4096),
            temperature=getattr(config, "temperature", 0.7),
            http_timeout=getattr(config, "http_timeout", 120.0),
            shutdown_timeout=getattr(config, "shutdown_timeout", 5.0),
        )

    def clone(self) -> "TyAgent":
        """Create an independent copy with identical configuration.

        The new agent gets its own HTTP client, message history,
        and event loop state — safe for concurrent use in
        different sessions.
        """
        return TyAgent(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tool_turns=self.max_tool_turns,
            system_prompt=self.system_prompt,
            reasoning_effort=self.reasoning_effort,
            compression=self._compression_config,
            home_dir=self.home_dir,
            context_length=self.context_length,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            http_timeout=self._http_timeout,
            shutdown_timeout=self._shutdown_timeout,
        )

class AgentError(Exception):
    """Raised when the AI agent encounters an error."""
    pass
