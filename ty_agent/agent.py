"""AI Agent interface for ty-agent.

Provides a simplified adapter layer for LLM interactions.
Supports OpenAI-compatible APIs, model routing, and function calling (tools).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class TyAgent:
    """Simplified AI agent for ty-agent.

    Uses OpenAI-compatible chat completions API with optional tool calling.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_turns: int = 50,
        max_tool_turns: int = 30,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_turns = max_turns
        self.max_tool_turns = max_tool_turns
        self.system_prompt = system_prompt
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> str:
        """Send messages to the LLM and return the response text.

        If *tools* are provided, the agent runs a tool-calling loop:
        it repeatedly sends the conversation to the LLM, executes any
        tool_calls returned, appends the results, and continues until
        the model produces a final text response (no more tool_calls)
        or ``max_tool_turns`` is reached.

        The *messages* list is mutated in-place so that tool calls and
        results are preserved in the caller's session history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Mutated in-place with assistant/tool messages.
            tools: Optional list of OpenAI-format tool definitions.
            stream: Whether to stream the response (not yet implemented).

        Returns:
            The assistant's final response text.
        """
        # Lazy import to avoid circular dependency
        from ty_agent.tools.registry import registry

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "KimiCLI/1.30.0",
        }

        # Inject system prompt if not present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload_base = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        if tools:
            payload_base["tools"] = tools
            payload_base["tool_choice"] = "auto"

        tool_turn = 0
        content: Optional[str] = None
        while True:
            if tool_turn >= self.max_tool_turns:
                logger.warning("Max tool turns (%d) reached, returning last content", self.max_tool_turns)
                break

            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload_base,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.error("LLM API error: %s - %s", exc.response.status_code, exc.response.text)
                raise AgentError(f"LLM API returned status {exc.response.status_code}") from exc
            except Exception as exc:
                logger.exception("LLM request failed")
                raise AgentError(f"LLM request failed: {type(exc).__name__}") from exc

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            reasoning_content = message.get("reasoning_content")

            # Append the assistant message (with or without tool_calls)
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content is not None:
                assistant_msg["content"] = content
            if reasoning_content is not None:
                assistant_msg["reasoning_content"] = reasoning_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # No tool calls -> we have the final answer
            if not tool_calls:
                return content or ""

            # Execute tool calls and append results
            tool_turn += 1
            logger.info("Tool turn %d/%d: executing %d tool call(s)",
                        tool_turn, self.max_tool_turns, len(tool_calls))

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_type = tc.get("type", "")
                func = tc.get("function", {})
                func_name = func.get("name", "")
                func_args_str = func.get("arguments", "")

                if tc_type != "function" or not func_name:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps({"error": "Malformed tool call"}),
                    })
                    continue

                # Parse arguments
                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps({"error": f"Invalid JSON arguments: {func_args_str}"}),
                    })
                    continue

                # Dispatch — run in executor to avoid blocking the event loop
                logger.info("  ⚡ %s(%s)", func_name, ", ".join(f"{k}={v!r}" for k, v in list(func_args.items())[:3]))
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, registry.dispatch, func_name, func_args
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })

        # If we broke out of the loop (max turns), return the last content we saw
        return content or ""

    async def close(self) -> None:
        await self._client.aclose()

    @classmethod
    def from_config(cls, config: Any) -> "TyAgent":
        """Create a TyAgent from an AgentConfig."""
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_turns=config.max_turns,
            system_prompt=config.system_prompt,
        )


class AgentError(Exception):
    """Raised when the AI agent encounters an error."""
    pass
