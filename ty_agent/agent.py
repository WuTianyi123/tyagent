"""AI Agent interface for ty-agent.

Provides a simplified adapter layer for LLM interactions.
Supports OpenAI-compatible APIs and model routing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class TyAgent:
    """Simplified AI agent for ty-agent.

    Uses OpenAI-compatible chat completions API.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_turns: int = 50,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = False,
    ) -> str:
        """Send messages to the LLM and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            stream: Whether to stream the response (not yet implemented).

        Returns:
            The assistant's response text.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Inject system prompt if not present
        prepared = list(messages)
        if not prepared or prepared[0].get("role") != "system":
            prepared.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": prepared,
            "max_tokens": 4096,
            "temperature": 0.7,
        }

        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            return content or ""
        except httpx.HTTPStatusError as exc:
            logger.error("LLM API error: %s - %s", exc.response.status_code, exc.response.text)
            return f"Error: LLM API returned {exc.response.status_code}"
        except Exception as exc:
            logger.exception("LLM request failed")
            return f"Error: {exc}"

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
