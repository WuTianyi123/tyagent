"""Gateway runner for ty-agent.

Manages platform adapters, routes messages to the AI agent,
and handles session lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any, Dict, List, Optional

from ty_agent.agent import TyAgent
from ty_agent.config import TyAgentConfig
from ty_agent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from ty_agent.session import SessionStore

logger = logging.getLogger(__name__)


class Gateway:
    """Main gateway managing platform adapters and AI agent interactions."""

    def __init__(self, config: TyAgentConfig):
        self.config = config
        self.adapters: Dict[str, BasePlatformAdapter] = {}
        self.session_store = SessionStore(sessions_dir=config.sessions_dir)
        self.agent = TyAgent.from_config(config.agent)
        self._running = False
        self._shutdown_event = asyncio.Event()

    def _load_adapters(self) -> None:
        """Load and initialize platform adapters from config."""
        connected = self.config.get_connected_platforms()
        for name in connected:
            platform_cfg = self.config.get_platform(name)
            if not platform_cfg:
                continue
            try:
                if name == "feishu":
                    from ty_agent.platforms.feishu import FeishuAdapter

                    adapter = FeishuAdapter(platform_cfg)
                else:
                    logger.warning("Unsupported platform: %s", name)
                    continue

                adapter.set_message_handler(self._on_message)
                self.adapters[name] = adapter
                logger.info("Loaded adapter: %s", name)
            except Exception as exc:
                logger.error("Failed to load adapter %s: %s", name, exc)

    async def _on_message(self, event: MessageEvent) -> Optional[str]:
        """Handle an incoming message event."""
        adapter = self._find_adapter_for_event(event)
        if not adapter:
            return None

        session_key = adapter.build_session_key(event)
        session = self.session_store.get(session_key)

        # Handle reset commands
        if event.is_command() and event.get_command() in self.config.reset_triggers:
            self.session_store.reset(session_key)
            await adapter.send_message(
                event.chat_id or "",
                "Session reset. Starting fresh!",
                reply_to_message_id=event.message_id,
            )
            return "Session reset"

        # Build message for LLM
        user_message = event.text
        if event.media_urls:
            media_desc = "\n".join(
                f"[Attached {mt or 'file'}: {url}]"
                for mt, url in zip(event.media_types or [], event.media_urls)
            )
            user_message = f"{user_message}\n\n{media_desc}" if user_message else media_desc

        session.add_message("user", user_message)

        # Send typing indicator if supported
        # (Feishu doesn't have typing, but we could add reactions)

        try:
            response = await self.agent.chat(session.messages)
        except Exception as exc:
            logger.exception("Agent error")
            response = f"Error: {exc}"

        session.add_message("assistant", response)
        self.session_store.save(session_key)

        # Send response
        result = await adapter.send_message(
            event.chat_id or "",
            response,
            reply_to_message_id=event.message_id,
        )
        if not result.success:
            logger.error("Failed to send response: %s", result.error)

        return response

    def _find_adapter_for_event(self, event: MessageEvent) -> Optional[BasePlatformAdapter]:
        """Find the adapter that should handle this event."""
        # For now, simple lookup by platform hint in event
        # In practice, events come from a specific adapter
        for adapter in self.adapters.values():
            return adapter
        return None

    async def start(self) -> None:
        """Start all adapters and run the gateway."""
        self._load_adapters()
        if not self.adapters:
            logger.error("No adapters loaded. Check your configuration.")
            return

        self._running = True
        logger.info("Starting ty-agent gateway with %d adapter(s)", len(self.adapters))

        # Start all adapters
        tasks = []
        for name, adapter in self.adapters.items():
            task = asyncio.create_task(self._start_adapter(name, adapter), name=f"adapter-{name}")
            tasks.append(task)

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Stop all adapters
        logger.info("Shutting down gateway...")
        for name, adapter in self.adapters.items():
            try:
                await asyncio.wait_for(adapter.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Adapter %s stop timed out", name)

        await asyncio.gather(*tasks, return_exceptions=True)
        await self.agent.close()
        logger.info("Gateway stopped")

    async def _start_adapter(self, name: str, adapter: BasePlatformAdapter) -> None:
        """Start a single adapter with retry logic."""
        while self._running:
            try:
                await adapter.start()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Adapter %s crashed: %s", name, exc)
                if self._running:
                    await asyncio.sleep(5)

    def stop(self) -> None:
        """Signal the gateway to shut down."""
        self._running = False
        self._shutdown_event.set()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)


async def run_gateway(config_path: Optional[str] = None) -> None:
    """Entry point to start the gateway."""
    from pathlib import Path

    from ty_agent.config import load_config

    config = load_config(Path(config_path) if config_path else None)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    gateway = Gateway(config)
    gateway._setup_signal_handlers()
    await gateway.start()
