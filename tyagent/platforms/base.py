"""Base platform adapter interface for tyagent.

All platform adapters inherit from this and implement the required methods.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of incoming messages."""

    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    COMMAND = "command"


@dataclass
class MessageEvent:
    """Normalized incoming message from a platform."""

    text: str
    message_type: MessageType = MessageType.TEXT
    platform: str = ""  # Platform name (e.g., "feishu", "telegram")
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None  # "private", "group", "supergroup"
    message_id: Optional[str] = None
    raw_message: Any = None
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def is_command(self) -> bool:
        return self.text.startswith("/")

    def get_command(self) -> Optional[str]:
        if not self.is_command():
            return None
        parts = self.text.split(maxsplit=1)
        raw = parts[0][1:].lower() if parts else None
        if raw and "@" in raw:
            raw = raw.split("@", 1)[0]
        if raw and "/" in raw:
            return None
        return raw

    def get_command_args(self) -> str:
        if not self.is_command():
            return self.text
        parts = self.text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""


@dataclass
class SendResult:
    """Result of sending a message."""

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None
    retryable: bool = False
    msg_type: Optional[str] = None


# Type alias for message handlers
MessageHandler = Callable[[MessageEvent], Awaitable[Optional[str]]]


class BasePlatformAdapter(ABC):
    """Base class for platform adapters.

    Subclasses implement platform-specific logic for:
    - Connecting and authenticating
    - Receiving messages
    - Sending messages/responses
    """

    MAX_MESSAGE_LENGTH: int = 4096

    def __init__(self, config: Any, platform_name: str, *, home_dir: Optional[Path] = None):
        self.config = config
        self.platform_name = platform_name
        self.home_dir = home_dir
        self._message_handler: Optional[MessageHandler] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the callback for incoming messages."""
        self._message_handler = handler

    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """Dispatch an incoming message to the handler."""
        if self._message_handler is None:
            logger.warning("No message handler set for %s", self.platform_name)
            return None
        try:
            return await self._message_handler(event)
        except Exception:
            logger.exception("Error handling message on %s", self.platform_name)
            return None

    @abstractmethod
    async def start(self) -> None:
        """Start the platform adapter."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the platform adapter."""
        ...

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a text message to a chat."""
        ...

    async def send_photo(
        self,
        chat_id: str,
        photo_path: str,
        *,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a photo. Default implementation sends as text with path."""
        text = f"[Photo: {photo_path}]"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send_message(chat_id, text, **kwargs)

    async def send_document(
        self,
        chat_id: str,
        document_path: str,
        *,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a document. Default implementation sends as text with path."""
        text = f"[Document: {document_path}]"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send_message(chat_id, text, **kwargs)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        msg_type: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Edit an existing message. Default: not supported — subclasses override."""
        return SendResult(success=False, error="Not supported", retryable=False)

    def build_session_key(self, event: MessageEvent) -> str:
        """Build a unique session key for a message event.

        Default: platform_name:chat_id:sender_id
        """
        parts = [self.platform_name]
        if event.chat_id:
            parts.append(event.chat_id)
        if event.sender_id:
            parts.append(event.sender_id)
        return ":".join(parts)
