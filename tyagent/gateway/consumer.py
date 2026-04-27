"""Stream consumer — bridges sync stream delta from agent thread to async platform message edit.

Usage:
    consumer = StreamConsumer(adapter, chat_id)
    consumer_task = asyncio.create_task(consumer.run())
    await agent.chat(..., stream=True,
                    stream_delta_callback=consumer.on_delta,
                    on_segment_break=consumer.on_segment_break)
    consumer.finish()
    await consumer_task
    final_content = consumer.final_content
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Optional

from tyagent.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

# Sentinels for StreamConsumer
_DONE = object()
_NEW_SEGMENT = object()


class StreamConsumer:
    """Bridges sync stream_delta from agent thread to async platform message edit.

    Usage:
        consumer = StreamConsumer(adapter, chat_id)
        consumer_task = asyncio.create_task(consumer.run())
        await agent.chat(..., stream=True,
                        stream_delta_callback=consumer.on_delta,
                        on_segment_break=consumer.on_segment_break)
        consumer.finish()
        await consumer_task
        final_content = consumer.final_content
    """

    def __init__(self, adapter: BasePlatformAdapter, chat_id: str, *, reply_to_message_id: Optional[str] = None):
        self.adapter = adapter
        self.chat_id = chat_id
        self._queue: "queue.Queue" = queue.Queue()
        self._message_id: Optional[str] = None
        self._msg_type: Optional[str] = None
        self._accumulated: str = ""
        self._last_edit_time: float = 0.0
        self._edit_interval: float = 1.0
        self._buffer_threshold: int = 30
        self._last_sent_text: str = ""
        self._edit_supported: bool = True
        self._flood_strikes: int = 0
        self._MAX_FLOOD_STRIKES: int = 3
        self._current_edit_interval: float = 1.0
        self.final_content: str = ""
        self._already_sent: bool = False
        self._reply_to_message_id = reply_to_message_id

    def on_delta(self, text: str) -> None:
        """Called from agent's thread (sync) for each text delta."""
        self._queue.put(text)

    def on_segment_break(self) -> None:
        """Called on tool boundary — finalize current message, prepare for new one."""
        self._queue.put(_NEW_SEGMENT)

    def finish(self) -> None:
        """Signal that the stream is complete."""
        self._queue.put(_DONE)

    async def run(self) -> str:
        """Drain queue, progressively edit platform message, return final content."""
        try:
            _raw_limit = self.adapter.MAX_MESSAGE_LENGTH
            _safe_limit = max(500, _raw_limit - 100)
        except (AttributeError, TypeError):
            _raw_limit = 4096
            _safe_limit = 3500

        try:
            while True:
                # Drain available items
                got_done = False
                got_segment_break = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is _DONE:
                            got_done = True
                            break
                        if item is _NEW_SEGMENT:
                            got_segment_break = True
                            break
                        self._accumulated += item
                    except queue.Empty:
                        break

                # Decide whether to flush
                now = time.monotonic()
                elapsed = now - self._last_edit_time
                should_edit = (
                    got_done or got_segment_break
                    or (elapsed >= self._current_edit_interval and self._accumulated)
                    or len(self._accumulated) >= self._buffer_threshold
                )

                if should_edit and self._accumulated:
                    # Don't send tiny fragments on first edit (avoid "<cursor>")
                    if not self._already_sent and len(self._accumulated) < 4 and not got_done:
                        pass  # Wait for more content
                    elif self._message_id is None:
                        # First send: create the message
                        display = self._accumulated
                        if not got_done:
                            display += " ▉"
                        result = await self.adapter.send_message(self.chat_id, display, reply_to_message_id=self._reply_to_message_id)
                        if result.success:
                            self._message_id = result.message_id
                            self._already_sent = True
                            self._last_sent_text = display
                            # Preserve msg_type from send_message for edit consistency.
                            # The adapter should communicate the type used; if not,
                            # leave None so edit_message auto-detects.
                            if hasattr(result, "msg_type") and result.msg_type:
                                self._msg_type = result.msg_type
                    else:
                        # Edit existing message
                        display = self._accumulated
                        await self._try_edit(display, add_cursor=not got_done)
                    self._last_edit_time = time.monotonic()

                # --- 截断检查：在发送前确保累积文本不超 safe_limit ---
                # 截断末尾以保留开头的内容完整性（避免破坏 Markdown 结构）
                if len(self._accumulated) > _safe_limit:
                    logger.warning(
                        "StreamConsumer accumulated text exceeds safe limit (%d > %d), truncating from end",
                        len(self._accumulated), _safe_limit,
                    )
                    self._accumulated = self._accumulated[:_safe_limit]

                # Segment break: finalize current message
                if got_segment_break:
                    self._message_id = None
                    self._accumulated = ""

                if got_done:
                    # Final send without cursor
                    if self._accumulated:
                        if self._message_id:
                            await self._try_edit(self._accumulated)
                        elif not self._already_sent:
                            result = await self.adapter.send_message(self.chat_id, self._accumulated, reply_to_message_id=self._reply_to_message_id)
                            if result.success:
                                self._message_id = result.message_id
                                self._already_sent = True
                    self.final_content = self._accumulated
                    return self.final_content

                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            if self._accumulated and self._message_id:
                try:
                    await self._try_edit(self._accumulated)
                except Exception:
                    pass
            self.final_content = self._accumulated
            return self.final_content
        except Exception as e:
            logger.error("StreamConsumer error: %s", e)
            self.final_content = self._accumulated
            return self.final_content

    async def _try_edit(self, text: str, *, add_cursor: bool = False, safe_limit: int = 0) -> None:
        """Try to edit platform message with flood control protection."""
        if safe_limit > 0 and len(text) > safe_limit:
            logger.warning(
                "_try_edit text exceeds safe limit (%d > %d), truncating",
                len(text), safe_limit,
            )
            text = text[:safe_limit]
        if add_cursor:
            text = text + " ▉"
        if self._message_id and text == self._last_sent_text:
            return
        self._last_sent_text = text

        if self._message_id and self._edit_supported:
            result = await self.adapter.edit_message(
                self.chat_id, self._message_id, text,
                msg_type=self._msg_type,
            )
            if result.success:
                self._flood_strikes = 0
                self._current_edit_interval = self._edit_interval
                self._already_sent = True
                return
            if self._is_flood_error(result.error):
                self._flood_strikes += 1
                self._current_edit_interval = min(self._current_edit_interval * 2, 10.0)
                self._last_edit_time = time.monotonic()
                if self._flood_strikes >= self._MAX_FLOOD_STRIKES:
                    self._edit_supported = False
                    logger.warning("Flood control: progressive edit disabled after %d strikes", self._flood_strikes)
                return
            self._edit_supported = False

        # Fallback: send new message
        result = await self.adapter.send_message(self.chat_id, text)
        if result.success:
            self._message_id = result.message_id
            self._already_sent = True

    @staticmethod
    def _is_flood_error(error: Optional[str]) -> bool:
        if not error:
            return False
        return any(code in error for code in ["99991400", "99991401", "99991402", "429 ", "rate limit"])
