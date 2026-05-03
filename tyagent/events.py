"""Event collector bridging subagent completion to parent agent loop."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional


class EventCollector:
    """Collects events from subagents for the parent chat() loop to consume.

    Pattern:
      1. Child completes → calls notify_child_done(task_id, result)
      2. Chat() loop → calls drain_completed() before each LLM turn
      3. If model exits while children run → call wait_next() to block
    """

    def __init__(self):
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._event = asyncio.Event()
        self._last_drained: List[Dict[str, Any]] = []  # Debug aid for loss investigation

    def notify_child_done(self, task_id: str, result: Dict[str, Any]) -> None:
        """Called by _run_child_async when a child agent completes."""
        self._completed[task_id] = result
        self._event.set()

    def drain_completed(self) -> List[Dict[str, Any]]:
        """Return a list of completed child events and clear buffer.

        Each event dict:
          {"type": "child_complete", "task_id": str, "result": {...}}
        """
        events = [
            {"type": "child_complete", "task_id": tid, "result": result}
            for tid, result in self._completed.items()
        ]
        self._last_drained = events  # Preserve for debugging in case caller crashes
        self._completed.clear()
        self._event.clear()
        return events

    def peek(self) -> bool:
        """Check if events are available without consuming."""
        return bool(self._completed)

    async def wait_next(self, timeout: Optional[float] = None) -> bool:
        """Block until the next event arrives (or timeout).

        Returns True if an event arrived, False on timeout.
        """
        if self._completed:
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
