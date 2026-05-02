"""Tests for EventCollector."""

from __future__ import annotations

import asyncio
import pytest

from tyagent.events import EventCollector


class TestEventCollector:
    async def test_empty_drain(self):
        c = EventCollector()
        assert c.drain_completed() == []
        assert c.peek() is False

    async def test_notify_and_drain(self):
        c = EventCollector()
        c.notify_child_done("abc", {"success": True, "summary": "done"})
        assert c.peek() is True
        events = c.drain_completed()
        assert len(events) == 1
        assert events[0]["task_id"] == "abc"
        assert events[0]["type"] == "child_complete"
        assert c.peek() is False  # cleared

    async def test_drain_clears_previous(self):
        c = EventCollector()
        c.notify_child_done("a", {"summary": "first"})
        c.drain_completed()
        assert c.peek() is False

    async def test_multiple_children(self):
        c = EventCollector()
        c.notify_child_done("a", {})
        c.notify_child_done("b", {})
        events = c.drain_completed()
        assert len(events) == 2
        task_ids = {e["task_id"] for e in events}
        assert task_ids == {"a", "b"}

    async def test_wait_next_returns_immediately_if_event_exists(self):
        c = EventCollector()
        c.notify_child_done("x", {})
        result = await c.wait_next(timeout=1.0)
        assert result is True

    async def test_wait_next_timeout(self):
        c = EventCollector()
        result = await c.wait_next(timeout=0.05)
        assert result is False

    async def test_wait_next_notified(self):
        c = EventCollector()

        async def notify_later():
            await asyncio.sleep(0.05)
            c.notify_child_done("x", {})

        async def waiter():
            return await c.wait_next(timeout=5.0)

        # Run concurrently
        waiter_task = asyncio.ensure_future(waiter())
        notifier_task = asyncio.ensure_future(notify_later())
        result, _ = await asyncio.gather(waiter_task, notifier_task)
        assert result is True


# Make pytest-asyncio work
pytestmark = pytest.mark.asyncio
