"""Tests for sub-agent infrastructure — TaskTree and Mailbox."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tyagent.subagent.mailbox import (
    FinalNotification,
    InterAgentMessage,
    Mailbox,
)
from tyagent.subagent.task_tree import TaskTree

pytestmark = pytest.mark.asyncio


# ═══════════════════════════════════════════════════════════════
# TaskTree
# ═══════════════════════════════════════════════════════════════


class TestTaskTree:
    def test_root_path(self):
        tree = TaskTree()
        assert tree.root_path == "/root"

    def test_sanitize_name(self):
        assert TaskTree.sanitize_name("Database Query") == "database_query"
        assert TaskTree.sanitize_name("Hello World!") == "hello_world"
        assert TaskTree.sanitize_name("__test__") == "test"
        assert TaskTree.sanitize_name("") == "task"
        assert TaskTree.sanitize_name("   ") == "task"

    def test_register_and_lookup(self):
        tree = TaskTree()
        path = tree.register("/root", "task_a", agent="agent_a")
        assert path == "/root/task_a"

        node = tree.lookup("/root/task_a")
        assert node is not None
        assert node.agent == "agent_a"
        assert node.status == "running"

    def test_register_hierarchical(self):
        tree = TaskTree()
        tree.register("/root", "main", agent=None)
        path = tree.register("/root/main", "subtask", agent="child")
        assert path == "/root/main/subtask"
        assert tree.lookup("/root/main/subtask").agent == "child"

    def test_register_duplicate_raises(self):
        tree = TaskTree()
        tree.register("/root", "task_a", agent=None)
        with pytest.raises(ValueError, match="already exists"):
            tree.register("/root", "task_a", agent=None)

    def test_resolve_relative_child(self):
        tree = TaskTree()
        tree.register("/root", "task_a", agent=None)
        resolved = tree.resolve("/root", "task_a")
        assert resolved == "/root/task_a"

    def test_resolve_absolute(self):
        tree = TaskTree()
        tree.register("/root", "task_a", agent=None)
        resolved = tree.resolve("/root/main", "/root/task_a")
        assert resolved == "/root/task_a"

    def test_resolve_not_found(self):
        tree = TaskTree()
        assert tree.resolve("/root", "nope") is None

    def test_resolve_sibling(self):
        tree = TaskTree()
        tree.register("/root", "task_a", agent=None)
        tree.register("/root", "task_b", agent=None)
        # From /root/task_a, can find task_b as sibling
        resolved = tree.resolve("/root/task_a", "task_b")
        assert resolved == "/root/task_b"

    def test_descendants(self):
        tree = TaskTree()
        tree.register("/root", "main", agent=None)
        tree.register("/root/main", "sub1", agent=None)
        tree.register("/root/main", "sub2", agent=None)
        tree.register("/root", "side", agent=None)

        d = tree.descendants("/root/main")
        assert set(d) == {"/root/main", "/root/main/sub1", "/root/main/sub2"}

    def test_unregister_cascades(self):
        tree = TaskTree()
        tree.register("/root", "main", agent=None)
        tree.register("/root/main", "sub1", agent=None)
        tree.register("/root/main", "sub2", agent=None)

        tree.unregister("/root/main")
        assert tree.lookup("/root/main") is None
        assert tree.lookup("/root/main/sub1") is None
        assert tree.lookup("/root/main/sub2") is None

    def test_all_paths_root_first(self):
        tree = TaskTree()
        tree.register("/root", "a", agent=None)
        tree.register("/root", "b", agent=None)
        paths = tree.all_paths()
        assert paths[0] == "/root"
        assert "/root/a" in paths
        assert "/root/b" in paths

    def test_filter_by_prefix(self):
        tree = TaskTree()
        tree.register("/root", "main", agent=None)
        tree.register("/root/main", "sub1", agent=None)
        tree.register("/root", "side", agent=None)

        filtered = tree.filter_by_prefix("/root/main")
        assert set(filtered) == {"/root/main", "/root/main/sub1"}

    def test_set_and_get_status(self):
        tree = TaskTree()
        tree.register("/root", "task", agent=None)
        assert tree.path_status("/root/task") == "running"

        tree.set_status("/root/task", "completed")
        assert tree.path_status("/root/task") == "completed"

    def test_all_statuses(self):
        tree = TaskTree()
        tree.register("/root", "a", agent=None)
        tree.register("/root", "b", agent=None)
        tree.set_status("/root/a", "completed")
        statuses = tree.all_statuses()
        assert statuses["/root/a"] == "completed"
        assert statuses["/root/b"] == "running"


# ═══════════════════════════════════════════════════════════════
# Mailbox
# ═══════════════════════════════════════════════════════════════


class TestMailbox:
    def test_initial_empty(self):
        mb = Mailbox("/root/test")
        assert mb.peek() is False
        assert mb.owner_path == "/root/test"

    def test_send_and_drain_single(self):
        mb = Mailbox()
        fn = FinalNotification(
            task_path="/root/child", success=True,
            summary="done", error=None, duration_seconds=1.0,
        )
        mb.send(fn)
        assert mb.peek() is True

        items = mb.drain()
        assert len(items) == 1
        assert items[0] is fn
        assert mb.peek() is False

    def test_send_and_drain_multiple(self):
        mb = Mailbox()
        mb.send(FinalNotification(
            task_path="/root/a", success=True,
            summary="a done", error=None, duration_seconds=1.0,
        ))
        mb.send(InterAgentMessage(
            author="/root", recipient="/root/a",
            content="hello", trigger_turn=True,
        ))

        items = mb.drain()
        assert len(items) == 2
        assert isinstance(items[0], FinalNotification)
        assert isinstance(items[1], InterAgentMessage)

    async def test_wait_next_timeout(self):
        mb = Mailbox()
        result = await mb.wait_next(timeout=0.01)
        assert result is False

    async def test_wait_next_available(self):
        mb = Mailbox()
        mb.send(FinalNotification(
            task_path="/root/x", success=True,
            summary="ok", error=None, duration_seconds=0.0,
        ))
        # Should return immediately since message is queued
        result = await mb.wait_next(timeout=0.01)
        assert result is True

    async def test_wait_next_blocks_then_notified(self):
        mb = Mailbox()

        async def deliver_later():
            await asyncio.sleep(0.05)
            mb.send(FinalNotification(
                task_path="/root/late", success=True,
                summary="late", error=None, duration_seconds=0.0,
            ))

        asyncio.create_task(deliver_later())
        result = await mb.wait_next(timeout=1.0)
        assert result is True

    def test_drain_with_trigger_info_final(self):
        mb = Mailbox()
        mb.send(FinalNotification(
            task_path="/root/c", success=True,
            summary="done", error=None, duration_seconds=0.0,
        ))
        msgs, should_trigger = mb.drain_with_trigger_info()
        assert should_trigger is True
        assert len(msgs) == 1
        assert "done" in msgs[0]["content"]

    def test_drain_with_trigger_info_interagent(self):
        mb = Mailbox()
        mb.send(InterAgentMessage(
            author="/root", recipient="/root/a",
            content="hello", trigger_turn=False,
        ))
        msgs, should_trigger = mb.drain_with_trigger_info()
        assert should_trigger is False  # trigger_turn=False
        assert len(msgs) == 1
        assert "hello" in msgs[0]["content"]

    def test_drain_with_trigger_info_mixed(self):
        mb = Mailbox()
        mb.send(InterAgentMessage(
            author="/root", recipient="/root/a",
            content="no trigger", trigger_turn=False,
        ))
        mb.send(InterAgentMessage(
            author="/root", recipient="/root/a",
            content="trigger me", trigger_turn=True,
        ))
        msgs, should_trigger = mb.drain_with_trigger_info()
        assert should_trigger is True
        assert len(msgs) == 2
