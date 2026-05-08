"""Hierarchical task-path registry for sub-agent lifecycle management.

每个 session 的 TyAgent 实例持有一个 TaskTree 实例。spawn_task 时注册新
agent 为树上的一个叶子节点，路径自动层级继承（如 /root/main/db_query）。

支持：
- 相对和绝对路径解析
- 级联取消子树
- path_prefix 过滤
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional


# Sentinel for not-found lookups
_MISSING: Any = object()


class TaskNode:
    """A node in the task tree representing one spawned agent."""

    __slots__ = ("name", "parent", "children", "agent", "status")

    def __init__(self, name: str, agent: Any = None):
        self.name = name
        self.parent: Optional[TaskNode] = None
        # Children in insertion order so list_tasks is stable.
        self.children: Dict[str, TaskNode] = OrderedDict()
        self.agent = agent
        self.status = "running"  # running | completed | shutdown | error

    @property
    def path(self) -> str:
        """Canonical task path e.g. /root/main/subtask."""
        if self.parent is None:
            return f"/{self.name}"
        return f"{self.parent.path}/{self.name}"

    def descendant_paths(self) -> List[str]:
        """All paths in the subtree rooted at this node."""
        paths = [self.path]
        for child in self.children.values():
            paths.extend(child.descendant_paths())
        return paths

    def is_ancestor_of(self, other_path: str) -> bool:
        """Return True if *other_path* is in this node's subtree."""
        prefix = self.path if self.path.endswith("/") else self.path + "/"
        return other_path == self.path or other_path.startswith(prefix)


class TaskTree:
    """Hierarchical task-path registry for a session.

    The root node ``/root`` is created once per agent.  Each ``spawn_task``
    creates a child node whose canonical path is the concatenation of the
    parent path and the user-supplied ``task_name`` (after sanitisation).

    Paths can be referenced by:

    - **Relative name** (e.g. ``"db_query"``) — resolved in the caller's
      subtree first, then in siblings.
    - **Absolute path** (e.g. ``"/root/task_a/db_query"``) — looked up
      directly.

    Design notes:

    * Insertion order preserves ``list_tasks`` stability.
    * ``unregister`` cascade-removes the whole subtree so ``close_agent``
      doesn't leave dangling references for descendants.
    """

    _VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self, root_name: str = "root"):
        self._root = TaskNode(root_name)
        # Fast O(1) lookup by canonical path.  The OrderedDict here
        # preserves registration order for list_tasks.
        self._by_path: Dict[str, TaskNode] = OrderedDict()
        self._by_path[self._root.path] = self._root

    # ── properties ──────────────────────────────────────────────

    @property
    def root_path(self) -> str:
        return self._root.path

    # ── name utilities ──────────────────────────────────────────

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Normalise a user-supplied *task_name* to a path-safe identifier.

        Rules:
        - lowercase
        - non [a-z0-9_] → underscore
        - collapse consecutive underscores
        - strip leading/trailing underscores
        - empty input → 'task'
        """
        cleaned = re.sub(r"[^a-z0-9_]", "_", name.lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or "task"

    def _build_canonical_path(self, parent_path: str, task_name: str) -> str:
        """Derive canonical path without side-effects."""
        if task_name.startswith("/"):
            # Absolute paths must start with the tree root to prevent
            # path injection (e.g. "/etc/passwd" would corrupt _by_path).
            root_prefix = f"/{self._root.name}/"
            if not task_name.startswith(root_prefix) and task_name != f"/{self._root.name}":
                # Not under our root — sanitise and treat as relative
                name = self.sanitize_name(task_name.lstrip("/"))
                return f"{parent_path.rstrip('/')}/{name}"
            return task_name.rstrip("/")
        name = self.sanitize_name(task_name)
        return f"{parent_path.rstrip('/')}/{name}"

    # ── registration ────────────────────────────────────────────

    def register(
        self, parent_path: str, task_name: str, agent: Any
    ) -> str:
        """Register a new agent.  Returns the canonical path.

        Raises ValueError when the path is already taken.
        """
        path = self._build_canonical_path(parent_path, task_name)
        if path in self._by_path:
            raise ValueError(f"Task path already exists: {path}")

        parts = path.strip("/").split("/")
        node = self._root
        for i, part in enumerate(parts[1:], start=1):  # skip "root"
            existing = node.children.get(part)
            if existing is not None:
                node = existing
            else:
                new_node = TaskNode(
                    part,
                    agent if i == len(parts) - 1 else None,
                )
                new_node.parent = node
                node.children[part] = new_node
                node = new_node

        self._by_path[path] = node
        return path

    # ── lookup ──────────────────────────────────────────────────

    def lookup(self, path: str) -> Optional[TaskNode]:
        """Return the node at *path*, or None."""
        return self._by_path.get(path)

    def resolve(
        self, caller_path: str, target: str
    ) -> Optional[str]:
        """Resolve a relative or absolute target to a canonical path.

        Resolution order:
        1. Absolute paths — O(1) lookup.
        2. Relative name — try as child of *caller_path*.
        3. Relative name — try as sibling (child of caller's parent).
        4. None — not found.
        """
        if target.startswith("/"):
            return target if target in self._by_path else None

        sanitised = self.sanitize_name(target)

        # Try child
        candidate = f"{caller_path.rstrip('/')}/{sanitised}"
        if candidate in self._by_path:
            return candidate

        # Try sibling
        parent = caller_path.rsplit("/", 1)[0] if caller_path != "/root" else "/root"
        if parent and parent != caller_path:
            candidate = f"{parent.rstrip('/')}/{sanitised}"
            if candidate in self._by_path:
                return candidate

        return None

    def resolve_required(
        self, caller_path: str, target: str
    ) -> str:
        """Like :meth:`resolve`, but raises ValueError when not found."""
        result = self.resolve(caller_path, target)
        if result is None:
            raise ValueError(
                f"Task not found: {target} "
                f"(caller: {caller_path})"
            )
        return result

    # ── subtree / descendants ────────────────────────────────────

    def descendants(self, path: str) -> List[str]:
        """All paths in the subtree rooted at *path* (including itself)."""
        node = self._by_path.get(path)
        if node is None:
            return []
        return node.descendant_paths()

    # ── unregister ──────────────────────────────────────────────

    def unregister(self, path: str) -> Optional[TaskNode]:
        """Remove *path* and its entire subtree.  Returns the removed node."""
        node = self._by_path.get(path)
        if node is None:
            return None

        # Collect all paths to remove first (avoid dict-mutation-while-iterating)
        doomed = node.descendant_paths()
        for p in doomed:
            self._by_path.pop(p, None)

        # Detach from parent
        if node.parent is not None:
            node.parent.children.pop(node.name, None)

        node.parent = None
        return node

    # ── listing ─────────────────────────────────────────────────

    def all_paths(self) -> List[str]:
        """Return all registered paths in insertion order (root first)."""
        return list(self._by_path.keys())

    def filter_by_prefix(self, prefix: str) -> List[str]:
        """Paths whose canonical path starts with *prefix*."""
        pfx = prefix.rstrip("/")
        return [p for p in self._by_path if p == pfx or p.startswith(pfx + "/")]

    def path_status(self, path: str) -> Optional[str]:
        """Return the status string for *path*, or None."""
        node = self._by_path.get(path)
        return node.status if node else None

    def set_status(self, path: str, status: str) -> None:
        """Update the status of a node."""
        node = self._by_path.get(path)
        if node is not None:
            node.status = status

    def all_statuses(self) -> Dict[str, str]:
        """Return {path: status} for all registered nodes."""
        return {p: n.status for p, n in self._by_path.items()}
