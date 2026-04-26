"""Browser automation tools for tyagent.

Uses the ``agent-browser`` CLI (a Playwright-based Node.js tool) for local
headless browser automation.  Zero API keys required — just needs
``agent-browser`` and Chromium installed.

Session management:
- Each ``task_id`` gets its own browser session via ``--session-name``
- Sessions persist across tool calls within the same task
- Auto-cleanup on process exit via atexit hook

Install agent-browser:
    npm install -g agent-browser
    agent-browser install          # Download Chromium (one-time)

Usage:
    # Auto-registered; LLM calls them via function calling.
    browser_navigate(url="https://example.com")
    browser_snapshot()
    browser_click(ref="@e5")
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from tyagent.tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30.0
_MAX_SNAPSHOT_CHARS = 12_000
_SESSION_PREFIX = "tyagent"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_sessions: Dict[str, str] = {}
_sessions_lock = threading.Lock()


def _get_session_name(task_id: Optional[str] = None) -> str:
    """Get or create a session name for the given task_id."""
    task = task_id or "default"
    with _sessions_lock:
        if task not in _sessions:
            import uuid

            _sessions[task] = f"{_SESSION_PREFIX}_{task}_{uuid.uuid4().hex[:8]}"
        return _sessions[task]


def _close_all_sessions() -> None:
    """Close all active browser sessions on exit."""
    with _sessions_lock:
        names = list(_sessions.values())
        _sessions.clear()

    browser_cmd = _find_agent_browser(silent=True)
    if browser_cmd is None:
        return

    for name in names:
        try:
            cmd = [browser_cmd, "--session-name", name, "close"]
            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception:
            pass


atexit.register(_close_all_sessions)


# ---------------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------------

_cached_browser_cmd: Optional[str] = None


def _find_agent_browser(silent: bool = False) -> Optional[str]:
    """Find the agent-browser CLI executable.

    Search order:
    1. ``AGENT_BROWSER_CMD`` env var (explicit override)
    2. ``agent-browser`` on PATH
    3. Hermes local node_modules/.bin/agent-browser
    4. ``npx agent-browser`` fallback
    """
    global _cached_browser_cmd
    if _cached_browser_cmd is not None:
        return _cached_browser_cmd

    explicit = os.getenv("AGENT_BROWSER_CMD", "").strip()
    if explicit:
        _cached_browser_cmd = explicit
        return explicit

    # Try PATH
    found = shutil.which("agent-browser")
    if found:
        _cached_browser_cmd = found
        return found

    # Try Hermes local install
    hermes_node_bin = (
        os.path.expanduser("~/.hermes/hermes-agent/node_modules/.bin/agent-browser")
    )
    if os.path.isfile(hermes_node_bin) and os.access(hermes_node_bin, os.X_OK):
        _cached_browser_cmd = hermes_node_bin
        return hermes_node_bin

    # npx fallback
    if shutil.which("npx"):
        _cached_browser_cmd = "npx agent-browser"
        return _cached_browser_cmd

    if not silent:
        logger.warning(
            "agent-browser not found. Install: npm install -g agent-browser && agent-browser install"
        )
    return None


def _is_browser_available() -> bool:
    """Return True when agent-browser CLI is discoverable."""
    return _find_agent_browser(silent=True) is not None


# ---------------------------------------------------------------------------
# Low-level command runner
# ---------------------------------------------------------------------------


def _run_cmd(
    session_name: str,
    action: str,
    args: Optional[List[str]] = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Run an agent-browser command and return parsed result."""
    browser_cmd = _find_agent_browser()
    if browser_cmd is None:
        return {"success": False, "error": "agent-browser CLI not found"}

    cmd_parts = browser_cmd.split() if browser_cmd.startswith("npx ") else [browser_cmd]
    cmd = [*cmd_parts, "--session", session_name, action]
    if args:
        cmd.extend(str(a) for a in args)

    logger.debug("Browser cmd: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Browser command timed out after {timeout}s"}
    except Exception as exc:
        return {"success": False, "error": f"Failed to run browser: {exc}"}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Try JSON first (some commands support --json)
    if stdout.strip().startswith("{"):
        try:
            parsed = json.loads(stdout.strip())
            return parsed
        except json.JSONDecodeError:
            pass

    # Non-zero exit → error
    if proc.returncode != 0:
        err_text = stderr.strip() or stdout.strip()
        # Deduplicate Playwright "Executable doesn't exist" hint noise
        if "Executable doesn't exist" in err_text:
            err_text = (
                "Chromium not installed. Run: agent-browser install"
            )
        return {"success": False, "error": err_text or f"Exit code {proc.returncode}"}

    return {"success": True, "stdout": stdout.strip(), "stderr": stderr.strip()}


# ---------------------------------------------------------------------------
# Snapshot parser
# ---------------------------------------------------------------------------


def _parse_snapshot_text(text: str) -> Dict[str, Any]:
    """Parse agent-browser snapshot text into structured data.

    Extracts ref IDs (e.g., [ref=e1]) and builds a mapping.
    """
    refs: Dict[str, Dict[str, str]] = {}
    # Pattern: [ref=eN] or [ref=eN] [level=M]
    for match in re.finditer(
        r'\[ref=(e\d+)\](?:\s*\[level=(\d+)\])?',
        text,
    ):
        ref_id = match.group(1)
        level = match.group(2)
        # Extract nearby role/name from the line
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        refs[ref_id] = {"line": line}
        if level:
            refs[ref_id]["level"] = level

    return {"snapshot": text, "refs": refs, "ref_count": len(refs)}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_browser_navigate(args: Dict[str, Any]) -> str:
    url = args.get("url", "").strip()
    if not url:
        return tool_error("url is required")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    result = _run_cmd(session, "open", [url], timeout=max(30, _DEFAULT_TIMEOUT))

    if not result.get("success"):
        return tool_error(result.get("error", "Navigation failed"))

    stdout = result.get("stdout", "")
    # First line is usually "✓ Page Title"
    title = ""
    final_url = url
    for line in stdout.splitlines()[:3]:
        if line.startswith("✓ "):
            title = line[2:].strip()
        elif line.startswith("  http"):
            final_url = line.strip()

    # Auto-snapshot after navigate for convenience
    snap = _run_cmd(session, "snapshot", timeout=_DEFAULT_TIMEOUT)
    snap_data = {}
    if snap.get("success"):
        snap_data = _parse_snapshot_text(snap.get("stdout", ""))

    return tool_result(
        success=True,
        url=final_url,
        title=title,
        snapshot=snap_data.get("snapshot", "")[:_MAX_SNAPSHOT_CHARS],
        refs=snap_data.get("refs", {}),
        ref_count=snap_data.get("ref_count", 0),
    )


def _handle_browser_snapshot(args: Dict[str, Any]) -> str:
    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    full = bool(args.get("full", False))

    action = "snapshot"
    cmd_args: List[str] = []
    if not full:
        cmd_args.append("-i")  # interactive elements only

    result = _run_cmd(session, action, cmd_args, timeout=_DEFAULT_TIMEOUT)
    if not result.get("success"):
        return tool_error(result.get("error", "Snapshot failed"))

    data = _parse_snapshot_text(result.get("stdout", ""))
    snapshot_text = data.get("snapshot", "")

    # Truncate if too long
    truncated = False
    if len(snapshot_text) > _MAX_SNAPSHOT_CHARS:
        snapshot_text = snapshot_text[:_MAX_SNAPSHOT_CHARS]
        truncated = True

    return tool_result(
        success=True,
        snapshot=snapshot_text,
        refs=data.get("refs", {}),
        ref_count=data.get("ref_count", 0),
        truncated=truncated,
    )


def _handle_browser_click(args: Dict[str, Any]) -> str:
    ref = args.get("ref", "").strip()
    if not ref:
        return tool_error("ref is required (e.g., '@e5')")
    if not ref.startswith("@"):
        ref = "@" + ref

    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    result = _run_cmd(session, "click", [ref], timeout=_DEFAULT_TIMEOUT)

    if not result.get("success"):
        return tool_error(result.get("error", "Click failed"))

    return tool_result(success=True, action="click", ref=ref)


def _handle_browser_type(args: Dict[str, Any]) -> str:
    ref = args.get("ref", "").strip()
    text = args.get("text", "")
    if not ref:
        return tool_error("ref is required")
    if not ref.startswith("@"):
        ref = "@" + ref

    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    # Use "fill" to clear first, then type
    result = _run_cmd(session, "fill", [ref, text], timeout=_DEFAULT_TIMEOUT)

    if not result.get("success"):
        return tool_error(result.get("error", "Type failed"))

    return tool_result(success=True, action="type", ref=ref)


def _handle_browser_scroll(args: Dict[str, Any]) -> str:
    direction = args.get("direction", "down")
    if direction not in ("up", "down"):
        return tool_error("direction must be 'up' or 'down'")

    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    result = _run_cmd(session, "scroll", [direction, "500"], timeout=_DEFAULT_TIMEOUT)

    if not result.get("success"):
        return tool_error(result.get("error", "Scroll failed"))

    return tool_result(success=True, action="scroll", direction=direction)


def _handle_browser_back(args: Dict[str, Any]) -> str:
    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    # agent-browser doesn't have a native "back" command; use eval
    result = _run_cmd(session, "eval", ["history.back()"], timeout=_DEFAULT_TIMEOUT)

    if not result.get("success"):
        return tool_error(result.get("error", "Back navigation failed"))

    # Small delay for navigation
    time.sleep(0.5)
    # Return snapshot
    snap = _run_cmd(session, "snapshot", ["-i"], timeout=_DEFAULT_TIMEOUT)
    snap_data = {}
    if snap.get("success"):
        snap_data = _parse_snapshot_text(snap.get("stdout", ""))

    return tool_result(
        success=True,
        action="back",
        snapshot=snap_data.get("snapshot", "")[:_MAX_SNAPSHOT_CHARS],
        refs=snap_data.get("refs", {}),
        ref_count=snap_data.get("ref_count", 0),
    )


def _handle_browser_press(args: Dict[str, Any]) -> str:
    key = args.get("key", "").strip()
    if not key:
        return tool_error("key is required (e.g., 'Enter', 'Tab')")

    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    result = _run_cmd(session, "press", [key], timeout=_DEFAULT_TIMEOUT)

    if not result.get("success"):
        return tool_error(result.get("error", f"Key press '{key}' failed"))

    return tool_result(success=True, action="press", key=key)


def _handle_browser_get_images(args: Dict[str, Any]) -> str:
    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    result = _run_cmd(
        session, "eval", ["Array.from(document.images).map(i => ({src: i.src, alt: i.alt, width: i.width, height: i.height}))"],
        timeout=_DEFAULT_TIMEOUT,
    )

    if not result.get("success"):
        return tool_error(result.get("error", "Failed to get images"))

    # Parse eval output
    stdout = result.get("stdout", "")
    images: List[Dict[str, Any]] = []
    try:
        # eval output might be JSON string or plain text
        if stdout.startswith("[") or stdout.startswith("{"):
            parsed = json.loads(stdout)
            if isinstance(parsed, list):
                images = parsed
        else:
            # Try extracting JSON from output
            match = re.search(r"(\[.*\])", stdout.replace("\n", ""))
            if match:
                images = json.loads(match.group(1))
    except Exception:
        pass

    return tool_result(
        success=True,
        images=images[:50],  # cap
        count=len(images),
    )


def _handle_browser_vision(args: Dict[str, Any]) -> str:
    """Take a screenshot and return the path for vision analysis."""
    task_id = args.get("task_id")
    session = _get_session_name(task_id)
    question = args.get("question", "Describe this page")
    annotate = bool(args.get("annotate", False))

    # Screenshot path
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, f"tyagent_browser_{session_name_safe(session)}.png")

    cmd_args = [path]
    if annotate:
        cmd_args.append("--annotate")

    result = _run_cmd(session, "screenshot", cmd_args, timeout=_DEFAULT_TIMEOUT)
    if not result.get("success"):
        return tool_error(result.get("error", "Screenshot failed"))

    return tool_result(
        success=True,
        screenshot_path=path,
        question=question,
        hint=f"Use vision_analyze with image_url='{path}' and question='{question}'",
    )


def _handle_browser_console(args: Dict[str, Any]) -> str:
    expression = args.get("expression")
    clear = bool(args.get("clear", False))
    task_id = args.get("task_id")
    session = _get_session_name(task_id)

    if expression:
        result = _run_cmd(session, "eval", [expression], timeout=_DEFAULT_TIMEOUT)
        if not result.get("success"):
            return tool_error(result.get("error", "Eval failed"))
        return tool_result(
            success=True,
            expression=expression,
            result=result.get("stdout", ""),
        )

    # No expression — return console logs (not directly supported by agent-browser,
    # so we return a helpful message)
    return tool_result(
        success=True,
        message="Console logs not available in this browser backend. Use 'expression' to run JavaScript.",
    )


def session_name_safe(name: str) -> str:
    """Sanitize session name for filesystem paths."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

BROWSER_NAVIGATE_SCHEMA = {
    "name": "browser_navigate",
    "description": (
        "Navigate to a URL in the browser. Initializes the session and loads the page. "
        "Must be called before other browser tools. Returns a compact page snapshot with interactive elements and ref IDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to (e.g., 'https://example.com')"},
        },
        "required": ["url"],
    },
}

BROWSER_SNAPSHOT_SCHEMA = {
    "name": "browser_snapshot",
    "description": (
        "Get a text-based snapshot of the current page's accessibility tree. "
        "Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. "
        "full=false (default): compact view with interactive elements only. "
        "Requires browser_navigate first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "full": {
                "type": "boolean",
                "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.",
                "default": False,
            },
        },
        "required": [],
    },
}

BROWSER_CLICK_SCHEMA = {
    "name": "browser_click",
    "description": (
        "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). "
        "The ref IDs are shown in square brackets in the snapshot output. "
        "Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"},
        },
        "required": ["ref"],
    },
}

BROWSER_TYPE_SCHEMA = {
    "name": "browser_type",
    "description": (
        "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. "
        "Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "The element reference from the snapshot (e.g., '@e3')"},
            "text": {"type": "string", "description": "The text to type into the field"},
        },
        "required": ["ref", "text"],
    },
}

BROWSER_SCROLL_SCHEMA = {
    "name": "browser_scroll",
    "description": (
        "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. "
        "Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down"], "description": "Direction to scroll"},
        },
        "required": ["direction"],
    },
}

BROWSER_BACK_SCHEMA = {
    "name": "browser_back",
    "description": (
        "Navigate back to the previous page in browser history. Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

BROWSER_PRESS_SCHEMA = {
    "name": "browser_press",
    "description": (
        "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. "
        "Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')"},
        },
        "required": ["key"],
    },
}

BROWSER_GET_IMAGES_SCHEMA = {
    "name": "browser_get_images",
    "description": (
        "Get a list of all images on the current page with their URLs and alt text. "
        "Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

BROWSER_VISION_SCHEMA = {
    "name": "browser_vision",
    "description": (
        "Take a screenshot of the current page. Use this when you need to visually understand what's on the page "
        "- especially useful for CAPTCHAs, visual verification challenges, complex layouts, or when the text snapshot doesn't capture important visual information. "
        "Returns the screenshot path. Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What you want to know about the page visually. Be specific about what you're looking for.",
            },
            "annotate": {
                "type": "boolean",
                "default": False,
                "description": "If true, overlay numbered labels on interactive elements for spatial reasoning.",
            },
        },
        "required": ["question"],
    },
}

BROWSER_CONSOLE_SCHEMA = {
    "name": "browser_console",
    "description": (
        "Execute JavaScript in the page context and return the result. "
        "Runs in the browser like DevTools console — full access to DOM, window, document. "
        "Return values are serialized. Example: 'document.title' or 'document.querySelectorAll(\"a\").length'. "
        "Requires browser_navigate to be called first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "JavaScript expression to evaluate in the page context",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="browser_navigate",
    schema=BROWSER_NAVIGATE_SCHEMA,
    handler=_handle_browser_navigate,
    description="Navigate to a URL in the browser",
    emoji="🌐",
)
registry.register(
    name="browser_snapshot",
    schema=BROWSER_SNAPSHOT_SCHEMA,
    handler=_handle_browser_snapshot,
    description="Get a text-based snapshot of the current page",
    emoji="📸",
)
registry.register(
    name="browser_click",
    schema=BROWSER_CLICK_SCHEMA,
    handler=_handle_browser_click,
    description="Click an element by ref ID",
    emoji="👆",
)
registry.register(
    name="browser_type",
    schema=BROWSER_TYPE_SCHEMA,
    handler=_handle_browser_type,
    description="Type text into an input field",
    emoji="⌨️",
)
registry.register(
    name="browser_scroll",
    schema=BROWSER_SCROLL_SCHEMA,
    handler=_handle_browser_scroll,
    description="Scroll the page up or down",
    emoji="📜",
)
registry.register(
    name="browser_back",
    schema=BROWSER_BACK_SCHEMA,
    handler=_handle_browser_back,
    description="Navigate back in browser history",
    emoji="◀️",
)
registry.register(
    name="browser_press",
    schema=BROWSER_PRESS_SCHEMA,
    handler=_handle_browser_press,
    description="Press a keyboard key",
    emoji="⌨️",
)
registry.register(
    name="browser_get_images",
    schema=BROWSER_GET_IMAGES_SCHEMA,
    handler=_handle_browser_get_images,
    description="Get all images on the current page",
    emoji="🖼️",
)
registry.register(
    name="browser_vision",
    schema=BROWSER_VISION_SCHEMA,
    handler=_handle_browser_vision,
    description="Take a screenshot of the current page",
    emoji="👁️",
)
registry.register(
    name="browser_console",
    schema=BROWSER_CONSOLE_SCHEMA,
    handler=_handle_browser_console,
    description="Execute JavaScript in the page context",
    emoji="🖥️",
)
