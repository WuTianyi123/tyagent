"""Core tool implementations for tyagent.

read_file  — Read text files with line numbers and pagination.
write_file — Write content to a file, creating parent directories.
patch      — Targeted find-and-replace edits.
search_files — Search file contents or find files by name.
terminal   — Execute shell commands.
execute_code — Run Python code in a subprocess sandbox.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from tyagent.tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_MAX_READ_CHARS = 100_000
_DEFAULT_READ_LIMIT = 500
_DEFAULT_READ_MAX_LIMIT = 2000

_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})

_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov", ".mkv",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pyc", ".pyo", ".class", ".o", ".a",
    ".db", ".sqlite", ".sqlite3",
    ".ttf", ".otf", ".woff", ".woff2",
})


def _resolve_path(filepath: str) -> Path:
    """Resolve a path relative to CWD, supporting ~/ expansion."""
    p = Path(filepath).expanduser()
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return p.resolve()


def _is_blocked_device(filepath: str) -> bool:
    """Return True for device paths that would hang or produce infinite output."""
    normalized = os.path.expanduser(filepath)
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    if normalized.startswith("/proc/") and normalized.endswith(("/fd/0", "/fd/1", "/fd/2")):
        return True
    return False


def _is_binary_file(filepath: str) -> bool:
    """Check if a file has a known binary extension."""
    return Path(filepath).suffix.lower() in _BINARY_EXTENSIONS


def _truncate_output(text: str, max_chars: int = _DEFAULT_MAX_READ_CHARS) -> tuple[str, bool]:
    """Truncate text to max_chars and return (truncated_text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": (
        "Read a text file with line numbers and pagination. "
        "Use this instead of cat/head/tail in terminal. "
        "Output format: 'LINE_NUM|CONTENT'. "
        "Use offset and limit for large files. "
        "NOTE: Cannot read images or binary files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (absolute, relative, or ~/path)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed, default: 1)",
                "default": 1,
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: 500, max: 2000)",
                "default": 500,
                "maximum": 2000,
            },
        },
        "required": ["path"],
    },
}


def _handle_read_file(args: Dict[str, Any]) -> str:
    """Read a text file with line numbers and pagination."""
    path = args.get("path", "")
    offset = max(1, int(args.get("offset", 1)))
    limit = max(1, min(int(args.get("limit", _DEFAULT_READ_LIMIT)), _DEFAULT_READ_MAX_LIMIT))

    # Block device paths
    if _is_blocked_device(path):
        return tool_error(
            f"Cannot read '{path}': this is a device file that would block or produce infinite output."
        )

    try:
        resolved = _resolve_path(path)
    except (OSError, ValueError) as exc:
        return tool_error(f"Invalid path '{path}': {exc}")

    # Binary file guard
    if _is_binary_file(str(resolved)):
        ext = resolved.suffix.lower()
        return tool_error(
            f"Cannot read binary file '{path}' ({ext}). Use other tools for binary data."
        )

    if not resolved.exists():
        # Suggest similar filenames
        parent = resolved.parent
        if parent.exists():
            similar = [f.name for f in parent.iterdir() if f.is_file()]
            similar = [f for f in similar if f.lower().startswith(resolved.name.lower()[:3])]
            if similar:
                return tool_error(
                    f"File not found: {path}. Did you mean: {', '.join(similar)}?"
                )
        return tool_error(f"File not found: {path}")

    if not resolved.is_file():
        return tool_error(f"Not a file: {path}")

    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = resolved.read_text(encoding="latin-1")
    except (OSError, PermissionError) as exc:
        return tool_error(f"Cannot read '{path}': {exc}")

    # Character-count guard
    content, was_truncated = _truncate_output(content, _DEFAULT_MAX_READ_CHARS)
    if was_truncated:
        total_lines = content.count("\n") + 1
        return tool_error(
            f"File exceeds {_DEFAULT_MAX_READ_CHARS:,} characters. "
            f"Use offset and limit to read a smaller range. "
            f"File has approximately {total_lines} lines."
        )

    lines = content.splitlines()
    total_lines = len(lines)

    # Pagination
    start_idx = offset - 1
    if start_idx >= total_lines:
        return tool_result(
            content="",
            total_lines=total_lines,
            offset=offset,
            limit=limit,
            hint=f"File has {total_lines} lines. Use offset <= {total_lines}.",
        )

    end_idx = min(start_idx + limit, total_lines)
    selected = lines[start_idx:end_idx]

    # Format with line numbers
    formatted = "\n".join(f"{i + 1}|{line}" for i, line in enumerate(selected, start=start_idx))

    result: Dict[str, Any] = {
        "content": formatted,
        "total_lines": total_lines,
        "offset": offset,
        "limit": limit,
        "truncated": end_idx < total_lines,
    }
    if end_idx < total_lines:
        result["hint"] = f"Use offset={end_idx + 1} to continue reading."

    return tool_result(result)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": (
        "Write content to a file, completely replacing existing content. "
        "Use this instead of echo/cat heredoc in terminal. "
        "Creates parent directories automatically. "
        "OVERWRITES the entire file — use 'patch' for targeted edits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)",
            },
            "content": {
                "type": "string",
                "description": "Complete content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
}


def _handle_write_file(args: Dict[str, Any]) -> str:
    """Write content to a file."""
    path = args.get("path", "")
    content = args.get("content", "")

    try:
        resolved = _resolve_path(path)
    except (OSError, ValueError) as exc:
        return tool_error(f"Invalid path '{path}': {exc}")

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return tool_result(
            success=True,
            path=str(resolved),
            bytes_written=len(content.encode("utf-8")),
        )
    except (OSError, PermissionError) as exc:
        return tool_error(f"Failed to write '{path}': {exc}")


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------

PATCH_SCHEMA = {
    "name": "patch",
    "description": (
        "Targeted find-and-replace edits in files. "
        "Use this instead of sed/awk in terminal.\n\n"
        "Replace mode: find a unique string and replace it. "
        "Include enough surrounding context to ensure uniqueness."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Text to find in the file. Must be unique unless replace_all=true. Include enough surrounding context to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text. Can be empty string to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
}


def _handle_patch(args: Dict[str, Any]) -> str:
    """Patch a file using find-and-replace."""
    path = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))

    if old_string is None or new_string is None:
        return tool_error("old_string and new_string are required")

    try:
        resolved = _resolve_path(path)
    except (OSError, ValueError) as exc:
        return tool_error(f"Invalid path '{path}': {exc}")

    if not resolved.exists():
        return tool_error(f"File not found: {path}")

    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, PermissionError) as exc:
        return tool_error(f"Cannot read '{path}': {exc}")

    if replace_all:
        if old_string not in content:
            return tool_error(
                f"Could not find old_string in '{path}'. "
                "The file may have changed. Use read_file to verify current content."
            )
        new_content = content.replace(old_string, new_string)
        count = content.count(old_string)
    else:
        count = content.count(old_string)
        if count == 0:
            return tool_error(
                f"Could not find old_string in '{path}'. "
                "The file may have changed. Use read_file to verify current content."
            )
        if count > 1:
            return tool_error(
                f"old_string appears {count} times in '{path}'. "
                "Include more surrounding context to make it unique, or use replace_all=true."
            )
        new_content = content.replace(old_string, new_string, 1)
        count = 1

    try:
        resolved.write_text(new_content, encoding="utf-8")
    except (OSError, PermissionError) as exc:
        return tool_error(f"Failed to write '{path}': {exc}")

    return tool_result(
        success=True,
        path=str(resolved),
        replacements=count,
    )


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": (
        "Search file contents or find files by name. "
        "Use this instead of grep/rg/find/ls in terminal.\n\n"
        "Content search (target='content'): Regex search inside files. "
        "File search (target='files'): Find files by glob pattern."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern for content search, or glob pattern for file search",
            },
            "target": {
                "type": "string",
                "enum": ["content", "files"],
                "description": "'content' searches inside file contents, 'files' searches for files by name",
                "default": "content",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current working directory)",
                "default": ".",
            },
            "file_glob": {
                "type": "string",
                "description": "Filter files by pattern in grep mode (e.g., '*.py')",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 50)",
                "default": 50,
            },
            "offset": {
                "type": "integer",
                "description": "Skip first N results for pagination (default: 0)",
                "default": 0,
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_only", "count"],
                "description": "Output format: 'content' shows matching lines, 'files_only' lists file paths, 'count' shows match counts per file",
                "default": "content",
            },
            "context": {
                "type": "integer",
                "description": "Number of context lines before and after each match (grep mode only)",
                "default": 0,
            },
        },
        "required": ["pattern"],
    },
}


def _handle_search_files(args: Dict[str, Any]) -> str:
    """Search file contents or find files by name."""
    pattern = args.get("pattern", "")
    target = args.get("target", "content")
    path = args.get("path", ".")
    file_glob = args.get("file_glob")
    limit = max(1, int(args.get("limit", 50)))
    offset = max(0, int(args.get("offset", 0)))
    output_mode = args.get("output_mode", "content")
    context_lines = max(0, int(args.get("context", 0)))

    try:
        search_path = _resolve_path(path)
    except (OSError, ValueError) as exc:
        return tool_error(f"Invalid search path '{path}': {exc}")

    if target == "files":
        return _search_files_by_name(pattern, search_path, limit, offset)

    return _search_file_content(
        pattern, search_path, file_glob, limit, offset, output_mode, context_lines
    )


def _search_files_by_name(
    pattern: str, search_path: Path, limit: int, offset: int
) -> str:
    """Find files by glob pattern."""
    import fnmatch

    if search_path.is_file():
        candidates = [search_path]
    else:
        candidates = list(search_path.rglob(pattern))

    # Sort by modification time (newest first), then by name
    candidates = sorted(
        candidates,
        key=lambda p: (p.stat().st_mtime if p.exists() else 0, str(p)),
        reverse=True,
    )

    total = len(candidates)
    page = candidates[offset : offset + limit]
    paths = [str(p) for p in page]

    return tool_result(
        total_count=total,
        matches=paths,
        offset=offset,
        limit=limit,
        truncated=offset + limit < total,
    )


def _search_file_content(
    pattern: str,
    search_path: Path,
    file_glob: Optional[str],
    limit: int,
    offset: int,
    output_mode: str,
    context_lines: int,
) -> str:
    """Search file contents using ripgrep (rg). Falls back to Python regex if rg unavailable."""
    # Try ripgrep first
    rg_cmd = ["rg", "--json", "--max-count", str(limit + offset), "-n"]
    if context_lines > 0:
        rg_cmd.extend(["-C", str(context_lines)])
    if file_glob:
        rg_cmd.extend(["-g", file_glob])
    rg_cmd.extend(["--", pattern])
    if search_path.is_file():
        rg_cmd.append(str(search_path))
    else:
        rg_cmd.append(str(search_path))

    try:
        proc = subprocess.run(
            rg_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        # ripgrep not available, fall back to Python regex
        return _search_content_python(
            pattern, search_path, file_glob, limit, offset, output_mode, context_lines
        )
    except subprocess.TimeoutExpired:
        return tool_error("Search timed out after 30 seconds. Try a more specific pattern or narrower path.")

    matches: List[Dict[str, Any]] = []
    files_with_matches: set = set()
    match_counts: Dict[str, int] = {}

    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg_type = obj.get("type")
        if msg_type == "begin":
            files_with_matches.add(obj.get("data", {}).get("path", {}).get("text", ""))
        elif msg_type == "match":
            data = obj.get("data", {})
            path_text = data.get("path", {}).get("text", "")
            line_num = data.get("line_number", 0)
            submatches = data.get("submatches", [])
            for sm in submatches:
                text = sm.get("match", {}).get("text", "")
                files_with_matches.add(path_text)
                match_counts[path_text] = match_counts.get(path_text, 0) + 1
                if output_mode == "content":
                    matches.append({
                        "path": path_text,
                        "line": line_num,
                        "content": text,
                    })
        elif msg_type == "summary":
            stats = obj.get("data", {})
            # total matches from summary
            pass

    if output_mode == "count":
        return tool_result(
            total_count=sum(match_counts.values()),
            match_counts=match_counts,
        )

    if output_mode == "files_only":
        files = sorted(files_with_matches)
        total = len(files)
        page = files[offset : offset + limit]
        return tool_result(
            total_count=total,
            matches=page,
            offset=offset,
            limit=limit,
            truncated=offset + limit < total,
        )

    # content mode
    total = len(matches)
    page = matches[offset : offset + limit]
    return tool_result(
        total_count=total,
        matches=page,
        offset=offset,
        limit=limit,
        truncated=offset + limit < total,
    )


def _search_content_python(
    pattern: str,
    search_path: Path,
    file_glob: Optional[str],
    limit: int,
    offset: int,
    output_mode: str,
    context_lines: int,
) -> str:
    """Fallback file content search using Python regex."""
    import re

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return tool_error(f"Invalid regex pattern: {exc}")

    matches: List[Dict[str, Any]] = []
    files_with_matches: set = set()
    match_counts: Dict[str, int] = {}

    if search_path.is_file():
        candidates = [search_path]
    else:
        candidates = [p for p in search_path.rglob("*") if p.is_file()]
        if file_glob:
            import fnmatch
            candidates = [p for p in candidates if fnmatch.fnmatch(p.name, file_glob)]

    for file_path in candidates:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                files_with_matches.add(str(file_path))
                match_counts[str(file_path)] = match_counts.get(str(file_path), 0) + 1
                if output_mode == "content":
                    matches.append({
                        "path": str(file_path),
                        "line": i,
                        "content": line,
                    })
                    if len(matches) >= limit + offset:
                        break
        if len(matches) >= limit + offset:
            break

    if output_mode == "count":
        return tool_result(
            total_count=sum(match_counts.values()),
            match_counts=match_counts,
        )

    if output_mode == "files_only":
        files = sorted(files_with_matches)
        total = len(files)
        page = files[offset : offset + limit]
        return tool_result(
            total_count=total,
            matches=page,
            offset=offset,
            limit=limit,
            truncated=offset + limit < total,
        )

    total = len(matches)
    page = matches[offset : offset + limit]
    return tool_result(
        total_count=total,
        matches=page,
        offset=offset,
        limit=limit,
        truncated=offset + limit < total,
    )


# ---------------------------------------------------------------------------
# terminal
# ---------------------------------------------------------------------------

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": (
        "Execute a shell command on the VM. "
        "Returns output, exit_code, and error if any. "
        "For long-running tasks (>180s), prefer background processes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default: 180). Returns instantly when command finishes.",
                "minimum": 1,
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory.",
            },
        },
        "required": ["command"],
    },
}


def _handle_terminal(args: Dict[str, Any]) -> str:
    """Execute a shell command."""
    command = args.get("command", "")
    timeout = int(args.get("timeout", 180))
    workdir = args.get("workdir")

    if not isinstance(command, str) or not command.strip():
        return tool_error("Invalid command: expected non-empty string")

    # Safety: strip trailing & to avoid background confusion (we don't support it here)
    command = command.strip()

    cwd = None
    if workdir:
        try:
            cwd = str(_resolve_path(workdir))
        except (OSError, ValueError) as exc:
            return tool_error(f"Invalid workdir '{workdir}': {exc}")

    try:
        # Ensure isolated HOME is inherited by subprocess
        env = os.environ.copy()
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        output = proc.stdout
        stderr = proc.stderr
        if stderr:
            output = (output or "") + "\n" + stderr

        # Truncate very large output
        max_out = 50_000
        was_truncated = False
        if len(output) > max_out:
            output = output[:max_out]
            was_truncated = True

        result: Dict[str, Any] = {
            "output": output,
            "exit_code": proc.returncode,
        }
        if was_truncated:
            result["truncated"] = True
            result["hint"] = "Output was truncated. Use grep/head/tail or redirect to a file for large outputs."
        if proc.returncode != 0:
            result["error"] = f"Command exited with code {proc.returncode}"
        return tool_result(result)
    except subprocess.TimeoutExpired:
        return tool_error(
            f"Command timed out after {timeout} seconds. "
            "For long-running tasks, consider running in background or increasing timeout."
        )
    except Exception as exc:
        return tool_error(f"Command execution failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------

EXECUTE_CODE_SCHEMA = {
    "name": "execute_code",
    "description": (
        "Run a Python script and return its stdout/stderr. "
        "Use this for data processing, file analysis, or any task better done in code. "
        "Scripts run in a fresh subprocess with the active venv's Python. "
        "Print your final result to stdout."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use standard library (json, re, math, csv, datetime, collections, etc.) for processing between tool calls.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default: 300, max: 600).",
                "default": 300,
                "minimum": 1,
            },
        },
        "required": ["code"],
    },
}


def _handle_execute_code(args: Dict[str, Any]) -> str:
    """Execute Python code in a subprocess."""
    code = args.get("code", "")
    timeout = min(int(args.get("timeout", 300)), 600)

    if not isinstance(code, str) or not code.strip():
        return tool_error("Invalid code: expected non-empty string")

    # Write code to a temporary file
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name
    except OSError as exc:
        return tool_error(f"Failed to create temp file: {exc}")

    try:
        # Use the same Python interpreter
        python = os.environ.get("PYTHON", sys.executable)
        # Ensure isolated HOME is inherited by subprocess
        env = os.environ.copy()
        proc = subprocess.run(
            [python, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
            env=env,
        )
        output = proc.stdout
        stderr = proc.stderr
        if stderr:
            output = (output or "") + "\n[stderr]:\n" + stderr

        # Truncate very large output
        max_out = 50_000
        was_truncated = False
        if len(output) > max_out:
            output = output[:max_out]
            was_truncated = True

        result: Dict[str, Any] = {
            "output": output,
            "exit_code": proc.returncode,
        }
        if was_truncated:
            result["truncated"] = True
            result["hint"] = "Output was truncated. Consider writing results to a file instead of printing everything."
        if proc.returncode != 0:
            result["error"] = f"Script exited with code {proc.returncode}"
        return tool_result(result)
    except subprocess.TimeoutExpired:
        return tool_error(f"Script timed out after {timeout} seconds.")
    except Exception as exc:
        return tool_error(f"Script execution failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="read_file",
    schema=READ_FILE_SCHEMA,
    handler=_handle_read_file,
    description="Read a text file with line numbers and pagination",
    emoji="📖",
)
registry.register(
    name="write_file",
    schema=WRITE_FILE_SCHEMA,
    handler=_handle_write_file,
    description="Write content to a file",
    emoji="✍️",
)
registry.register(
    name="patch",
    schema=PATCH_SCHEMA,
    handler=_handle_patch,
    description="Find-and-replace edits in files",
    emoji="🔧",
)
registry.register(
    name="search_files",
    schema=SEARCH_FILES_SCHEMA,
    handler=_handle_search_files,
    description="Search file contents or find files by name",
    emoji="🔎",
)
registry.register(
    name="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    description="Execute shell commands",
    emoji="💻",
)
registry.register(
    name="execute_code",
    schema=EXECUTE_CODE_SCHEMA,
    handler=_handle_execute_code,
    description="Run Python code in a subprocess",
    emoji="🐍",
)
