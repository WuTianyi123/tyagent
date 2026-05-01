"""tyagent tools package.

Provides a simple tool registry and core tool implementations
(read_file, write_file, patch, search_files, terminal, execute_code,
browser_navigate, browser_snapshot, browser_click, browser_type,
browser_scroll, browser_back, browser_press, browser_get_images,
browser_vision, browser_console).
"""

from tyagent.tools.registry import registry, tool_error, tool_result

# Import core tools to trigger self-registration
import tyagent.tools.core  # noqa: F401
import tyagent.tools.browser_tools  # noqa: F401
import tyagent.tools.memory_tool  # noqa: F401
import tyagent.tools.search_tool  # noqa: F401
import tyagent.tools.delegate_tool  # noqa: F401

__all__ = ["registry", "tool_error", "tool_result"]
