"""ty-agent tools package.

Provides a lightweight tool registry and core tool implementations
(read_file, write_file, patch, search_files, terminal, execute_code).
"""

from ty_agent.tools.registry import registry, tool_error, tool_result

# Import core tools to trigger self-registration
import ty_agent.tools.core  # noqa: F401

__all__ = ["registry", "tool_error", "tool_result"]
