"""ty-agent: A lightweight, extensible agent framework.

Entry point. Use `python -m ty_agent_cli` for the CLI.
"""

from ty_agent_cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
