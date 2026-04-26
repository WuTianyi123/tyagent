"""tyagent: A lightweight, extensible agent framework.

Entry point. Use `python -m tyagent_cli` for the CLI.
"""

from tyagent_cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
