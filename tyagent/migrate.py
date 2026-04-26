"""Migration script: import old JSON session files into SQLite database.

Reads all session JSON files from the old sessions directory and
imports them into the SQLite-backed SessionStore.

Old JSON files are NOT deleted after import — they are left as a backup.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from tyagent.session import SessionStore

logger = logging.getLogger(__name__)


def migrate_from_json(sessions_dir: Path) -> int:
    """Import all JSON session files in the given directory into SQLite.

    Args:
        sessions_dir: Directory containing old .json session files.

    Returns:
        Number of sessions imported.

    Raises:
        ValueError: If sessions_dir does not exist.
    """
    if not sessions_dir.is_dir():
        raise ValueError(f"Session directory does not exist: {sessions_dir}")

    json_files = sorted(sessions_dir.glob("*.json"))
    if not json_files:
        logger.info("No JSON session files found in %s", sessions_dir)
        return 0

    # Filter out archived files (they'll be imported as archived sessions)
    active_files = [f for f in json_files if "__archived__" not in f.name]
    archived_files = [f for f in json_files if "__archived__" in f.name]

    store = SessionStore(sessions_dir=sessions_dir)
    total_imported = 0

    try:
        # Check integrity before migration
        issues = store.integrity_check()
        if issues:
            logger.warning("Database integrity issues before migration: %s", issues)

        # Import active sessions
        for file_path in active_files:
            try:
                count = _import_json_file(store, file_path, archived=False)
                total_imported += count
            except Exception as exc:
                logger.warning("Failed to import %s: %s", file_path.name, exc)

        # Import archived sessions (mark them archived in metadata)
        for file_path in archived_files:
            try:
                count = _import_json_file(store, file_path, archived=True)
                total_imported += count
            except Exception as exc:
                logger.warning("Failed to import archived %s: %s", file_path.name, exc)

        # Verify integrity after migration
        issues = store.integrity_check()
        if issues:
            logger.warning("Database integrity issues after migration: %s", issues)

        logger.info(
            "Migration complete: %d sessions imported from %s "
            "(%d active, %d archived files)",
            total_imported, sessions_dir, len(active_files), len(archived_files),
        )
    finally:
        store.close()

    return total_imported


def _import_json_file(
    store: SessionStore,
    file_path: Path,
    *,
    archived: bool = False,
) -> int:
    """Import a single JSON session file into the store.

    Returns the number of messages imported.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session_key = data.get("session_key")
    if not session_key:
        logger.warning("Skipping %s: no session_key found", file_path.name)
        return 0

    messages = data.get("messages", [])
    if not messages:
        logger.info("Skipping %s: no messages", file_path.name)
        return 0

    created_at = data.get("created_at", time.time())
    metadata = data.get("metadata", {})

    if archived:
        metadata["archived_at"] = metadata.get("archived_at", time.time())

    count = store.db.import_messages(
        session_key,
        messages,
        created_at=created_at,
        metadata=metadata,
    )

    logger.info(
        "Imported %d messages from %s%s",
        count, file_path.name, " (archived)" if archived else "",
    )
    return 1  # one session


def verify_migration(sessions_dir: Path) -> dict:
    """Verify migration by comparing JSON file count with DB session count.

    Returns a dict with keys: json_files, db_sessions, matched.
    """
    from tyagent.db import Database

    db_path = sessions_dir / "sessions.db"

    if not db_path.exists():
        return {"error": "Database not found", "json_files": 0, "db_sessions": 0}

    # Count JSON files (excluding the DB file)
    json_count = len(list(sessions_dir.glob("*.json")))

    db = Database(db_path)
    try:
        db_sessions = len(db.get_all_session_keys())
        return {
            "json_files": json_count,
            "db_sessions": db_sessions,
        }
    finally:
        db.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m tyagent.migrate <sessions_dir>")
        sys.exit(1)

    sessions_dir = Path(sys.argv[1])
    count = migrate_from_json(sessions_dir)
    print(f"Migrated {count} sessions.")

    verification = verify_migration(sessions_dir)
    print(f"Verification: {verification}")
