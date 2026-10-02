"""Delete guard shared by the record ledgers.

A ledger row records what was decided about a conversation's memory. It is
kept for as long as the conversation exists and leaves with it:
``delete_conversation`` marks the conversation deleted in
``conversation_lifecycle`` before it removes any row, and that mark is the
only thing that lets a ledger row be deleted.
"""

from __future__ import annotations

_MESSAGE = "rows are kept while their conversation exists"


def ensure_ledger_delete_guard(conn, dialect: str, table: str) -> None:
    """Refuse DELETE on *table* unless its conversation is marked deleted."""
    trigger = f"guard_{table}_delete"
    if dialect == "postgres":
        conn.execute(f"""CREATE OR REPLACE FUNCTION guard_ledger_delete()
            RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM conversation_lifecycle
                WHERE conversation_id = OLD.conversation_id AND deleted) THEN
                RAISE EXCEPTION '% {_MESSAGE}', TG_TABLE_NAME;
            END IF;
            RETURN OLD;
            END $$""")
        if not conn.execute(
            "SELECT 1 FROM pg_trigger WHERE tgrelid=%s::regclass AND tgname=%s",
            (table, trigger),
        ).fetchone():
            conn.execute(f"""CREATE TRIGGER {trigger}
                BEFORE DELETE ON {table} FOR EACH ROW
                EXECUTE FUNCTION guard_ledger_delete()""")
        return
    if conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='trigger' AND name=?", (trigger,)
    ).fetchone():
        return
    conn.execute(f"""CREATE TRIGGER {trigger}
        BEFORE DELETE ON {table}
        WHEN NOT EXISTS (SELECT 1 FROM conversation_lifecycle
            WHERE conversation_id = OLD.conversation_id AND deleted)
        BEGIN SELECT RAISE(ABORT, '{table} {_MESSAGE}'); END""")
