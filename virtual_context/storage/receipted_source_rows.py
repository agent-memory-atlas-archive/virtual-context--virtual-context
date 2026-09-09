"""Bounded maintenance reads for canonical rows protected by audit receipts."""

from __future__ import annotations

_RECEIPT_READ_CHUNK_SIZE = 500


def get_receipted_canonical_turn_ids(
    conn,
    conversation_id: str,
    canonical_turn_ids: list[str],
    *,
    dialect: str,
) -> set[str]:
    """Return only selected physical rows under the exact owner with receipts.

    Receipt presence protects evidence even if its proof later fails validation.
    This is a skip list for ordinary maintenance, never repair authorization.
    Schema installation and journal settings are deliberately left untouched.
    """
    if dialect not in {"sqlite", "postgres"}:
        raise ValueError("unsupported receipt lookup dialect")
    if not conversation_id or not canonical_turn_ids:
        return set()
    ids = list(dict.fromkeys(canonical_turn_ids))
    placeholder = "?" if dialect == "sqlite" else "%s"
    protected = set()
    for offset in range(0, len(ids), _RECEIPT_READ_CHUNK_SIZE):
        chunk = ids[offset : offset + _RECEIPT_READ_CHUNK_SIZE]
        slots = ",".join(placeholder for _ in chunk)
        rows = conn.execute(
            f"""SELECT c.canonical_turn_id FROM canonical_turns c
                WHERE c.conversation_id = {placeholder}
                  AND c.canonical_turn_id IN ({slots})
                  AND (
                    EXISTS (SELECT 1 FROM canonical_audience_reassignments ar
                             WHERE ar.canonical_turn_id = c.canonical_turn_id)
                    OR EXISTS (SELECT 1 FROM canonical_assistant_channel_enrichments ce
                                WHERE ce.assistant_canonical_turn_id = c.canonical_turn_id)
                  )""",
            (conversation_id, *chunk),
        ).fetchall()
        protected.update(str(row["canonical_turn_id"]) for row in rows)
    return protected
