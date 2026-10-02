"""The newest stored turn's time is read from the conversation's own rows."""
from __future__ import annotations

from virtual_context.storage.sqlite import SQLiteStore


def _insert(store, conv: str, turn_id: str, created_at: str, sort_key: float) -> None:
    with store._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, 1, 'u','a','u','a', ?, 'b', ?, ?, 1, NULL, ?, ?)
            """,
            (turn_id, conv, f"h_{turn_id}", sort_key, created_at, created_at, created_at, created_at),
        )


def test_newest_turn_time_is_the_latest_created_row(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    _insert(store, "a", "t1", "2026-09-01T10:00:00+00:00", 1000.0)
    _insert(store, "a", "t2", "2026-10-01T09:00:00+00:00", 2000.0)
    _insert(store, "b", "t3", "2026-10-02T00:00:00+00:00", 1000.0)
    assert store.newest_canonical_turn_at("a") == "2026-10-01T09:00:00+00:00"


def test_a_conversation_without_turns_has_no_newest_turn(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    assert store.newest_canonical_turn_at("empty") is None
