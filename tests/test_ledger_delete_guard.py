"""Ledger rows stay while their conversation exists and go with it."""
from __future__ import annotations

import sqlite3

import pytest

from virtual_context.storage.sqlite import SQLiteStore

CONV = "conv-ledger"
LEDGERS = ("fact_decisions", "turn_edits", "fact_verdicts", "record_edit_operations")


def _seed(store):
    with store._get_conn() as conn:
        conn.execute(
            """INSERT INTO fact_decisions (decision_id, conversation_id, fact_id,
                replacement_fact_id, action, accepted, reason, observed_at, event_date,
                policy_version, proposal_json, before_json, after_json, source_versions_json)
               VALUES ('d1', ?, 'f1', '', 'supersede', 0, 'stale', 't', '', 'v1', '{}', '{}', '{}', '{}')""",
            (CONV,),
        )
        conn.execute(
            """INSERT INTO turn_edits (edit_id, conversation_id, canonical_turn_id, action, actor,
                reason, operation_id, created_at, before_user_content, before_assistant_content,
                before_tags_json, before_turn_hash)
               VALUES ('e1', ?, 't1', 'edit', 'admin', '', 'op1', 't', 'u', 'a', '[]', 'h')""",
            (CONV,),
        )
        conn.execute(
            """INSERT INTO fact_verdicts (verdict_id, conversation_id, fact_key, fact_id, verdict,
                actor, reason, created_at, fact_json, source_turn_ids_json)
               VALUES ('v1', ?, 'k', 'f1', 'rejected', 'admin', '', 't', '{}', '[]')""",
            (CONV,),
        )
        conn.execute(
            """INSERT INTO record_edit_operations (operation_id, conversation_id, action, target,
                actor, status, segments_json, created_at, updated_at)
               VALUES ('op1', ?, 'edit_turn', 't1', 'admin', 'completed', '[]', 't', 't')""",
            (CONV,),
        )


@pytest.mark.parametrize("table", LEDGERS)
def test_a_ledger_row_cannot_be_deleted_while_its_conversation_exists(tmp_path, table):
    store = SQLiteStore(tmp_path / "vc.db")
    _seed(store)
    with pytest.raises(sqlite3.DatabaseError, match="kept while their conversation exists"):
        with store._get_conn() as conn:
            conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (CONV,))


def test_deleting_the_conversation_removes_its_ledgers(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    _seed(store)
    store.delete_conversation(CONV)
    with store._get_conn() as conn:
        for table in LEDGERS:
            count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE conversation_id = ?", (CONV,)).fetchone()[0]
            assert count == 0, table
