"""A request context records the conversation turn it served."""
from __future__ import annotations

import sqlite3

from virtual_context.storage.sqlite import SQLiteStore


def _context(conv: str, turn: int) -> dict:
    return {
        "conversation_id": conv, "request_turn": 0, "turn": turn,
        "timestamp": "2026-10-01T21:33:48+00:00", "user_message": "My bad.",
        "inbound_tags": [], "retrieval_method": "rrf", "candidates_found": 0,
        "candidates_selected": 0, "segments_injected": [], "facts_injected": [],
        "facts_count": 0, "facts_tags": [], "pool_used": 0, "pool_budget": 0,
        "total_context_tokens": 0, "non_virtualizable_floor": 0, "tool_call_count": 0,
    }


def test_saved_context_carries_the_conversation_turn(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    request_turn = store.save_request_context(_context("c", 3715))
    [loaded] = store.load_request_contexts("c")
    assert loaded["turn"] == 3715
    assert loaded["request_turn"] == request_turn


def test_a_store_created_before_the_column_gains_it(tmp_path):
    path = tmp_path / "vc.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE request_context DROP COLUMN turn")
    conn.commit()
    conn.close()
    store = SQLiteStore(path)
    store.save_request_context(_context("c", 12))
    assert store.load_request_contexts("c")[0]["turn"] == 12
