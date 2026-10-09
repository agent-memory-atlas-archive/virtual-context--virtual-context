"""Regrouping writes only the turns whose group changed."""
from __future__ import annotations

import pytest

from tests.test_context_hint_prewarm import _make_engine
from tests.test_noop_ingest_skips_whole_conversation_passes import _body, _ingest


@pytest.mark.regression("BUG-124")
def test_appending_a_turn_updates_only_rows_whose_group_changed(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        store = engine._ingest_reconciler._store
        conv = engine.config.conversation_id
        _ingest(engine, _body(40))
        sqlite_store = store._segments if hasattr(store, "_segments") else store
        conn = sqlite_store._get_conn()
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        try:
            changed = store.recompute_canonical_turn_groups(conv)
        finally:
            conn.set_trace_callback(None)
        assert changed == 0
        assert not [s for s in statements if s.lstrip().upper().startswith("UPDATE CANONICAL_TURNS")]
        groups = [r.turn_group_number for r in store.get_all_canonical_turns(conv)]
        assert groups == sorted(groups) and len(set(groups)) == 40
    finally:
        engine.close()
