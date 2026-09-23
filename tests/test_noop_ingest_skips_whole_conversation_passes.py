"""An ingest that writes no rows skips the whole-conversation group and anchor passes.

Every call of a tool loop resends the same history. Those calls write nothing,
so recomputing turn groups and refreshing anchors across the whole stored
conversation (about 1.7 s on a 7,000-row conversation) bought nothing.
"""
from __future__ import annotations

import pytest

from tests.test_context_hint_prewarm import _make_engine
from virtual_context.proxy.formats import detect_format


def _ingest(engine, body):
    return engine._ingest_reconciler.ingest_batch(
        engine.config.conversation_id, body=body, fmt=detect_format(body),
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )


def _body(n):
    msgs = []
    for i in range(n):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    return {"messages": msgs}


@pytest.mark.regression("PROXY-032")
def test_resend_skips_regroup_and_anchor_refresh_but_new_rows_do_not(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)
    try:
        store = engine._ingest_reconciler._store
        calls = {"regroup": 0, "anchors": 0}
        real_regroup = store.recompute_canonical_turn_groups
        real_anchors = engine._ingest_reconciler._refresh_persisted_anchors
        monkeypatch.setattr(store, "recompute_canonical_turn_groups",
                            lambda *a, **k: (calls.__setitem__("regroup", calls["regroup"] + 1), real_regroup(*a, **k))[1])
        monkeypatch.setattr(engine._ingest_reconciler, "_refresh_persisted_anchors",
                            lambda *a, **k: (calls.__setitem__("anchors", calls["anchors"] + 1), real_anchors(*a, **k))[1])

        first = _ingest(engine, _body(3))
        assert first.turns_written > 0 and calls == {"regroup": 1, "anchors": 1}

        again = _ingest(engine, _body(3))
        assert again.turns_written == 0
        assert calls == {"regroup": 1, "anchors": 1}, "a no-op resend must not rerun the passes"

        more = _ingest(engine, _body(4))
        assert more.turns_written > 0 and calls == {"regroup": 2, "anchors": 2}
        groups = [r.turn_group_number for r in store.get_all_canonical_turns(engine.config.conversation_id)]
        assert groups == sorted(groups) and len(set(groups)) == 4
    finally:
        engine.close()
