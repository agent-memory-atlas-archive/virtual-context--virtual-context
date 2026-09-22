"""A finished pair is stored even when its own tagging raises a compaction signal."""
from __future__ import annotations

from tests.test_engine_lifecycle_epoch import _make_test_engine, _inner_store
from virtual_context.types import Message


def test_pair_is_persisted_when_compaction_is_signalled(tmp_path):
    engine = _make_test_engine(tmp_path, conversation_id="persist-under-signal")
    engine.config.monitor.context_window = 200
    history = [Message(role="user", content="please remember the blue door"),
               Message(role="assistant", content="noted: the blue door")]
    signal = engine.tag_turn(history, payload_tokens=5000)
    assert signal is not None, "the tiny window must signal compaction"
    store = _inner_store(engine)
    assert store.count_canonical_turns("persist-under-signal") > 0
    rows = store.get_canonical_turn_reconcile_rows("persist-under-signal") or []
    assert rows, "the finished pair must be in canonical storage"


def test_pair_is_persisted_without_a_signal_too(tmp_path):
    engine = _make_test_engine(tmp_path, conversation_id="persist-no-signal")
    history = [Message(role="user", content="hello there"), Message(role="assistant", content="hi")]
    assert engine.tag_turn(history, payload_tokens=10) is None
    assert _inner_store(engine).count_canonical_turns("persist-no-signal") > 0
