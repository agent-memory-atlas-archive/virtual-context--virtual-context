"""An admin can reject a fact, and the rejection holds across rebuilds.

A rejected fact is withheld from every read. The verdict is keyed on the
fact's statement and the turns it came from, and a later extraction of the same
statement from the same turns is not stored; a restore lifts the rejection and
brings the fact back. Every verdict is kept in an audit table a trigger keeps
from being changed.
"""

from __future__ import annotations

import json
import os
import sqlite3

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import Fact, SegmentMetadata, StoredSegment

CONV = "conv-verdicts"
TURN = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def engine(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": CONV,
    }))
    engine._store.save_canonical_turn(
        CONV, 0, "I moved to Denver", "noted", canonical_turn_id=TURN, turn_group_number=0,
        sort_key=1.0, primary_tag="home", tags=["home"],
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-home", conversation_id=CONV, primary_tag="home", tags=["home"], summary="Denver",
        metadata=SegmentMetadata(canonical_turn_ids=[TURN], source_mapping_complete=True),
    ))
    _store_fact(engine, "f1", "lives in Boston")
    _store_fact(engine, "f2", "moved to Denver")
    return engine


def _store_fact(engine, fact_id, statement):
    verb, obj = statement.split(" ", 1)[0], statement.split(" ", 1)[1]
    engine._store.store_facts([Fact(id=fact_id, subject="user", verb=verb, object=obj, what=statement,
                                    segment_ref="seg-home", conversation_id=CONV, tags=["home"])])


def _served(engine):
    out = json.loads(execute_vc_tool(engine, "vc_query_facts", {"subject": "user"}))
    return {f["what"] for f in out["facts"]}


def _states(engine):
    return {f.what: f.trust_state for f in engine._store.query_facts(conversation_id=CONV, limit=20)}


def test_a_rejected_fact_is_stored_but_never_served(engine):
    assert _served(engine) == {"lives in Boston", "moved to Denver"}
    engine.record_editor.reject_fact("f1", actor="admin", reason="wrong city")
    assert _states(engine)["lives in Boston"] == "rejected"
    assert _served(engine) == {"moved to Denver"}
    facts = engine._store.query_facts(conversation_id=CONV, limit=20)
    assert {f.what for f in engine._retrieval._facts_for_request(facts, None)} == {"moved to Denver"}
    verdict = engine._store.get_fact_verdicts(CONV)[0]
    assert (verdict["verdict"], verdict["actor"], verdict["reason"]) == ("rejected", "admin", "wrong city")


def _extract_again(engine):
    return engine._store.replace_facts_for_segment(CONV, "seg-home", [
        Fact(id="f1-again", subject="User", verb="lives", object="in Boston", what="lives in Boston",
             segment_ref="seg-home", conversation_id=CONV, tags=["home"]),
        Fact(id="f2-again", subject="user", verb="moved", object="to Denver", what="moved to Denver",
             segment_ref="seg-home", conversation_id=CONV, tags=["home"]),
    ])


def test_the_same_statement_extracted_again_is_not_stored(engine):
    engine.record_editor.reject_fact("f1", actor="admin")
    assert _extract_again(engine) == (2, 1)
    engine._store.refresh_fact_trust(CONV, ["seg-home"])
    assert _states(engine) == {"moved to Denver": "verified"}
    assert _served(engine) == {"moved to Denver"}


def test_the_same_statement_from_other_turns_is_judged_again(engine):
    other = "00000000-0000-0000-0000-000000000002"
    engine._store.save_canonical_turn(
        CONV, 1, "Still in Boston", "ok", canonical_turn_id=other, turn_group_number=1,
        sort_key=2.0, primary_tag="home", tags=["home"],
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-later", conversation_id=CONV, primary_tag="home", tags=["home"], summary="Boston",
        metadata=SegmentMetadata(canonical_turn_ids=[other], source_mapping_complete=True),
    ))
    engine.record_editor.reject_fact("f1", actor="admin")
    engine._store.replace_facts_for_segment(CONV, "seg-later", [
        Fact(id="f3", subject="user", verb="lives", object="in Boston", what="lives in Boston",
             segment_ref="seg-later", conversation_id=CONV, tags=["home"]),
    ])
    assert [f.id for f in engine._store.get_facts_by_segment("seg-later")] == ["f3"]


def test_a_restore_brings_back_a_fact_whose_row_was_replaced(engine):
    engine.record_editor.reject_fact("f1", actor="admin")
    _extract_again(engine)
    engine.record_editor.restore_fact("f1", actor="admin", reason="it was right")
    assert _served(engine) == {"lives in Boston", "moved to Denver"}


def test_a_restore_lifts_the_rejection(engine):
    engine.record_editor.reject_fact("f1", actor="admin")
    engine.record_editor.restore_fact("f1", actor="admin", reason="it was right")
    assert _states(engine)["lives in Boston"] == "verified"
    assert _served(engine) == {"lives in Boston", "moved to Denver"}
    assert [v["verdict"] for v in engine._store.get_fact_verdicts(CONV)] == ["restored", "rejected"]


def test_verdicts_cannot_be_changed(engine):
    engine.record_editor.reject_fact("f1", actor="admin")
    conn = engine._store._store._segments._get_conn() if hasattr(engine._store, "_store") else None
    conn = conn or engine._store._get_conn()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        conn.execute("UPDATE fact_verdicts SET verdict='restored'")
    conn.rollback()
