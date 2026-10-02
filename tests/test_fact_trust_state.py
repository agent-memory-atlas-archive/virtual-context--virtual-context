"""A fact carries whether the record still supports it, and a retracted fact is never served.

After an admin edit changes a source turn, the facts extracted from the
segments holding it are retracted in the same transaction and withheld from
every fact read until the rebuild replaces them. Rebuilt facts are verified
when their source turns re-prove them.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.fact_lifecycle import fact_version
from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.ingest import supersession
from virtual_context.types import (
    CompactionResult, Fact, SegmentMetadata, StoredSegment, TagSummary,
)

CONV = "conv-trust"
TURN = "00000000-0000-0000-0000-000000000001"


class _Compactor:
    model_name = "stub"
    fail = False

    def compact(self, segments, **_kwargs):
        if self.fail:
            raise RuntimeError("provider down")
        out = []
        for seg in segments:
            text = " | ".join(m.content for m in seg.messages)
            out.append(CompactionResult(
                segment_id=seg.id, primary_tag=seg.primary_tag, tags=list(seg.tags),
                summary=text, summary_tokens=5, full_text=text, original_tokens=10,
                messages=[{"role": m.role, "content": m.content} for m in seg.messages],
                metadata=SegmentMetadata(), compression_ratio=0.5,
                facts=[Fact(subject="user", verb="lives in", object=text, what=text, tags=[seg.primary_tag])],
            ))
        return out

    def compact_tag_summaries(self, *, cover_tags, max_turn, **_kwargs):
        return [TagSummary(tag=t, summary=f"topic {t}", covers_through_turn=max_turn) for t in cover_tags]


@pytest.fixture
def engine(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": CONV,
    }))
    engine._compactor = engine._compaction._compactor = _Compactor()
    engine._store.save_canonical_turn(
        CONV, 0, "I live in Boston", "noted", canonical_turn_id=TURN, turn_group_number=0,
        sort_key=1.0, primary_tag="home", tags=["home"],
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-home", conversation_id=CONV, primary_tag="home", tags=["home"], summary="Boston",
        metadata=SegmentMetadata(canonical_turn_ids=[TURN], source_mapping_complete=True),
    ))
    engine._store.store_facts([Fact(id="boston", subject="user", verb="lives in", object="Boston",
                                    what="lives in Boston", segment_ref="seg-home",
                                    conversation_id=CONV, tags=["home"])])
    return engine


def _stored(engine):
    return {f.what: f.trust_state for f in engine._store.query_facts(conversation_id=CONV, limit=20)}


def _served(engine):
    out = json.loads(execute_vc_tool(engine, "vc_query_facts", {"subject": "user"}))
    return {f["what"] for f in out["facts"]}


def test_new_facts_are_unverified_until_their_sources_prove_them(engine):
    assert _stored(engine) == {"lives in Boston": "unverified"}
    assert engine._store.refresh_fact_trust(CONV, ["seg-home"]) == {"verified": 1, "unverified": 0, "rejected": 0}
    assert _stored(engine) == {"lives in Boston": "verified"}


def test_an_edit_retracts_the_facts_it_affects_until_the_rebuild(engine):
    assert _served(engine) == {"lives in Boston"}
    report = engine.record_editor.edit_turn(TURN, user_content="I live in Denver", actor="admin")

    # Retracted in the edit's transaction: still stored, no longer served.
    assert _stored(engine) == {"lives in Boston": "retracted"}
    assert _served(engine) == set()
    facts = engine._store.query_facts(conversation_id=CONV, limit=20)
    assert engine._retrieval._facts_for_request(facts, None) == []

    engine.record_editor.process(report["operation_id"])

    assert _stored(engine) == {"I live in Denver | noted": "verified"}
    assert _served(engine) == {"I live in Denver | noted"}


def test_a_failed_rebuild_keeps_the_facts_withheld(engine):
    engine._compactor.fail = True
    report = engine.record_editor.edit_turn(TURN, user_content="I live in Denver", actor="admin")
    op = engine.record_editor.process(report["operation_id"])
    assert op["status"] == "failed"
    assert _stored(engine) == {"lives in Boston": "retracted"}
    assert _served(engine) == set()


def test_supersession_does_not_consider_retracted_facts(engine):
    engine.record_editor.edit_turn(TURN, user_content="I live in Denver", actor="admin")
    retracted = engine._store.query_facts(conversation_id=CONV, limit=20)[0]
    new = Fact(id="new", subject="user", verb="lives in", object="Denver", conversation_id=CONV)
    engine._store.store_facts([new])
    _snapshot, candidates, _ = supersession._admitted_snapshots(engine._store, new, [retracted])
    assert candidates == []


def test_trust_state_is_not_part_of_the_fact_version(engine):
    fact = engine._store.query_facts(conversation_id=CONV, limit=20)[0]
    before = fact_version(fact)
    fact.trust_state = "verified"
    assert fact_version(fact) == before
