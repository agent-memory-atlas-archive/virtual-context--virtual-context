"""Facts are shown only to the audience their source turns may be shown to.

The audience check read source turn ids off each item; summaries carry them
and facts do not, so every fact passed it and a fact distilled from a DM inside
an owner conversation reached a guild request. A fact's source turns are its
segment's, so a fact is now admitted exactly when its segment would be.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.summary_identity import facts_for_audience
from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import Fact, SegmentMetadata, SpeakerRetrievalContext, StoredSegment

OWNER = "sk:agent:demo:discord:guild:1"
DM = "sk:agent:demo:discord:direct:9"
GUILD_TURN = "00000000-0000-0000-0000-00000000000a"
DM_TURN = "00000000-0000-0000-0000-00000000000b"


@pytest.fixture
def engine(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": OWNER,
    }))
    for n, (turn, audience, text, ref) in enumerate((
        (GUILD_TURN, OWNER, "I run the guild league on Fridays", "seg-guild"),
        (DM_TURN, DM, "privately: I am interviewing at another company", "seg-dm"),
    )):
        engine._store.save_canonical_turn(
            OWNER, n, text, "noted", canonical_turn_id=turn, turn_group_number=n,
            sort_key=float(n + 1), primary_tag="life", tags=["life"],
            audience_conversation_id=audience, audience_attribution_version=1,
        )
        engine._store.store_segment(StoredSegment(
            ref=ref, conversation_id=OWNER, primary_tag="life", tags=["life"], summary=text,
            metadata=SegmentMetadata(canonical_turn_ids=[turn], source_mapping_complete=True),
        ))
        engine._store.store_facts([Fact(
            subject="user", verb="said", object=text, what=text, segment_ref=ref,
            conversation_id=OWNER, tags=["life"],
        )])
    return engine


def _context(audience: str) -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="t1", owner_conversation_id=OWNER, audience_conversation_id=audience,
    )


def _query_facts(engine, audience):
    out = json.loads(execute_vc_tool(
        engine, "vc_query_facts", {"subject": "user"}, speaker_context=_context(audience),
    ))
    return {fact["what"] for fact in out["facts"]}


@pytest.mark.regression("BUG-112")
def test_vc_query_facts_withholds_a_dm_fact_from_a_guild_request(engine):
    assert _query_facts(engine, OWNER) == {"I run the guild league on Fridays"}
    # The withheld fact is stored and served to its own audience.
    assert "privately: I am interviewing at another company" in _query_facts(engine, DM)


@pytest.mark.regression("BUG-112")
def test_a_dm_request_does_not_see_guild_facts(engine):
    assert _query_facts(engine, DM) == {"privately: I am interviewing at another company"}


@pytest.mark.regression("BUG-112")
def test_injected_facts_are_filtered_before_curation(engine):
    facts = engine._store.query_facts(conversation_id=OWNER, limit=10)
    assert len(facts) == 2
    kept = engine._retrieval._facts_for_request(facts, None)
    assert [f.what for f in kept] == ["I run the guild league on Fridays"]


@pytest.mark.regression("BUG-112")
def test_a_fact_whose_segment_cannot_be_read_is_withheld(engine):
    class Broken:
        def get_segment(self, *_a, **_k):
            raise RuntimeError("down")

    fact = Fact(subject="user", verb="said", object="x", segment_ref="seg-guild", conversation_id=OWNER)
    assert facts_for_audience([fact], store=Broken(), conversation_id=OWNER, speaker_context=None) == []


@pytest.mark.regression("BUG-112")
def test_remember_when_fact_hits_follow_their_segment(engine):
    hits = [{"type": "fact", "what": "guild", "segment_ref": "seg-guild"},
            {"type": "fact", "what": "dm", "segment_ref": "seg-dm"}]
    kept, _ = engine._temporal._scope_fact_results_for_request(hits, speaker_context=_context(OWNER))
    assert [h["what"] for h in kept] == ["guild"]
    kept, _ = engine._temporal._scope_fact_results_for_request(hits, speaker_context=_context(DM))
    assert [h["what"] for h in kept] == ["dm"]
