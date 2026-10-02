"""A fact's event date holds only a date extraction produced.

``when_date`` is when the thing happened; ``session_date`` is when it was said.
A fact whose extraction yields no date keeps an empty ``when_date``, and the
readers that need some date fall back to the session date in the query.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from virtual_context.core.compactor import DomainCompactor
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CompactorConfig, Fact, Message, TaggedSegment

SESSION = "2023/04/20 (Thu) 04:17"


def _compactor(facts):
    llm = MagicMock()
    llm.complete.return_value = (json.dumps({
        "summary": "s", "entities": [], "key_decisions": [], "action_items": [],
        "date_references": [], "refined_tags": ["travel"], "related_tags": [], "facts": facts,
    }), {})
    return DomainCompactor(llm_provider=llm, config=CompactorConfig())


def _segment():
    return TaggedSegment(id="seg", primary_tag="travel", tags=["travel"],
                         messages=[Message(role="user", content="trip talk")], session_date=SESSION)


def _fact(what, when=""):
    return {"subject": "user", "verb": "visited", "object": "Lisbon", "status": "completed",
            "fact_type": "personal", "what": what, "who": "", "when": when, "where": "", "why": ""}


def test_an_undated_fact_keeps_an_empty_event_date():
    fact = _compactor([_fact("User visited Lisbon.")]).compact([_segment()])[0].facts[0]
    assert fact.when_date == ""
    assert fact.session_date == SESSION


def test_a_relative_date_is_resolved_against_the_session():
    fact = _compactor([_fact("User visited Lisbon yesterday.")]).compact([_segment()])[0].facts[0]
    assert fact.when_date == "2023-04-19"


def test_an_extracted_date_is_kept():
    fact = _compactor([_fact("User visited Lisbon.", "2022/06/01")]).compact([_segment()])[0].facts[0]
    assert fact.when_date == "2022/06/01"


def test_lane_extraction_keeps_an_empty_event_date():
    compactor = _compactor([_fact("User visited Lisbon.")])
    facts = compactor._extract_facts_for_text("trip talk", segment=_segment(), role="user")
    assert [f.when_date for f in facts] == [""]


def test_the_date_window_finds_an_undated_fact_by_its_session_date(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    store.store_facts([
        Fact(id="undated", subject="user", verb="visited", object="Lisbon", conversation_id="c",
             session_date="2023-04-20"),
        Fact(id="dated", subject="user", verb="visited", object="Porto", conversation_id="c",
             when_date="2022-06-01", session_date="2023-04-20"),
    ])
    found = store.query_experience_facts_by_date("2023-04-01", "2023-04-30", conversation_id="c")
    assert [f.id for f in found] == ["undated"]
