"""BUG-075: time-bounded response agreements survive the full card pipeline.

All sources are synthetic. Fixed provider outputs exercise normalization,
independent admission routing, persistence and serving, not model accuracy.
"""

import json

import pytest

from tests.test_actor_cards import (
    _admission, _card_pipeline, _conversation, _curation, _now, _turn,
    store as store,
)


ACTOR = "actor:test:agreement-member"
START = "2020-01-01T00:00:00.000000+00:00"
END = "2099-01-15T00:00:00.000000+00:00"
BODY = "Wants the agent to address them as Navigator until January 15, 2099."


def _seed(store):
    _conversation(store, "community")
    _conversation(store, "private")
    _turn(store, "agreement", "community", ACTOR, "community", "lounge",
          content="You agreed to call me Navigator until January 15, 2099.")
    store._get_conn().execute(
        "UPDATE canonical_turns SET assistant_content=?, turn_group_number=0 "
        "WHERE canonical_turn_id='agreement'",
        ("Agreed, Navigator. That address ends on January 15, 2099.",),
    )
    store.upsert_actor_profile_from_turn("community", ACTOR, "Member", seen_at=_now())


def _proposal(**updates):
    return {
        "kind": "communication_pref", "body": BODY, "confidence": 0.7,
        "fact_ids": [], "turn_ids": ["agreement"],
        "valid_from": START, "expires_at": END, **updates,
    }


class _Curator:
    def __init__(self, entries):
        self.entries = entries

    def complete(self, **kwargs):
        return json.dumps(_curation(self.entries)), {}


class _Judge:
    def __init__(self):
        self.candidates = []

    def complete(self, **kwargs):
        self.candidates = json.loads(kwargs["user"])["candidates"]
        return json.dumps(_admission([
            {"candidate_id": entry["candidate_id"], "admit": True, "reason": "durable"}
            for entry in self.candidates
        ], substantive=True)), {}


def _read(store, audience="community"):
    return store.get_actor_card("t1", ACTOR, owner_conversation_id=audience,
                                audience_conversation_id=audience,
                                audience_channel_id="another-channel")


@pytest.mark.regression("BUG-075")
def test_timed_preference_reaches_admission_and_stays_in_source_audience(store):
    _seed(store)
    judge = _Judge()
    pipeline = _card_pipeline(store, _Curator([_proposal()]), admission=judge)
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    assert judge.candidates[0]["valid_from"] == START
    assert judge.candidates[0]["expires_at"] == END
    card = _read(store)
    assert card is not None
    assert [(e.body, e.valid_from, e.expires_at) for e in card.entries] == [(BODY, START, END)]
    assert card.entries[0].audience_scope == "same_conversation"
    assert _read(store, "private") is None


@pytest.mark.regression("BUG-075")
def test_timed_preference_carryover_keeps_original_expiry_when_curator_omits_it(store):
    _seed(store)
    curator, judge = _Curator([_proposal()]), _Judge()
    pipeline = _card_pipeline(store, curator, admission=judge)
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    original = _read(store).entries[0]
    curator.entries = [{"kind": "relevant_history", "body": "Discussed a navigation challenge.",
                        "confidence": 0.7, "fact_ids": [], "turn_ids": ["agreement"]}]
    assert pipeline._rebuild_actor_card(ACTOR, force=True) == 2
    retained = next(e for e in _read(store).entries if e.kind == "communication_pref")
    assert (retained.id, retained.valid_from, retained.expires_at) == (original.id, START, END)
    candidate = next(c for c in judge.candidates if c["kind"] == "communication_pref")
    assert candidate["origin"] == "existing"


@pytest.mark.regression("BUG-075")
def test_corrected_validity_creates_new_immutable_candidate_identity(store):
    _seed(store)
    curator, judge = _Curator([_proposal()]), _Judge()
    pipeline = _card_pipeline(store, curator, admission=judge)
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    old_id = _read(store).entries[0].id
    curator.entries = [_proposal(valid_from="2020-01-02T00:00:00+00:00")]
    pipeline._rebuild_actor_card(ACTOR, force=True)
    fresh = next(c for c in judge.candidates if c["origin"] == "fresh")
    assert fresh["candidate_id"] != old_id


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("changes", [
    {"expires_at": "2099-01-15"},
    {"valid_from": "2100-01-01T00:00:00Z"},
    {"expires_at": 1},
    {"kind": "relevant_history"},
])
def test_malformed_or_wrong_kind_validity_never_reaches_admission(store, changes):
    _seed(store)
    judge = _Judge()
    pipeline = _card_pipeline(store, _Curator([_proposal(**changes)]), admission=judge)
    expected = (
        "no valid entries array"
        if changes.get("kind") == "relevant_history"
        else "rejected every model entry"
    )
    with pytest.raises(RuntimeError, match=expected):
        pipeline._rebuild_actor_card(ACTOR)
    assert judge.candidates == []
    assert _read(store) is None
