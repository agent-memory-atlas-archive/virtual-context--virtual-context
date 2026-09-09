"""Malformed curator entries cannot silently replace a complete card (BUG-075)."""

import json
from dataclasses import asdict

import pytest

from tests.test_actor_card_commitment_rebuild import (
    ACTOR, _Curator, _Judge, _proposal, _read, _seed,
)
from tests.test_actor_cards import _card_pipeline, _curation, store as store


def _history():
    return {
        "kind": "relevant_history",
        "body": "The agent acknowledged their agreement about a form of address.",
        "confidence": 0.7,
        "fact_ids": [],
        "turn_ids": ["agreement"],
    }


class _FallbackCurator:
    def __init__(self, fallback):
        self.fallback = fallback
        self.fallback_calls = 0

    def complete_with_source(self, **kwargs):
        return json.dumps(_curation([
            _proposal(), {**_history(), "turn_ids_note": None},
        ])), {}, "primary"

    def complete_fallback(self, **kwargs):
        self.fallback_calls += 1
        return json.dumps(_curation(self.fallback)), {}


@pytest.mark.regression("BUG-075")
def test_malformed_history_triggers_fallback_before_any_candidate_is_admitted(store):
    _seed(store)
    curator = _FallbackCurator([_proposal(), _history()])
    judge = _Judge()
    pipeline = _card_pipeline(store, curator, admission=judge)

    assert pipeline._rebuild_actor_card(ACTOR) == 2
    assert curator.fallback_calls == 1
    assert {entry.kind for entry in _read(store).entries} == {
        "communication_pref", "relevant_history",
    }
    assert len(judge.candidates) == 2
    assert all("turn_ids_note" not in candidate for candidate in judge.candidates)


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("with_fallback", [False, True])
def test_malformed_curation_without_valid_fallback_preserves_previous_card(store, with_fallback):
    _seed(store)
    initial = _card_pipeline(store, _Curator([_proposal(), _history()]), admission=_Judge())
    assert initial._rebuild_actor_card(ACTOR) == 2
    before = asdict(_read(store))
    malformed = [_proposal(), {**_history(), "turn_ids_note": None}]
    curator = _FallbackCurator(malformed) if with_fallback else _Curator(malformed)
    judge = _Judge()
    pipeline = _card_pipeline(store, curator, admission=judge)

    with pytest.raises(RuntimeError, match="no valid entries array"):
        pipeline._rebuild_actor_card(ACTOR, force=True)

    assert asdict(_read(store)) == before
    assert judge.candidates == []
    assert store.get_actor_card_rebuild_status("t1", ACTOR)["outcome"] == "invalid_response"
    if with_fallback:
        assert curator.fallback_calls == 1


@pytest.mark.regression("BUG-075")
def test_grouped_curation_keeps_history_and_preference_as_distinct_claims(store):
    _seed(store)
    groups = {"relevant_history": [_history()], "communication_pref": [_proposal()],
              "active_goal": [], "interaction_style": []}
    pipeline = _card_pipeline(store, _GroupedCurator(groups), admission=_Judge())
    assert pipeline._rebuild_actor_card(ACTOR) == 2
    assert {entry.kind for entry in _read(store).entries} == {"communication_pref", "relevant_history"}


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("defect", ["missing_group", "wrong_group"])
def test_grouped_curation_rejects_incomplete_or_mislabeled_groups(store, defect):
    _seed(store)
    groups = {"relevant_history": [_history()], "communication_pref": [_proposal()],
              "active_goal": [], "interaction_style": []}
    if defect == "missing_group":
        del groups["active_goal"]
    else:
        groups["active_goal"] = groups.pop("relevant_history")
        groups["relevant_history"] = []
    pipeline = _card_pipeline(store, _GroupedCurator(groups), admission=_Judge())
    with pytest.raises(RuntimeError, match="no valid entries array"):
        pipeline._rebuild_actor_card(ACTOR)


class _GroupedCurator:
    def __init__(self, groups):
        self.groups = groups

    def complete(self, **kwargs):
        return json.dumps({"substantive": True, "coverage_reason": "substantive", "entries": self.groups}), {}
