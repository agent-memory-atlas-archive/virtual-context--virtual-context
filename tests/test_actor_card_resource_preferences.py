"""BUG-071: external actions are not standing response preferences.

Classification remains the independent semantic judge's responsibility. These
tests pin its shared contract and exercise rejection through the real rebuild
path; fixed provider decisions do not measure a model's classification accuracy.
All actors, sources, and requested resources below are synthetic.
"""

import json
import hashlib

import pytest

from tests.test_actor_cards import (
    _admission,
    _card_pipeline,
    _conversation,
    _curation,
    _entry,
    _now,
    _turn,
    _turn_source,
    store as store,
)
from virtual_context.core.community.actor_card_policy import _ACTOR_CARD_POLICY_VERSION
from virtual_context.core.community.actor_card_policy import _EmptyResponseFallbackProvider
from virtual_context.types import CARD_KIND_COMMUNICATION_PREF, CARD_SCOPE_CROSS_CONTEXT


ACTOR = "actor:test:resource-member"
ACTION_BODY = (
    "Wants the agent to rename a workspace role Navigator, use a compass icon, "
    "and color its label violet."
)
RESPONSE_BODY = (
    "Wants the agent to address them as Captain and explain access roles in "
    "plain language in future replies."
)


class _Curator:
    def __init__(self, entries):
        self.entries = entries
        self.system = ""
        self.turns = []

    def complete(self, **kwargs):
        self.system = kwargs["system"]
        self.turns = json.loads(kwargs["user"])["turns"]
        return json.dumps(_curation(self.entries)), {}


class _Judge:
    def __init__(self):
        self.system = ""
        self.candidates = []
        self.turns = []

    def complete(self, **kwargs):
        self.system = kwargs["system"]
        payload = json.loads(kwargs["user"])
        self.candidates = payload["candidates"]
        self.turns = payload["actor_turns"]
        return json.dumps(_admission([
            {
                "candidate_id": candidate["candidate_id"],
                "admit": candidate["body"] != ACTION_BODY,
                "reason": "wrong_kind" if candidate["body"] == ACTION_BODY else "durable",
            }
            for candidate in self.candidates
        ])), {}


def _seed(store):
    _conversation(store, "shared-space")
    _conversation(store, "private-space")
    for group, (turn_id, request, reply) in enumerate((
        (
            "action",
            "Rename my workspace role Navigator, use a compass icon, and make its label violet.",
            "Done; I updated the role's name, icon, and color.",
        ),
        (
            "response",
            "For future replies, address me as Captain and explain access roles in plain language.",
            "Understood, Captain. I will keep those explanations in plain language.",
        ),
    )):
        _turn(store, turn_id, "shared-space", ACTOR, "shared-space", "source-space", content=request)
        store._get_conn().execute(
            "UPDATE canonical_turns SET assistant_content=?, turn_group_number=? "
            "WHERE canonical_turn_id=?",
            (reply, group, turn_id),
        )
    store.upsert_actor_profile_from_turn("shared-space", ACTOR, "Member", seen_at=_now())


def _proposal(body, turn_id):
    return {
        "kind": CARD_KIND_COMMUNICATION_PREF,
        "body": body,
        "confidence": 0.7,
        "fact_ids": [],
        "turn_ids": [turn_id],
    }


@pytest.mark.regression("BUG-071")
def test_both_prompt_surfaces_distinguish_response_preferences_from_external_actions(store):
    _seed(store)
    curator = _Curator([_proposal(ACTION_BODY, "action"), _proposal(RESPONSE_BODY, "response")])
    judge = _Judge()
    _card_pipeline(store, curator, admission=judge)._rebuild_actor_card(ACTOR)

    for system in (curator.system, judge.system):
        assert "External-resource actions, one-off or repeated, are not communication_pref" in system
        assert "a persistent result is not a persistent response preference" in system
        assert "addressing the person in conversation" in system
        assert "changing a stored name, display label, role, color, or icon" in system
        assert "Honoring an external action does not establish a communication preference" in system
        assert "A stored setting that itself governs how the agent replies" in system
    assert "wrong_kind" in judge.system
    assert "wrong_kind" not in curator.system
    assert "Reject an immutable candidate with insufficient_evidence" not in curator.system
    assert "agent_refused" not in curator.system
    assert "safety_posture_request" not in curator.system
    for turns in (curator.turns, judge.turns):
        action = next(turn for turn in turns if turn["id"] == "action")
        assert action["agent_reply"] == "Done; I updated the role's name, icon, and color."
    assert _ACTOR_CARD_POLICY_VERSION >= 18


@pytest.mark.regression("BUG-071")
@pytest.mark.parametrize("existing_action", [False, True], ids=["fresh", "carryover"])
def test_rebuild_rejects_customization_but_keeps_genuine_address_and_response_preference(
    store, existing_action,
):
    _seed(store)
    entries = [_proposal(RESPONSE_BODY, "response")]
    if existing_action:
        old = _entry(
            "old-action", CARD_KIND_COMMUNICATION_PREF, ACTION_BODY,
            scope=CARD_SCOPE_CROSS_CONTEXT, confidence=0.7,
        )
        assert store.replace_actor_card(
            "t1", ACTOR,
            [(old, [_turn_source(old.id, "shared-space", "shared-space", "action", "source-space")])],
            input_hash="older-policy", expected_source_epochs={"shared-space": 1},
        )
    else:
        entries.insert(0, _proposal(ACTION_BODY, "action"))
    curator, judge = _Curator(entries), _Judge()
    assert _card_pipeline(store, curator, admission=judge)._rebuild_actor_card(ACTOR) == 1
    action = next(candidate for candidate in judge.candidates if candidate["body"] == ACTION_BODY)
    assert action["origin"] == ("existing" if existing_action else "fresh")
    assert action["turn_ids"] == ["action"]
    assert action["body"] == ACTION_BODY
    assert "External-resource actions, one-off or repeated, are not communication_pref" in judge.system
    card = store.get_actor_card(
        "t1", ACTOR, owner_conversation_id="private-space",
        audience_conversation_id="private-space",
    )
    assert card is not None
    assert [(entry.kind, entry.body) for entry in card.entries] == [
        (CARD_KIND_COMMUNICATION_PREF, RESPONSE_BODY),
    ]
    status = store.get_actor_card_rebuild_status("t1", ACTOR)
    assert status["rejected_counts"] == {"semantic_wrong_kind": 1}
    assert status["failure_count"] == 0


@pytest.mark.regression("BUG-071")
@pytest.mark.parametrize("gate", ["primary", "empty_fallback", "coverage_adjudicator"])
def test_action_only_substantive_actor_can_reject_every_candidate_without_failure(store, gate):
    _seed(store)
    store._get_conn().execute("DELETE FROM canonical_turns WHERE canonical_turn_id='response'")
    curator = _Curator([_proposal(ACTION_BODY, "action")])
    prompts = []

    class RejectAll:
        def __init__(self, substantive):
            self.substantive = substantive

        def complete(self, **kwargs):
            prompts.append(kwargs["system"])
            assert "never admit a candidate to satisfy coverage" in kwargs["system"]
            assert "must finish with at least one admitted card entry" not in kwargs["system"]
            candidates = json.loads(kwargs["user"])["candidates"]
            return json.dumps(_admission([
                {"candidate_id": item["candidate_id"], "admit": False, "reason": "wrong_kind"}
                for item in candidates
            ], substantive=self.substantive)), {}

    class Empty:
        def complete(self, **kwargs):
            return "", {}

    judge = RejectAll(True)
    if gate != "primary":
        judge = _EmptyResponseFallbackProvider(
            Empty() if gate == "empty_fallback" else RejectAll(False),
            judge, primary_model="synthetic-primary", fallback_model="synthetic-fallback",
        )
    assert _card_pipeline(store, curator, admission=judge)._rebuild_actor_card(ACTOR) == 0
    assert len(prompts) == (2 if gate == "coverage_adjudicator" else 1)
    status = store.get_actor_card_rebuild_status("t1", ACTOR)
    assert status["outcome"] == "no_durable_entries"
    assert status["failure_count"] == 0
    profile = store.get_actor_profile("t1", ACTOR)
    assert not profile.card_dirty and not profile.card_invalid
    assert store.get_actor_card(
        "t1", ACTOR, owner_conversation_id="private-space",
        audience_conversation_id="private-space",
    ) is None


@pytest.mark.regression("BUG-071")
def test_curator_can_classify_action_only_evidence_as_no_durable_context(store):
    _seed(store)
    store._get_conn().execute("DELETE FROM canonical_turns WHERE canonical_turn_id='response'")

    class EmptyCurator:
        def complete(self, **kwargs):
            assert "One-shot service or external-resource requests alone are no_durable_context" in kwargs["system"]
            return json.dumps(_curation([])), {}

    judge = _Judge()
    assert _card_pipeline(store, EmptyCurator(), admission=judge)._rebuild_actor_card(ACTOR) == 0
    assert "One-shot service or external-resource requests alone are no_durable_context" in judge.system
    status = store.get_actor_card_rebuild_status("t1", ACTOR)
    assert status["outcome"] == "clean_empty"
    assert status["failure_count"] == 0


@pytest.mark.regression("BUG-071")
def test_reproposed_carryover_keeps_existing_origin_and_is_rejected(store):
    _seed(store)
    digest = hashlib.sha256(json.dumps(
        [ACTOR, CARD_KIND_COMMUNICATION_PREF, ACTION_BODY, [], ["action"]], separators=(",", ":"),
    ).encode()).hexdigest()[:32]
    old = _entry(
        f"card-{digest}", CARD_KIND_COMMUNICATION_PREF, ACTION_BODY,
        scope=CARD_SCOPE_CROSS_CONTEXT, confidence=0.7,
    )
    assert store.replace_actor_card(
        "t1", ACTOR,
        [(old, [_turn_source(old.id, "shared-space", "shared-space", "action", "source-space")])],
        input_hash="older-policy", expected_source_epochs={"shared-space": 1},
    )
    curator = _Curator([_proposal(ACTION_BODY, "action"), _proposal(RESPONSE_BODY, "response")])
    judge = _Judge()
    assert _card_pipeline(store, curator, admission=judge)._rebuild_actor_card(ACTOR) == 1
    candidates = [candidate for candidate in judge.candidates if candidate["body"] == ACTION_BODY]
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == old.id
    assert candidates[0]["origin"] == "existing"
    assert store.get_actor_card_rebuild_status("t1", ACTOR)["rejected_counts"] == {"semantic_wrong_kind": 1}


@pytest.mark.regression("BUG-071")
def test_actor_subject_action_keeps_wrong_subject_priority(store):
    _seed(store)
    body = "Renames workspace roles and changes their icons and colors."

    class SubjectJudge:
        def complete(self, **kwargs):
            assert "a role error is wrong_subject even when the kind is also wrong" in kwargs["system"]
            candidates = json.loads(kwargs["user"])["candidates"]
            return json.dumps(_admission([
                {"candidate_id": item["candidate_id"], "admit": item["body"] != body,
                 "reason": "wrong_subject" if item["body"] == body else "durable"}
                for item in candidates
            ])), {}

    curator = _Curator([_proposal(body, "action"), _proposal(RESPONSE_BODY, "response")])
    assert _card_pipeline(store, curator, admission=SubjectJudge())._rebuild_actor_card(ACTOR) == 1
    assert store.get_actor_card_rebuild_status("t1", ACTOR)["rejected_counts"] == {"semantic_wrong_subject": 1}


@pytest.mark.regression("BUG-071")
def test_repeated_resource_actions_do_not_exclude_stored_reply_preferences(store):
    _seed(store)
    _turn(
        store, "action-repeat", "shared-space", ACTOR, "shared-space", "source-space",
        content="Set my workspace role back to Navigator with the compass icon and violet label.",
    )
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET turn_group_number=2, assistant_content=? WHERE canonical_turn_id=?",
        ("Done; the role settings are restored.", "action-repeat"),
    )
    conn.execute(
        "UPDATE canonical_turns SET user_content=?, assistant_content=? WHERE canonical_turn_id=?",
        (
            "Save French as my default reply language for future conversations.",
            "Saved. I will reply in French in future conversations.", "response",
        ),
    )
    action = _proposal(ACTION_BODY, "action")
    action["turn_ids"].append("action-repeat")
    body = "Wants future replies in French, saved as their default reply-language setting."
    curator, judge = _Curator([action, _proposal(body, "response")]), _Judge()
    assert _card_pipeline(store, curator, admission=judge)._rebuild_actor_card(ACTOR) == 1
    rejected = next(item for item in judge.candidates if item["body"] == ACTION_BODY)
    assert rejected["turn_ids"] == ["action", "action-repeat"]
    assert "A stored setting that itself governs how the agent replies" in judge.system
    card = store.get_actor_card(
        "t1", ACTOR, owner_conversation_id="private-space",
        audience_conversation_id="private-space",
    )
    assert [entry.body for entry in card.entries] == [body]
