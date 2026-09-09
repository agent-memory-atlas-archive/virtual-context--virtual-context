"""Honored finite agreements remain useful without becoming permanent (BUG-075)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from virtual_context.core.community.actor_card_admission import ActorCardAdmissionService
from virtual_context.core.community.actor_card_curation import ActorCardCurationService
from virtual_context.core.community.actor_card_policy import _ACTOR_CARD_POLICY_VERSION


class CaptureProvider:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        return json.dumps(self.response), {}


def _candidate(**bounds):
    return {
        "kind": "communication_pref",
        "body": "Wants the agent to use a playful title until October 15, 2032.",
        "confidence": 0.7,
        "fact_ids": [],
        "turn_ids": ["synthetic-request"],
        **bounds,
    }


def _prompt(surface, candidate=None, decision=None):
    candidate = _candidate() if candidate is None else candidate
    config = SimpleNamespace(
        assembler=SimpleNamespace(actor_card_entries_per_kind=3),
        compactor=SimpleNamespace(max_summary_tokens=1200),
    )
    compactor = SimpleNamespace(_parse_response=json.loads)
    turns = [
        SimpleNamespace(turn=SimpleNamespace(canonical_turn_id="synthetic-request")),
    ]

    def prompt_turns(*args, **kwargs):
        return [
            {
                "id": "synthetic-request",
                "content": (
                    "From 2032-10-01T00:00:00Z until 2032-10-16T00:00:00Z, "
                    "please use that playful title. Deal?"
                ),
                "agent_reply": "Deal. I will use that title through the agreed period.",
                "truncated": False,
            }
        ]

    if surface == "curation":
        provider = CaptureProvider(
            {
                "substantive": True,
                "coverage_reason": "substantive",
                "entries": [candidate],
            }
        )
        service = ActorCardCurationService(
            config=config,
            compactor=compactor,
            prompt_turns=prompt_turns,
            curation_provider=lambda: provider,
            provider_for_model=lambda _: provider,
        )
        result = service.curate_partition([], turns)
    else:
        provider = CaptureProvider(
            {
                "substantive": True,
                "coverage_reason": "substantive",
                "decisions": [
                    decision
                    or {"candidate_id": "synthetic-entry", "admit": True, "reason": "durable"}
                ],
            }
        )
        service = ActorCardAdmissionService(
            config=config,
            compactor=compactor,
            admission_provider=lambda: provider,
            evidence_segments=lambda *args, **kwargs: ([], set()),
            prompt_turns=prompt_turns,
        )
        entry = SimpleNamespace(
            id="synthetic-entry",
            kind=candidate["kind"],
            body=candidate["body"],
            confidence=candidate["confidence"],
            **{key: candidate[key] for key in ("valid_from", "expires_at") if key in candidate},
        )
        sources = [SimpleNamespace(fact_id="", canonical_turn_id="synthetic-request")]
        result = service.admit_entries(
            "actor:synthetic:one",
            "synthetic-audience",
            [],
            turns,
            [(entry, sources)],
            curator_substantive=True,
        )
    return provider.requests[0], result


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("surface", ["curation", "admission"])
def test_honored_finite_agreement_contract_reaches_both_model_surfaces(surface):
    request, _ = _prompt(surface)
    contract = request["system"]
    assert "explicitly honored ongoing agreement" in contract
    assert "valid_from" in contract and "expires_at" in contract
    assert "ingestion timestamps" in contract
    assert "supplied attested occurred_at" in contract
    assert "initial honored agreement" in contract
    assert "starting date is unknown" in contract
    assert "relevant_history" in contract
    assert "Reject temporary, test/probe, one-turn" not in contract
    assert "Do not promote temporary, test-only" not in contract


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("surface", ["curation", "admission"])
def test_resolved_relationship_history_survives_without_reclassifying_a_goal(surface):
    request, _ = _prompt(surface)
    contract = request["system"]
    assert "resolved relationship history" in contract
    assert "playful register alone" in contract
    assert "completed outcome is not an active_goal" in contract
    assert "Kinds are exclusive per claim, not per source event" in contract
    assert "retain the distinct acknowledged outcome" in contract
    assert "external-resource action" in contract
    assert "safety_posture_request" in contract or "safety posture" in contract
    assert "without a visible honored signal" in contract or "no visible honored signal" in contract


@pytest.mark.regression("BUG-075")
def test_independent_admission_receives_exact_immutable_validity_bounds():
    candidate = _candidate(
        valid_from="2032-10-01T00:00:00Z",
        expires_at="2032-10-16T00:00:00Z",
    )
    request, result = _prompt("admission", candidate)
    admitted = json.loads(request["user"])["candidates"][0]
    assert admitted["valid_from"] == candidate["valid_from"]
    assert admitted["expires_at"] == candidate["expires_at"]
    assert admitted["body"] == candidate["body"]
    assert result[0][0][0].body == candidate["body"]


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("surface", ["curation", "admission"])
def test_current_utc_time_is_supplied_separately_from_source_evidence(surface):
    before = datetime.now(timezone.utc)
    request, _ = _prompt(surface)
    after = datetime.now(timezone.utc)
    payload = json.loads(request["user"])
    as_of = datetime.fromisoformat(payload["as_of"].replace("Z", "+00:00"))
    assert as_of.utcoffset().total_seconds() == 0
    assert before <= as_of <= after


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("bounds", [{}, {"expires_at": "2032-10-16T00:00:00Z"}])
def test_curator_preserves_legacy_five_fields_and_optional_bound_output(bounds):
    candidate = _candidate(**bounds)
    _, result = _prompt("curation", candidate)
    assert result[3] == [candidate]


@pytest.mark.regression("BUG-075")
def test_commitment_policy_invalidates_previous_policy_cache():
    assert _ACTOR_CARD_POLICY_VERSION >= 19


@pytest.mark.regression("BUG-075")
def test_expired_rejection_is_a_valid_independent_admission_decision():
    request, result = _prompt(
        "admission",
        decision={
            "candidate_id": "synthetic-entry",
            "admit": False,
            "reason": "expired",
        },
    )
    assert result[0] == []
    assert '"expired"' in request["system"]


@pytest.mark.regression("BUG-075")
def test_curator_must_not_propose_a_preference_that_already_expired():
    request, _ = _prompt("curation")
    assert (
        "Do not propose a communication_pref whose expires_at is at or before as_of"
        in request["system"]
    )


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("surface", ["curation", "admission"])
def test_source_occurrence_does_not_date_an_event_narrated_in_the_message(surface):
    request, _ = _prompt(surface)
    contract = request["system"]
    assert "occurred_at dates the source message, not every event it narrates" in contract
    assert "do not assign that date to a narrated earlier outcome" in contract


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("surface", ["curation", "admission"])
def test_narrower_kind_exclusion_applies_to_claim_not_entire_source_event(surface):
    request, _ = _prompt(surface)
    contract = request["system"]
    assert "when no narrower goal or communication preference applies" not in contract
    assert "when no narrower durable preference or goal is justified" not in contract
    assert "The four kinds are mutually exclusive" not in contract
    assert "Choose the narrowest kind for each distinct claim" in contract
    assert "An explanatory mention inside an expiring preference" in contract


@pytest.mark.regression("BUG-075")
def test_curator_receives_separate_kind_budgets_and_their_total_ceiling():
    request, _ = _prompt("curation")
    limits = json.loads(request["user"])["limits"]
    assert limits["by_kind"] == {
        "communication_pref": 3,
        "active_goal": 3,
        "relevant_history": 3,
        "interaction_style": 3,
    }
    assert limits["max_total_entries"] == 12
    assert "entries_per_kind" not in limits
    assert "not a target to fill" in request["system"]


@pytest.mark.regression("BUG-075")
@pytest.mark.parametrize("surface", ["curation", "admission"])
def test_paired_example_separates_resolved_history_from_finite_preference(surface):
    request, _ = _prompt(surface)
    contract = request["system"]
    assert "Illustrative classification, not source evidence" in contract
    assert "two distinct claims" in contract
    assert "a relevant_history claim that the actor won" in contract
    assert "a communication_pref claim for the agreed form of address" in contract
