import json

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.community import actor_card_admission as adm
from virtual_context.core.community.actor_card_admission import (
    ADMISSION_COVERAGE_REASONS, ADMISSION_REASONS, ADMISSION_SYSTEM_PROMPT,
    build_admission_request, parse_admission_response,
)
from virtual_context.core.judgment import COVERAGE_CRITERIA, REASON_CRITERIA, build_runtime, judge_admission
from virtual_context.core.llm_utils import parse_llm_json
from virtual_context.types import JudgmentConfig


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


CANDS = [
    {"candidate_id": "e1", "origin": "fresh", "kind": "communication_pref", "body": "Wants to be called broski.",
     "proposed_confidence": 0.9, "fact_ids": [], "turn_ids": ["t1"], "source_segments": []},
    {"candidate_id": "e2", "origin": "fresh", "kind": "relevant_history", "body": "Ran a marathon in 2024.",
     "proposed_confidence": 0.8, "fact_ids": ["f1"], "turn_ids": [], "source_segments": []},
]


def test_reason_sets_match_validator_and_criteria_cover_them():
    assert ADMISSION_COVERAGE_REASONS == ("substantive", "greeting_only", "one_off_trivia", "bot_meta_or_test",
                                          "no_durable_context", "insufficient_evidence")
    assert ADMISSION_REASONS[0] == "durable" and len(ADMISSION_REASONS) == 17
    assert set(COVERAGE_CRITERIA) == set(ADMISSION_COVERAGE_REASONS)
    assert set(REASON_CRITERIA) == set(ADMISSION_REASONS)
    assert ADMISSION_SYSTEM_PROMPT.endswith(adm._ACTOR_CARD_ADMISSION_REJECTION_RULES)


def test_build_admission_request_shape():
    req = build_admission_request(candidates=CANDS, compact_facts=[], actor_turns=[], evidence_segments=[],
                                  curator_substantive=True, as_of="2026-09-16T00:00:00+00:00")
    assert req["system"] == ADMISSION_SYSTEM_PROMPT
    assert req["max_tokens"] == 800
    payload = json.loads(req["user"])
    assert payload == req["payload"]
    assert list(payload) == ["as_of", "curator_substantive_claim", "candidates", "facts", "actor_turns", "evidence_segments"]
    assert payload["as_of"] == "2026-09-16T00:00:00+00:00" and payload["curator_substantive_claim"] is True


def test_parse_admission_response_accepts_valid_and_rejects_invalid():
    ok = json.dumps({"substantive": True, "coverage_reason": "substantive",
                     "decisions": [{"candidate_id": "e1", "admit": False, "reason": "insufficient_evidence"},
                                   {"candidate_id": "e2", "admit": True, "reason": "durable"}]})
    substantive, decisions = parse_admission_response(ok, parse_json=parse_llm_json, eligible=["e1", "e2"])
    assert substantive is True and decisions["e2"]["admit"] is True
    bad = json.dumps({"substantive": True, "coverage_reason": "substantive",
                      "decisions": [{"candidate_id": "e1", "admit": True, "reason": "insufficient_evidence"}]})
    with pytest.raises(adm._ActorCardAdmissionError):
        parse_admission_response(bad, parse_json=parse_llm_json, eligible=["e1", "e2"])
    with pytest.raises(adm._ActorCardAdmissionError):
        parse_admission_response("nope", parse_json=parse_llm_json, eligible=["e1"])


def _runtime(mode, coverage, reasons):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        answers = {"coverage": {"type": "choice", "choice": coverage, "confidence": 0.9,
                                "probabilities": {coverage: 0.9}}}
        for key in body["questions"]:
            if key.startswith("reason__"):
                cid = key[len("reason__"):]
                answers[key] = {"type": "choice", "choice": reasons[cid], "confidence": 0.8,
                                "probabilities": {reasons[cid]: 0.8}}
        return httpx.Response(200, json={"model": "jev-t", "answers": answers,
                                         "usage": {"input_tokens": 3, "output_tokens": 1}})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(JudgmentConfig(mode=mode), environ={"TYPESAFE_API_KEY": "k"}, http_client=http), seen


def test_judge_admission_legacy_returns_none():
    req = build_admission_request(candidates=CANDS, compact_facts=[], actor_turns=[], evidence_segments=[], curator_substantive=True)
    assert judge_admission(req["payload"], ["e1", "e2"]) is None


def test_judge_admission_jev_mode_produces_validator_compatible_text():
    rt, seen = _runtime("jev", "substantive", {"e1": "insufficient_evidence", "e2": "durable"})
    req = build_admission_request(candidates=CANDS, compact_facts=[], actor_turns=[], evidence_segments=[], curator_substantive=True)
    with judgment.override(rt):
        out = judge_admission(req["payload"], ["e1", "e2"])
    assert out is not None
    substantive, decisions = parse_admission_response(out.text, parse_json=parse_llm_json, eligible=["e1", "e2"])
    assert substantive is True and decisions["e1"]["admit"] is False and decisions["e2"]["admit"] is True
    body = seen[0]
    assert body["state"] == req["payload"]
    assert set(body["questions"]) == {"coverage", "reason__e1", "reason__e2"}
    assert set(body["questions"]["reason__e1"]["criteria"]) == set(ADMISSION_REASONS)
    assert "candidate_id 'e1'" in body["questions"]["reason__e1"]["instructions"]


def test_judge_admission_shadow_mode_also_returns_judgment_for_comparison():
    rt, _ = _runtime("shadow", "greeting_only", {"e1": "not_durable", "e2": "durable"})
    req = build_admission_request(candidates=CANDS, compact_facts=[], actor_turns=[], evidence_segments=[], curator_substantive=False)
    with judgment.override(rt):
        out = judge_admission(req["payload"], ["e1", "e2"])
    assert out is not None and out.substantive is False and out.decisions["e1"]["reason"] == "not_durable"


def test_judge_admission_failure_returns_none():
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    rt = build_runtime(JudgmentConfig(mode="jev"), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    req = build_admission_request(candidates=CANDS, compact_facts=[], actor_turns=[], evidence_segments=[], curator_substantive=True)
    with judgment.override(rt):
        assert judge_admission(req["payload"], ["e1", "e2"]) is None
