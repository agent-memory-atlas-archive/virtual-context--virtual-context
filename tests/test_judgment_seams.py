import json

import httpx
import pytest
from virtual_context.core.retriever import ContextRetriever
from virtual_context.types import RetrieverConfig
from virtual_context.core.structured_summary import (
    is_safety_critical_personal_evidence,
    is_safety_critical_personal_evidence_legacy,
)

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime
from virtual_context.core.quote_search import _detect_query_intent
from virtual_context.types import JudgmentConfig


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _runtime(mode, answers_fn):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"model": "jev-t", "answers": answers_fn(body),
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    rt = build_runtime(JudgmentConfig(mode=mode), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    return rt, seen


def _choice(key, choice, confidence, options):
    return {key: {"type": "choice", "choice": choice, "confidence": confidence,
                  "probabilities": {o: (confidence if o == choice else (1 - confidence) / (len(options) - 1)) for o in options}}}


def test_intent_legacy_mode_is_unchanged():
    assert _detect_query_intent("what are we doing currently") == "current_state"
    assert _detect_query_intent("find quote about magnesium") == "default"


def test_intent_jev_mode_uses_choice_when_confident():
    rt, seen = _runtime("jev", lambda b: _choice("intent", "current_state", 0.9, ["current_state", "default"]))
    with judgment.override(rt):
        assert _detect_query_intent("find quote about magnesium") == "current_state"
    assert seen[0]["state"] == {"query": "find quote about magnesium"}
    assert set(seen[0]["questions"]["intent"]["criteria"]) == {"current_state", "default"}


def test_intent_jev_mode_low_confidence_falls_back_to_regex():
    rt, _ = _runtime("jev", lambda b: _choice("intent", "current_state", 0.4, ["current_state", "default"]))
    with judgment.override(rt):
        assert _detect_query_intent("find quote about magnesium") == "default"


def test_intent_shadow_mode_returns_regex_answer_and_calls_jev():
    rt, seen = _runtime("shadow", lambda b: _choice("intent", "current_state", 0.9, ["current_state", "default"]))
    with judgment.override(rt):
        assert _detect_query_intent("find quote about magnesium") == "default"
    assert len(seen) == 1


def _noul(key, p):
    return {key: {"type": "noul", "noul": p}}


def _retriever():
    return ContextRetriever(tag_generator=None, store=None, config=RetrieverConfig(), inbound_tagger=None)


def test_temporal_legacy_matches_pattern_list():
    r = _retriever()
    assert r._detect_temporal("what was the very first thing we discussed") is True
    assert r._detect_temporal("how much protein should I eat") is False


def test_temporal_jev_mode_uses_noul_threshold():
    rt, seen = _runtime("jev", lambda b: _noul("temporal", 0.8))
    with judgment.override(rt):
        assert _retriever()._detect_temporal("how much protein should I eat") is True
    assert seen[0]["state"] == {"message": "how much protein should I eat"}
    rt2, _ = _runtime("jev", lambda b: _noul("temporal", 0.2))
    with judgment.override(rt2):
        assert _retriever()._detect_temporal("what was the very first thing we discussed") is False


def test_temporal_shadow_mode_keeps_legacy():
    rt, seen = _runtime("shadow", lambda b: _noul("temporal", 0.8))
    with judgment.override(rt):
        assert _retriever()._detect_temporal("how much protein should I eat") is False
    assert len(seen) == 1


def test_safety_legacy_wrapper_matches_original():
    for text in ["I stopped taking metformin last week.", "Correction: I am not on statins.",
                 "he stopped by the store", "", "We talked about cars."]:
        assert is_safety_critical_personal_evidence(text) == is_safety_critical_personal_evidence_legacy(text)


def test_safety_empty_text_never_calls_jev():
    rt, seen = _runtime("jev", lambda b: _noul("safety_critical", 0.9))
    with judgment.override(rt):
        assert is_safety_critical_personal_evidence("   ") is False
    assert seen == []


def test_safety_jev_mode_thresholds_noul():
    rt, seen = _runtime("jev", lambda b: _noul("safety_critical", 0.7))
    with judgment.override(rt):
        assert is_safety_critical_personal_evidence("We talked about cars.") is True
    assert seen[0]["state"] == {"text": "We talked about cars."}
    rt2, _ = _runtime("jev", lambda b: _noul("safety_critical", 0.1))
    with judgment.override(rt2):
        assert is_safety_critical_personal_evidence("I stopped taking metformin last week.") is False


def test_safety_jev_failure_falls_back_to_regex():
    def boom(request):
        return httpx.Response(503, text="down")
    http = httpx.Client(transport=httpx.MockTransport(boom))
    rt = build_runtime(JudgmentConfig(mode="jev"), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    with judgment.override(rt):
        assert is_safety_critical_personal_evidence("I stopped taking metformin last week.") == \
            is_safety_critical_personal_evidence_legacy("I stopped taking metformin last week.")
