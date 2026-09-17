import json

import httpx
import pytest

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
