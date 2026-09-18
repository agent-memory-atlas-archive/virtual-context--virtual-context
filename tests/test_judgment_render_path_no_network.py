"""Rendering stored summaries must never call the judgment model, whatever the mode.

The safety-critical seam runs where claims are built (compaction). Re-validating
stored claims for display happens on every prepare, so it stays on the
deterministic predicate. Regression for the 2026-09-17 prepare-latency incident.
"""
import json

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime
from virtual_context.core.structured_summary import is_safety_critical_personal_evidence
from virtual_context.core.summary_identity import _validated_structured_claims
from virtual_context.types import JudgmentConfig


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _counting_runtime(mode="shadow"):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "jev-t", "answers": {"safety_critical": {"type": "noul", "noul": 0.9}},
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(JudgmentConfig(mode=mode), environ={"TYPESAFE_API_KEY": "k"}, http_client=http), calls


def test_wrapper_calls_the_model_in_shadow_mode_when_asked_directly():
    rt, calls = _counting_runtime()
    with judgment.override(rt):
        is_safety_critical_personal_evidence("I stopped taking metformin.", runtime=rt)
    assert len(calls) == 1


def test_claim_validation_makes_no_model_calls_even_in_shadow_mode(monkeypatch):
    rt, calls = _counting_runtime()
    seen = []
    import virtual_context.core.summary_identity as si
    monkeypatch.setattr(si, "is_safety_critical_personal_evidence_legacy", lambda text: seen.append(text) or True)
    # Module registry AND explicit runtime both in shadow: the render path must ignore both.
    with judgment.override(rt):
        try:
            _validated_structured_claims(object(), rows={}, conversation_id="c", speaker_context=None, colliding_labels=frozenset())
        except Exception:
            pass  # a bare object is not a valid summary; the point is the call accounting below
    assert calls == []
    import inspect
    src = inspect.getsource(_validated_structured_claims)
    assert "is_safety_critical_personal_evidence(" not in src
    assert "is_safety_critical_personal_evidence_legacy(" in src
