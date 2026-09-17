import json
import logging

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import (
    JevClient, JevOutcome, JudgmentMode, build_runtime, choice_q, decide, noul_q,
)
from virtual_context.types import JudgmentConfig


def _client(handler, **cfg):
    config = JudgmentConfig(**cfg)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return JevClient(config, http_client=http, environ={"TYPESAFE_API_KEY": "k"})


def _ok_handler(request):
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer k"
    assert body["model"] == "jev-latest"
    answers = {}
    for key, q in body["questions"].items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.9}
        else:
            first = sorted(q["criteria"])[0]
            answers[key] = {"type": "choice", "choice": first, "confidence": 0.8,
                            "probabilities": {k: (0.8 if k == first else 0.2) for k in q["criteria"]}}
    return httpx.Response(200, json={"model": "jev-1", "answers": answers,
                                     "usage": {"input_tokens": 10, "output_tokens": 2}})


def test_mode_parse_and_env_resolution():
    assert JudgmentMode.parse("jev") is JudgmentMode.JEV
    assert JudgmentMode.resolve("shadow", environ={}) is JudgmentMode.SHADOW
    assert JudgmentMode.resolve("legacy", environ={"VC_JUDGMENT_MODE": "jev"}) is JudgmentMode.JEV
    with pytest.raises(ValueError):
        JudgmentMode.resolve("legacy", environ={"VC_JUDGMENT_MODE": "sometimes"})
    with pytest.raises(ValueError):
        JudgmentMode.parse("")


def test_client_parses_answers_and_usage():
    client = _client(_ok_handler)
    resp = client.ask(seam="t", state={"x": 1},
                      questions={"a": noul_q("is it?"), "b": choice_q("which?", {"p": "P", "q": "Q"})})
    assert resp.answers["a"].kind == "noul" and resp.answers["a"].value == 0.9
    assert resp.answers["b"].kind == "choice" and resp.answers["b"].value == "p"
    assert resp.answers["b"].probabilities == {"p": 0.8, "q": 0.2}
    assert resp.answers["b"].confidence == 0.8
    assert (resp.input_tokens, resp.output_tokens, resp.model) == (10, 2, "jev-1")
    assert resp.latency_ms >= 0


def _raise_connect(request):
    raise httpx.ConnectError("down")


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="boom"),
    lambda r: httpx.Response(200, text="not json"),
    _raise_connect,
])
def test_client_returns_none_on_failures(handler, caplog):
    client = _client(handler)
    with caplog.at_level(logging.WARNING):
        assert client.ask(seam="t", state="s", questions={"a": noul_q("?")}) is None
    assert any("JUDGMENT_JEV_ERROR seam=t" in rec.message for rec in caplog.records)


def test_client_without_api_key_returns_none(caplog):
    http = httpx.Client(transport=httpx.MockTransport(_ok_handler))
    client = JevClient(JudgmentConfig(), http_client=http, environ={})
    with caplog.at_level(logging.WARNING):
        assert client.ask(seam="t", state="s", questions={"a": noul_q("?")}) is None
    assert any("error=missing_api_key" in rec.message for rec in caplog.records)


def test_build_runtime_legacy_has_no_client_and_is_disabled():
    rt = build_runtime(JudgmentConfig(), environ={})
    assert rt.mode is JudgmentMode.LEGACY and rt.client is None and rt.enabled is False


def test_build_runtime_env_pin_wins_over_yaml():
    rt = build_runtime(JudgmentConfig(mode="legacy"), environ={"VC_JUDGMENT_MODE": "jev", "TYPESAFE_API_KEY": "k"})
    assert rt.mode is JudgmentMode.JEV and rt.client is not None and rt.enabled


def test_install_current_override_reset():
    judgment.reset()
    assert judgment.current().mode is JudgmentMode.LEGACY
    rt = build_runtime(JudgmentConfig(mode="shadow"), environ={"TYPESAFE_API_KEY": "k"})
    judgment.install(rt)
    assert judgment.current() is rt
    legacy = build_runtime(JudgmentConfig(), environ={})
    with judgment.override(legacy):
        assert judgment.current() is legacy
    assert judgment.current() is rt
    judgment.reset()
    assert judgment.current().mode is JudgmentMode.LEGACY


def _rt(mode, handler=_ok_handler):
    config = JudgmentConfig(mode=mode)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(config, environ={"TYPESAFE_API_KEY": "k"}, http_client=http)


def test_decide_legacy_never_calls_jev():
    calls = []
    def jev(client):
        calls.append(1)
        return JevOutcome(value="J", detail={})
    assert decide("s", lambda: "L", jev, runtime=_rt("legacy")) == "L"
    assert calls == []


def test_decide_shadow_returns_legacy_and_logs_agreement(caplog):
    with caplog.at_level(logging.INFO):
        out = decide("s", lambda: "L", lambda c: JevOutcome(value="J", detail={"p": 0.9}), runtime=_rt("shadow"))
    assert out == "L"
    line = next(r.message for r in caplog.records if "JUDGMENT_SHADOW" in r.message)
    assert "seam=s" in line and "agree=False" in line and "legacy='L'" in line and "jev='J'" in line and "p=0.9" in line


def test_decide_shadow_uses_custom_agree(caplog):
    with caplog.at_level(logging.INFO):
        decide("s", lambda: [1, 2], lambda c: JevOutcome(value=[1, 3], detail={}),
               runtime=_rt("shadow"), agree=lambda a, b: a[0] == b[0])
    assert any("agree=True" in r.message for r in caplog.records)


def test_decide_jev_uses_jev_value():
    assert decide("s", lambda: "L", lambda c: JevOutcome(value="J", detail={}), runtime=_rt("jev")) == "J"


def test_decide_jev_falls_back_on_none_and_on_fallback_outcome(caplog):
    with caplog.at_level(logging.WARNING):
        assert decide("s", lambda: "L", lambda c: None, runtime=_rt("jev")) == "L"
        assert decide("s", lambda: "L", lambda c: JevOutcome.fallback("low_confidence"), runtime=_rt("jev")) == "L"
    msgs = [r.message for r in caplog.records if "JUDGMENT_FALLBACK" in r.message]
    assert any("reason=jev_unavailable" in m for m in msgs) and any("reason=low_confidence" in m for m in msgs)


def test_decide_jev_swallows_exceptions_from_jev_fn(caplog):
    def jev(client):
        raise RuntimeError("bug")
    with caplog.at_level(logging.WARNING):
        assert decide("s", lambda: "L", jev, runtime=_rt("jev")) == "L"
    assert any("JUDGMENT_JEV_ERROR seam=s" in r.message for r in caplog.records)


def test_per_seam_modes_override_the_global_mode():
    http = httpx.Client(transport=httpx.MockTransport(_ok_handler))
    rt = build_runtime(JudgmentConfig(mode="legacy", seams={"admission": "shadow", "rerank": "jev"}),
                       environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    assert rt.client is not None and rt.enabled
    assert rt.mode is JudgmentMode.LEGACY
    assert rt.mode_for("admission") is JudgmentMode.SHADOW
    assert rt.mode_for("rerank") is JudgmentMode.JEV
    assert rt.mode_for("query_intent") is JudgmentMode.LEGACY
    assert rt.enabled_for("query_intent") is False and rt.enabled_for("rerank") is True
    calls = []
    def jev(client):
        calls.append(1)
        return JevOutcome(value="J", detail={})
    assert decide("query_intent", lambda: "L", jev, runtime=rt) == "L" and calls == []
    assert decide("rerank", lambda: "L", jev, runtime=rt) == "J"
    assert decide("admission", lambda: "L", jev, runtime=rt) == "L" and len(calls) == 2


def test_env_override_changes_global_mode_only():
    http = httpx.Client(transport=httpx.MockTransport(_ok_handler))
    rt = build_runtime(JudgmentConfig(mode="jev", seams={"admission": "legacy"}),
                       environ={"TYPESAFE_API_KEY": "k", "VC_JUDGMENT_MODE": "shadow"}, http_client=http)
    assert rt.mode is JudgmentMode.SHADOW and rt.mode_for("admission") is JudgmentMode.LEGACY
