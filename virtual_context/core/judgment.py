"""Typed-judgment layer: route selected engine decisions through TypeSafe Jev.

Three modes (``JudgmentConfig.mode`` / ``VC_JUDGMENT_MODE``):

* ``legacy`` - Jev is never called; every seam behaves exactly as before.
* ``shadow`` - the legacy answer is used; the Jev answer is logged as
  ``JUDGMENT_SHADOW`` for comparison.
* ``jev`` - the Jev answer is used; any failure falls back to legacy and logs
  ``JUDGMENT_FALLBACK``.

Each engine owns a ``JudgmentRuntime`` (``engine.judgment_runtime``) and hands
it to the objects and functions that host the seams, so two engines in one
process never share a mode. The module-level registry (``install``,
``current``, ``override``) is only the default for code paths with no engine
in hand, such as tests and the benchmark harness; it is never written by the
engine.
"""
from __future__ import annotations

import contextlib
import difflib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import httpx

from ..types import JUDGMENT_MODES, JUDGMENT_SEAMS, FactLink, JudgmentConfig

logger = logging.getLogger(__name__)

ENV_MODE = "VC_JUDGMENT_MODE"
T = TypeVar("T")


class JudgmentUnavailable(RuntimeError):
    """Raised by a strict seam when the model answer is required and unavailable."""


class JudgmentMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    JEV = "jev"

    @classmethod
    def parse(cls, raw: str) -> "JudgmentMode":
        if not isinstance(raw, str) or raw not in JUDGMENT_MODES:
            raise ValueError(
                f"judgment mode must be one of {', '.join(JUDGMENT_MODES)}; got {raw!r}"
            )
        return cls(raw)

    @classmethod
    def resolve(
        cls, configured: str, *, environ: Mapping[str, str] | None = None,
    ) -> "JudgmentMode":
        """Env override wins over YAML. Raises on an invalid value in either."""
        env = environ if environ is not None else os.environ
        raw = env.get(ENV_MODE)
        if raw is not None:
            return cls.parse(raw)
        return cls.parse(configured)


# --- questions ---------------------------------------------------------------

def noul_q(instructions: str, *, true: str | None = None, false: str | None = None) -> dict:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {"true": true or "", "false": false or ""}
    return q


def choice_q(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


# --- client ------------------------------------------------------------------

@dataclass(frozen=True)
class JevAnswer:
    kind: str
    value: Any
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None


@dataclass(frozen=True)
class JevResponse:
    answers: dict[str, JevAnswer]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class JevClient:
    """Thin client for ``POST /v1/systemone``. Never raises out of ``ask``."""

    def __init__(
        self,
        config: JudgmentConfig,
        *,
        http_client: httpx.Client | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        env = environ if environ is not None else os.environ
        self._api_key = (env.get(config.api_key_env) or "").strip()
        self._http = http_client
        self._warned_missing_key = False

    def _client(self) -> httpx.Client:
        if self._http is not None:
            return self._http
        from ..providers.base import _get_client
        return _get_client(self.config.timeout_s)

    def _post(self, body: dict) -> dict:
        def send() -> httpx.Response:
            return self._client().post(
                self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.config.timeout_s,
            )

        try:
            resp = send()
        except httpx.RemoteProtocolError:
            # A pooled keep-alive connection the server already closed fails
            # before any response; judgments are side-effect free, so resend once.
            resp = send()
        resp.raise_for_status()
        return resp.json()

    def ask(self, *, seam: str, state: Any, questions: dict[str, dict]) -> JevResponse | None:
        if not self._api_key:
            if not self._warned_missing_key:
                logger.warning("JUDGMENT_JEV_ERROR seam=%s error=missing_api_key env=%s",
                               seam, self.config.api_key_env)
                self._warned_missing_key = True
            return None
        body = {"state": state, "model": self.config.model, "questions": questions}
        started = time.monotonic()
        try:
            data = self._post(body)
            answers = _parse_answers(data["answers"])
        except Exception as exc:  # transport, HTTP, JSON, shape
            logger.warning("JUDGMENT_JEV_ERROR seam=%s error=%s detail=%s",
                           seam, type(exc).__name__, str(exc)[:160])
            return None
        usage = data.get("usage") or {}
        return JevResponse(
            answers=answers,
            model=str(data.get("model", "")),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            latency_ms=(time.monotonic() - started) * 1000.0,
        )


def _parse_answers(raw: Any) -> dict[str, JevAnswer]:
    if not isinstance(raw, dict):
        raise ValueError("answers is not an object")
    out: dict[str, JevAnswer] = {}
    for key, item in raw.items():
        kind = item["type"]
        if kind == "noul":
            out[key] = JevAnswer(kind="noul", value=float(item["noul"]))
        elif kind == "choice":
            out[key] = JevAnswer(
                kind="choice", value=str(item["choice"]),
                probabilities={str(k): float(v) for k, v in (item.get("probabilities") or {}).items()},
                confidence=(float(item["confidence"]) if item.get("confidence") is not None else None),
            )
        elif kind == "score":
            out[key] = JevAnswer(
                kind="score", value=float(item["score"]),
                probabilities={str(k): float(v) for k, v in (item.get("probabilities") or {}).items()},
                confidence=(float(item["confidence"]) if item.get("confidence") is not None else None),
            )
        else:
            raise ValueError(f"unknown answer type {kind!r}")
    return out


# --- runtime -----------------------------------------------------------------

@dataclass(frozen=True)
class JevOutcome:
    value: Any
    detail: dict = field(default_factory=dict)
    response: JevResponse | None = None
    fallback_reason: str | None = None

    @classmethod
    def fallback(cls, reason: str, *, response: JevResponse | None = None) -> "JevOutcome":
        return cls(value=None, detail={}, response=response, fallback_reason=reason)


@dataclass(frozen=True)
class JudgmentRuntime:
    mode: JudgmentMode
    client: JevClient | None
    config: JudgmentConfig
    seam_modes: dict[str, JudgmentMode] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """True when any seam can reach Jev."""
        if self.client is None:
            return False
        return self.mode is not JudgmentMode.LEGACY or any(
            m is not JudgmentMode.LEGACY for m in self.seam_modes.values()
        )

    def mode_for(self, seam: str) -> JudgmentMode:
        return self.seam_modes.get(seam, self.mode)

    def enabled_for(self, seam: str) -> bool:
        return self.client is not None and self.mode_for(seam) is not JudgmentMode.LEGACY


def build_runtime(
    config: JudgmentConfig,
    *,
    environ: Mapping[str, str] | None = None,
    http_client: httpx.Client | None = None,
) -> JudgmentRuntime:
    mode = JudgmentMode.resolve(config.mode, environ=environ)
    seam_modes: dict[str, JudgmentMode] = {}
    for seam, raw in (config.seams or {}).items():
        if seam not in JUDGMENT_SEAMS:
            raise ValueError(f"unknown judgment seam {seam!r}; known: {', '.join(JUDGMENT_SEAMS)}")
        seam_modes[seam] = JudgmentMode.parse(raw)
    client = None
    if mode is not JudgmentMode.LEGACY or any(m is not JudgmentMode.LEGACY for m in seam_modes.values()):
        client = JevClient(config, http_client=http_client, environ=environ)
    return JudgmentRuntime(mode=mode, client=client, config=config, seam_modes=seam_modes)


_LEGACY_RUNTIME = JudgmentRuntime(JudgmentMode.LEGACY, None, JudgmentConfig())
_installed: JudgmentRuntime = _LEGACY_RUNTIME
_lock = threading.Lock()


def install(runtime: JudgmentRuntime) -> None:
    global _installed
    with _lock:
        _installed = runtime


def current() -> JudgmentRuntime:
    return _installed


def reset() -> None:
    install(_LEGACY_RUNTIME)


@contextlib.contextmanager
def override(runtime: JudgmentRuntime) -> Iterator[JudgmentRuntime]:
    previous = _installed
    install(runtime)
    try:
        yield runtime
    finally:
        install(previous)


# --- three-mode decision -----------------------------------------------------

def _format_detail(detail: Mapping[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in detail.items())


def _run_jev(seam: str, jev: Callable[[JevClient], JevOutcome | None], client: JevClient) -> JevOutcome | None:
    try:
        return jev(client)
    except Exception as exc:
        logger.warning("JUDGMENT_JEV_ERROR seam=%s error=%s detail=%s",
                       seam, type(exc).__name__, str(exc)[:160])
        return None


def decide(
    seam: str,
    legacy: Callable[[], T],
    jev: Callable[[JevClient], JevOutcome | None],
    *,
    runtime: JudgmentRuntime | None = None,
    agree: Callable[[T, T], bool] | None = None,
    describe: Callable[[T], str] = repr,
) -> T:
    rt = runtime if runtime is not None else current()
    if not rt.enabled_for(seam):
        return legacy()
    assert rt.client is not None
    if rt.mode_for(seam) is JudgmentMode.SHADOW:
        legacy_value = legacy()
        outcome = _run_jev(seam, jev, rt.client)
        if outcome is None or outcome.fallback_reason:
            reason = outcome.fallback_reason if outcome else "jev_unavailable"
            logger.info("JUDGMENT_SHADOW seam=%s agree=None legacy=%s jev=None reason=%s",
                        seam, describe(legacy_value), reason)
            return legacy_value
        same = agree(legacy_value, outcome.value) if agree else (describe(legacy_value) == describe(outcome.value))
        resp = outcome.response
        logger.info(
            "JUDGMENT_SHADOW seam=%s agree=%s legacy=%s jev=%s %s ms=%.0f tokens=%d/%d",
            seam, same, describe(legacy_value), describe(outcome.value), _format_detail(outcome.detail),
            resp.latency_ms if resp else 0.0,
            resp.input_tokens if resp else 0, resp.output_tokens if resp else 0,
        )
        return legacy_value
    outcome = _run_jev(seam, jev, rt.client)
    if outcome is None or outcome.fallback_reason:
        reason = outcome.fallback_reason if outcome else "jev_unavailable"
        logger.warning("JUDGMENT_FALLBACK seam=%s reason=%s", seam, reason)
        return legacy()
    return outcome.value


# --- seam S2: query intent (find_quote) --------------------------------------

INTENT_CRITERIA: dict[str, str] = {
    "current_state": (
        "asks for the present or latest state of something: now, currently, these days, "
        "at the moment, latest, still, or a status question about how things stand"
    ),
    "default": (
        "a topical lookup, a request for a quote or a fact, or a question about the past "
        "with no present-state framing"
    ),
}


def jev_query_intent(client: JevClient, query: str, *, min_confidence: float = 0.5) -> JevOutcome | None:
    resp = client.ask(
        seam="query_intent",
        state={"query": query},
        questions={"intent": choice_q(
            "What kind of memory lookup does `query` ask for?", INTENT_CRITERIA,
        )},
    )
    if resp is None:
        return None
    ans = resp.answers.get("intent")
    if ans is None or ans.value not in INTENT_CRITERIA:
        return JevOutcome.fallback("bad_answer", response=resp)
    conf = ans.confidence if ans.confidence is not None else 1.0
    if conf < min_confidence:
        return JevOutcome.fallback("low_confidence", response=resp)
    return JevOutcome(value=ans.value, detail={"p": round(ans.probabilities.get(ans.value, conf), 3),
                                              "confidence": round(conf, 3)}, response=resp)


def judge_query_intent(query: str, legacy: Callable[[], str], *, runtime: JudgmentRuntime | None = None) -> str:
    return decide("query_intent", legacy, lambda c: jev_query_intent(c, query), runtime=runtime, describe=str)


# --- seam S3: inbound temporal intent ----------------------------------------

def jev_temporal_intent(client: JevClient, message: str, *, threshold: float) -> JevOutcome | None:
    resp = client.ask(
        seam="temporal_intent",
        state={"message": message},
        questions={"temporal": noul_q(
            "Does `message` ask about when something happened, the order of events, or an "
            "earlier point in the conversation, rather than about the topic itself?",
            true="the question is about timing, sequence, or an earlier point in the conversation",
            false="the question is about the subject matter with no timing or sequence framing",
        )},
    )
    if resp is None:
        return None
    ans = resp.answers.get("temporal")
    if ans is None or ans.kind != "noul":
        return JevOutcome.fallback("bad_answer", response=resp)
    return JevOutcome(value=bool(ans.value >= threshold), detail={"p": round(ans.value, 3)}, response=resp)


def judge_temporal_intent(message: str, legacy: Callable[[], bool], *, runtime: JudgmentRuntime | None = None) -> bool:
    rt = runtime if runtime is not None else current()
    return decide("temporal_intent", legacy,
                  lambda c: jev_temporal_intent(c, message, threshold=rt.config.noul_threshold),
                  runtime=rt, describe=str)


# --- seam S4: safety-critical personal evidence ------------------------------

def jev_safety_critical(client: JevClient, text: str, *, threshold: float) -> JevOutcome | None:
    resp = client.ask(
        seam="safety_critical",
        state={"text": text},
        questions={"safety_critical": noul_q(
            "Does `text` state a correction, a start or stop, or a change of the speaker's own "
            "situation (medication, health, regimen, relationship, plan) that a later reader "
            "must not miss?",
            true="the speaker corrects an earlier claim about themselves or reports starting, "
                 "stopping, switching, or changing something in their own life",
            false="no such statement, an incidental use of words like stopped or started, or a "
                  "change that concerns someone other than the speaker",
        )},
    )
    if resp is None:
        return None
    ans = resp.answers.get("safety_critical")
    if ans is None or ans.kind != "noul":
        return JevOutcome.fallback("bad_answer", response=resp)
    return JevOutcome(value=bool(ans.value >= threshold), detail={"p": round(ans.value, 3)}, response=resp)


def judge_safety_critical(text: str, legacy: Callable[[], bool], *, runtime: JudgmentRuntime | None = None) -> bool:
    rt = runtime if runtime is not None else current()
    return decide("safety_critical", legacy,
                  lambda c: jev_safety_critical(c, text, threshold=rt.config.noul_threshold),
                  runtime=rt, describe=str)


# --- seam S1: retrieval shortlist rerank -------------------------------------

def spearman(order_a: list, order_b: list) -> float:
    """Spearman rank correlation between two orderings of the same items."""
    n = len(order_a)
    if n < 2 or set(order_a) != set(order_b):
        return 0.0
    pos_b = {item: i for i, item in enumerate(order_b)}
    d2 = sum((i - pos_b[item]) ** 2 for i, item in enumerate(order_a))
    return round(1.0 - (6.0 * d2) / (n * (n * n - 1)), 4)


def jev_rerank(
    client: JevClient, query: str, candidates: list[tuple[str, str]], *, max_state_bytes: int,
) -> JevOutcome | None:
    state = {"query": query, "candidates": {key: text for key, text in candidates}}
    if len(json.dumps(state)) > max_state_bytes:
        logger.info("JUDGMENT_SKIP seam=rerank reason=state_size candidates=%d", len(candidates))
        return JevOutcome.fallback("state_size")
    questions = {
        key: noul_q(
            f"Does `candidates.{key}` contain information needed to answer `query`?",
            true="states facts, preferences, events, or decisions the query asks about",
            false="same general topic without the asked-for information, or unrelated",
        )
        for key, _ in candidates
    }
    resp = client.ask(seam="rerank", state=state, questions=questions)
    if resp is None:
        return None
    probs: dict[str, float] = {}
    for key, _ in candidates:
        ans = resp.answers.get(key)
        if ans is None or ans.kind != "noul":
            return JevOutcome.fallback("bad_answer", response=resp)
        probs[key] = float(ans.value)
    return JevOutcome(value=probs, detail={}, response=resp)


def rerank_summaries(query: str, summaries: list, *, runtime: JudgmentRuntime | None = None) -> list:
    """Reorder the retriever's candidate summaries in shadow/jev mode.

    Stable sort by descending relevance probability; candidates below
    ``rerank_min_probability`` move to the end. Nothing is dropped. Returns
    the input list object untouched in legacy mode.
    """
    rt = runtime if runtime is not None else current()
    if not rt.enabled_for("rerank") or len(summaries) < 2:
        return summaries
    cfg = rt.config
    keys = [f"c{i}" for i in range(len(summaries))]

    def legacy() -> list:
        return list(summaries)

    def jev(client: JevClient) -> JevOutcome | None:
        outcome = jev_rerank(client, query, [(k, s.summary) for k, s in zip(keys, summaries)],
                             max_state_bytes=cfg.rerank_max_state_bytes)
        if outcome is None or outcome.fallback_reason:
            return outcome
        probs = outcome.value
        order = sorted(
            range(len(summaries)),
            key=lambda i: (0 if probs[keys[i]] >= cfg.rerank_min_probability else 1, -probs[keys[i]], i),
        )
        reordered = [summaries[i] for i in order]
        legacy_refs = [s.ref for s in summaries]
        jev_refs = [s.ref for s in reordered]
        detail = {
            "spearman": spearman(legacy_refs, jev_refs),
            "top1_legacy": legacy_refs[0],
            "top1_jev": jev_refs[0],
            "p_top": round(probs[keys[order[0]]], 3),
            "n": len(summaries),
        }
        return JevOutcome(value=reordered, detail=detail, response=outcome.response)

    return decide(
        "rerank", legacy, jev, runtime=rt,
        agree=lambda a, b: a[0].ref == b[0].ref,
        describe=lambda items: ",".join(s.ref for s in items[:5]),
    )


def select_topics(
    query: str,
    ranked_tags: list[str],
    load_summaries: Callable[[list[str]], dict[str, str]],
    *,
    max_results: int,
    runtime: JudgmentRuntime | None = None,
    segment_candidates: list[tuple[str, str]] | tuple = (),
) -> tuple[list[str], bool, list[str]]:
    """Choose what to retrieve for *query* from the fused ranking and candidate segments.

    Returns ``(tags, chosen_by_jev, ordered)``. Legacy takes the top
    ``max_results`` tags and orders nothing. Jev scores, in one call, the
    stored summary of each tag in the top ``topic_pool_size`` and each
    ``(segment_ref, summary)`` candidate. Items at or above
    ``topic_min_probability`` are kept, most likely first (ties keep the
    candidate order), at most ``max_results`` topics and ``max_results``
    segments; ``ordered`` lists them as ``"T:<tag>"`` and ``"S:<ref>"``.
    When nothing qualifies, the legacy choice stands.
    """
    rt = runtime if runtime is not None else current()

    def legacy() -> tuple[list[str], bool, list[str]]:
        return list(ranked_tags[:max_results]), False, []

    if not rt.enabled_for("topic_select") or len(ranked_tags) + len(segment_candidates) < 2:
        return legacy()
    cfg = rt.config
    pool = list(ranked_tags[:max(cfg.topic_pool_size, max_results)])

    def jev(client: JevClient) -> JevOutcome | None:
        texts = load_summaries(pool)
        candidates = [(f"T:{tag}", texts[tag]) for tag in pool if texts.get(tag)]
        candidates += [(f"S:{ref}", text) for ref, text in segment_candidates if text]
        if len(candidates) < 2:
            return JevOutcome.fallback("no_candidates")
        outcome = jev_rerank(client, query, candidates, max_state_bytes=cfg.rerank_max_state_bytes)
        if outcome is None or outcome.fallback_reason:
            return outcome
        probs = outcome.value
        position = {key: index for index, (key, _) in enumerate(candidates)}
        ranked = sorted(
            (key for key in probs if probs[key] >= cfg.topic_min_probability),
            key=lambda key: (-probs[key], position[key]),
        )
        ordered: list[str] = []
        counts = {"T": 0, "S": 0}
        for key in ranked:
            kind = key[0]
            if counts[kind] < max_results:
                counts[kind] += 1
                ordered.append(key)
        if not ordered:
            return JevOutcome.fallback("none_relevant", response=outcome.response)
        tags = [key[2:] for key in ordered if key.startswith("T:")]
        detail = {
            "pool": len(candidates),
            "topics": counts["T"],
            "segments": counts["S"],
            "p_top": round(probs[ordered[0]], 3),
        }
        return JevOutcome(value=(tags, True, ordered), detail=detail, response=outcome.response)

    return decide(
        "topic_select", legacy, jev, runtime=rt,
        agree=lambda a, b: a[0][:3] == b[0][:3],
        describe=lambda value: ",".join(value[2][:5] or value[0][:5]),
    )


# --- seam S5: actor-card admission -------------------------------------------

COVERAGE_CRITERIA: dict[str, str] = {
    "substantive": "the interaction contains durable context about this actor worth remembering beyond this exchange",
    "greeting_only": "the actor only greeted, thanked, or made small talk",
    "one_off_trivia": "the actor asked a one-off question or made a remark with no lasting relevance to who they are",
    "bot_meta_or_test": "the actor was testing, probing, or talking about the bot itself rather than sharing durable context",
    "no_durable_context": "the exchange has content but nothing that would still matter about this actor later",
    "insufficient_evidence": "the visible evidence is too thin or truncated to judge the interaction",
}

REASON_CRITERIA: dict[str, str] = {
    "durable": "the body is fully entailed by the cited actor-authored evidence, has the right subject and kind, and describes something lasting about this actor",
    "temporary": "the evidence describes a momentary state or one-time situation, not something lasting",
    "test_probe": "the evidence is the actor testing or probing the agent rather than a genuine statement",
    "stopped_or_replaced": "the evidence shows the actor has stopped or replaced what the body claims",
    "completed": "the body describes a goal or activity the evidence shows is already finished",
    "expired": "the body is a finite communication preference whose expires_at has passed at as_of",
    "contradicted": "other cited evidence contradicts the body",
    "insufficient_evidence": "the cited evidence does not entail the body: a qualifier is dropped, the claim is broadened, frequency or habit is asserted from a single message, or the actor's exact terms were normalized away",
    "not_durable": "the evidence supports the body only as a passing remark with no lasting intent",
    "not_person_card": "the body exposes internal ontology or tag language, or serializes a machine fact rather than a natural statement about a person",
    "wrong_subject": "the body assigns an action, trait, or property to the actor that the evidence attributes to someone else, including instructions the actor gave the agent",
    "wrong_kind": "the claim is real but filed under the wrong kind, for example an external-action request classified as communication_pref, or an agent persona assignment classified as interaction_style",
    "irrelevant_citation": "a cited id does not materially support the body",
    "redundant": "an existing admitted entry already carries this claim",
    "explicit_privacy_request": "actor-authored evidence explicitly asks that the cited information not be retained or reused",
    "agent_refused": "the agent refused, deflected, or deferred the request, or a behavior-change request has no visible honored signal",
    "safety_posture_request": "the request asks the agent to change its safety posture, regardless of the reply",
}


@dataclass(frozen=True)
class AdmissionJudgment:
    text: str
    substantive: bool
    coverage_reason: str
    decisions: dict[str, dict]
    response: JevResponse | None


def jev_admission(client: JevClient, payload: dict, eligible: list[str], *, subject_threshold: float = 0.5) -> JevOutcome | None:
    questions: dict[str, dict] = {
        "coverage": choice_q(
            "Considering `actor_turns`, `facts`, and `evidence_segments`, how should this "
            "interaction be classified for actor-card coverage?", COVERAGE_CRITERIA,
        ),
    }
    for cid in eligible:
        questions[f"reason__{cid}"] = choice_q(
            f"For the candidate with candidate_id '{cid}' in `candidates`, judged against the "
            f"cited evidence in `actor_turns`, `facts`, and `evidence_segments` at `as_of`, "
            f"which admission reason applies? Only durable admits the candidate.",
            REASON_CRITERIA,
        )
        questions[f"subject__{cid}"] = noul_q(
            f"Is the claim in the candidate with candidate_id '{cid}' in `candidates` about the "
            f"actor themselves, the author of the messages in `actor_turns`?",
            true="the body describes the actor's own traits, facts, preferences, goals, or history",
            false="the body describes another person, or turns an instruction the actor gave the "
                  "agent into a property of the actor",
        )
    resp = client.ask(seam="admission", state=payload, questions=questions)
    if resp is None:
        return None
    cov = resp.answers.get("coverage")
    if cov is None or cov.value not in COVERAGE_CRITERIA:
        return JevOutcome.fallback("bad_answer", response=resp)
    decisions = []
    subject_flips = 0
    for cid in eligible:
        ans = resp.answers.get(f"reason__{cid}")
        if ans is None or ans.value not in REASON_CRITERIA:
            return JevOutcome.fallback("bad_answer", response=resp)
        reason = ans.value
        subject = resp.answers.get(f"subject__{cid}")
        if reason == "durable" and subject is not None and subject.kind == "noul" and subject.value < subject_threshold:
            reason = "wrong_subject"
            subject_flips += 1
        decisions.append({"candidate_id": cid, "admit": reason == "durable", "reason": reason})
    value = {"substantive": cov.value == "substantive", "coverage_reason": cov.value, "decisions": decisions}
    return JevOutcome(value=value, detail={"coverage_conf": round(cov.confidence or 0.0, 3),
                                          "subject_flips": subject_flips}, response=resp)


def judge_admission(payload: dict, eligible: list[str], *, runtime: JudgmentRuntime | None = None) -> AdmissionJudgment | None:
    """Return the Jev admission judgment in shadow or jev mode; None in legacy or on failure."""
    rt = runtime if runtime is not None else current()
    if not rt.enabled_for("admission"):
        return None
    assert rt.client is not None
    payload, trim = trim_admission_payload(payload, rt.config.admission_max_state_bytes)
    if payload is None:
        logger.info("JUDGMENT_SKIP seam=admission reason=state_too_large max_bytes=%d %s",
                    rt.config.admission_max_state_bytes, _format_detail(trim))
        logger.warning("JUDGMENT_FALLBACK seam=admission reason=state_too_large")
        return None
    if trim:
        logger.info("JUDGMENT_TRIM seam=admission %s bytes=%d max_bytes=%d",
                    _format_detail(trim), _payload_bytes(payload), rt.config.admission_max_state_bytes)
    outcome = _run_jev(
        "admission",
        lambda c: jev_admission(c, payload, eligible, subject_threshold=rt.config.noul_threshold),
        rt.client,
    )
    if outcome is None or outcome.fallback_reason:
        reason = outcome.fallback_reason if outcome else "jev_unavailable"
        logger.warning("JUDGMENT_FALLBACK seam=admission reason=%s", reason)
        return None
    value = outcome.value
    return AdmissionJudgment(
        text=json.dumps(value, separators=(",", ":")),
        substantive=value["substantive"],
        coverage_reason=value["coverage_reason"],
        decisions={d["candidate_id"]: d for d in value["decisions"]},
        response=outcome.response,
    )


def log_admission_shadow(judgment: AdmissionJudgment, legacy_substantive: bool, legacy_decisions: dict[str, dict]) -> None:
    ids = sorted(set(legacy_decisions) | set(judgment.decisions))
    agree = sum(
        1 for cid in ids
        if legacy_decisions.get(cid, {}).get("reason") == judgment.decisions.get(cid, {}).get("reason")
    )
    resp = judgment.response
    logger.info(
        "JUDGMENT_SHADOW seam=admission agree=%s coverage_agree=%s candidates=%d reason_agree=%d legacy=%s jev=%s ms=%.0f tokens=%d/%d",
        agree == len(ids) and legacy_substantive == judgment.substantive,
        legacy_substantive == judgment.substantive, len(ids), agree,
        ",".join(f"{cid}:{legacy_decisions.get(cid, {}).get('reason', '-')}" for cid in ids),
        ",".join(f"{cid}:{judgment.decisions.get(cid, {}).get('reason', '-')}" for cid in ids),
        resp.latency_ms if resp else 0.0, resp.input_tokens if resp else 0, resp.output_tokens if resp else 0,
    )


# --- admission state cap -----------------------------------------------------

def _payload_bytes(payload: Any) -> int:
    return len(json.dumps(payload, separators=(",", ":"), default=str))


def trim_admission_payload(payload: dict, max_bytes: int) -> tuple[dict | None, dict]:
    """Shrink an admission payload to ``max_bytes`` without touching candidates or facts.

    Evidence segments go first, then actor turns no candidate cites (truncated
    turns before intact ones, oldest first). Returns ``(payload, {})`` when
    nothing had to go, ``(trimmed, detail)`` after trimming, and
    ``(None, detail)`` when the cited material alone does not fit.
    """
    if max_bytes <= 0 or _payload_bytes(payload) <= max_bytes:
        return payload, {}
    trimmed = dict(payload)
    detail = {"dropped_evidence": 0, "dropped_turns": 0}
    evidence = list(trimmed.get("evidence_segments") or [])
    if evidence:
        detail["dropped_evidence"] = len(evidence)
        trimmed["evidence_segments"] = []
    size = _payload_bytes(trimmed)
    if size <= max_bytes:
        return trimmed, detail
    cited: set[str] = set()
    for cand in trimmed.get("candidates") or []:
        if isinstance(cand, dict):
            cited.update(str(t) for t in (cand.get("turn_ids") or []))
    turns = list(trimmed.get("actor_turns") or [])
    order = sorted(range(len(turns)), key=lambda i: (0 if turns[i].get("truncated") else 1, i))
    dropped: set[int] = set()
    for i in order:
        if str(turns[i].get("id")) in cited:
            continue
        dropped.add(i)
        size -= _payload_bytes(turns[i]) + 1
        if size <= max_bytes:
            break
    trimmed["actor_turns"] = [t for j, t in enumerate(turns) if j not in dropped]
    detail["dropped_turns"] = len(dropped)
    if _payload_bytes(trimmed) > max_bytes:
        return None, detail
    return trimmed, detail


def _merge_responses(responses: list[JevResponse]) -> JevResponse | None:
    if not responses:
        return None
    return JevResponse(
        answers={k: v for r in responses for k, v in r.answers.items()},
        model=responses[0].model,
        input_tokens=sum(r.input_tokens for r in responses),
        output_tokens=sum(r.output_tokens for r in responses),
        latency_ms=sum(r.latency_ms for r in responses),
    )


MAX_QUESTIONS_PER_CALL = 40


# --- seam S6: tag reuse ------------------------------------------------------

_TAG_TOKEN_STOP = frozenset({"and", "the", "for", "with", "from", "into"})


def _tag_tokens(tag: str) -> set[str]:
    return {t for t in re.split(r"[-_\s/]+", tag.lower()) if len(t) >= 3 and t not in _TAG_TOKEN_STOP}


def _lexical_similarity(a: str, b: str) -> float:
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = _tag_tokens(a), _tag_tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return round(0.5 * ratio + 0.5 * jaccard, 4)


def nearest_existing_tags(
    proposed: str, existing: list[str], *, limit: int, extra_scores: Mapping[str, float] | None = None,
) -> list[str]:
    """Rank ``existing`` tag names by similarity to ``proposed``.

    Name similarity blends character ratio with shared-token overlap;
    ``extra_scores`` (for example embedding cosine per tag) can lift a tag
    whose name alone would not surface it. ``proposed`` itself is excluded.
    """
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for tag in existing:
        if not tag or tag == proposed or tag in seen:
            continue
        seen.add(tag)
        score = _lexical_similarity(proposed, tag)
        if extra_scores:
            score = max(score, float(extra_scores.get(tag, 0.0)))
        scored.append((score, tag))
    scored.sort(key=lambda st: (-st[0], st[1]))
    return [tag for _, tag in scored[:limit]]


def jev_tag_reuse(
    client: JevClient, text: str, proposed: list[str], candidates_by_tag: Mapping[str, list[str]],
    *, min_confidence: float = 0.5,
) -> JevOutcome | None:
    keys = [(str(i), tag) for i, tag in enumerate(proposed) if candidates_by_tag.get(tag)]
    state = {
        "text": text,
        "proposed_tags": {k: tag for k, tag in keys},
        "existing_tags": {k: [c for c in candidates_by_tag[tag] if c != "none"] for k, tag in keys},
    }
    questions: dict[str, dict] = {}
    for k, tag in keys:
        criteria = {
            cand: f"the text is about the topic the existing tag '{cand}' already names"
            for cand in state["existing_tags"][k]
        }
        criteria["none"] = "no listed existing tag names this topic; the proposed tag is a genuinely new topic"
        questions[f"reuse__{k}"] = choice_q(
            f"`proposed_tags.{k}` is a new topic tag proposed for `text`. Which entry of "
            f"`existing_tags.{k}` already names the same topic, so the new tag should be "
            f"replaced by it? Choose none unless an existing tag covers the same subject.",
            criteria,
        )
    resp = client.ask(seam="tag_reuse", state=state, questions=questions)
    if resp is None:
        return None
    mapping: dict[str, str | None] = {tag: None for tag in proposed}
    detail: dict[str, Any] = {}
    for k, tag in keys:
        ans = resp.answers.get(f"reuse__{k}")
        if ans is None or ans.kind != "choice" or ans.value not in questions[f"reuse__{k}"]["criteria"]:
            return JevOutcome.fallback("bad_answer", response=resp)
        conf = ans.confidence if ans.confidence is not None else ans.probabilities.get(ans.value, 1.0)
        if ans.value != "none" and conf >= min_confidence:
            mapping[tag] = ans.value
        detail[f"p{k}"] = round(conf, 3)
    return JevOutcome(value=mapping, detail=detail, response=resp)


def judge_tag_reuse(
    text: str, proposed: list[str], candidates_by_tag: Mapping[str, list[str]],
    *, runtime: JudgmentRuntime | None = None,
) -> dict[str, str | None]:
    """Map each proposed new tag to an existing tag it duplicates, or None to keep it new."""
    legacy_value: dict[str, str | None] = {tag: None for tag in proposed}
    if not any(candidates_by_tag.get(tag) for tag in proposed):
        return legacy_value
    return decide(
        "tag_reuse", lambda: dict(legacy_value),
        lambda c: jev_tag_reuse(c, text, proposed, candidates_by_tag),
        runtime=runtime, agree=lambda a, b: a == b,
        describe=lambda m: ",".join(f"{k}->{v or 'new'}" for k, v in m.items()),
    )


# --- seam S7: fact supersession and links ------------------------------------

SUPERSESSION_CRITERIA: dict[str, str] = {
    "supersedes": (
        "`new_fact` gives a newer value for the same attribute of the same subject that this "
        "candidate records (an updated address, record, job, plan, preference), so the candidate is now stale"
    ),
    "duplicates": (
        "`new_fact` and this candidate describe the same underlying event or state in different "
        "words, and the candidate is the less detailed of the two"
    ),
    "contradicts": (
        "`new_fact` and this candidate cannot both be true, neither is a newer value of the other, "
        "and the candidate is the one that must give way"
    ),
    "independent": (
        "this candidate is about a different attribute, event, or subject (an event and a state that "
        "merely share a keyword are independent), or is more specific than `new_fact`, or carries a "
        "newer session_date than `new_fact`, and stays valid alongside it"
    ),
}
_SUPERSEDING = ("supersedes", "duplicates", "contradicts")
_LINK_RELATION = {"supersedes": "supersedes", "duplicates": "same_as", "contradicts": "contradicts"}


def jev_supersession(
    client: JevClient, new_fact: str, candidates: list[tuple[str, str, str]],
    *, new_fact_date: str = "", min_confidence: float = 0.5,
) -> JevOutcome | None:
    """``candidates`` are ``(fact_id, text, session_date)``; value maps id to (relation, confidence)."""
    state = {
        "new_fact": new_fact,
        "new_fact_session_date": new_fact_date,
        "candidates": {cid: {"content": text, "session_date": date} for cid, text, date in candidates},
    }
    questions = {
        f"rel__{cid}": choice_q(
            f"How does `new_fact` relate to `candidates.{cid}`? Mark supersedes, duplicates, or "
            f"contradicts only when both describe the same attribute of the same subject; "
            f"otherwise independent.",
            SUPERSESSION_CRITERIA,
        )
        for cid, _, _ in candidates
    }
    resp = client.ask(seam="supersession", state=state, questions=questions)
    if resp is None:
        return None
    relations: dict[str, tuple[str, float]] = {}
    for cid, _, _ in candidates:
        ans = resp.answers.get(f"rel__{cid}")
        if ans is None or ans.kind != "choice" or ans.value not in SUPERSESSION_CRITERIA:
            return JevOutcome.fallback("bad_answer", response=resp)
        conf = ans.confidence if ans.confidence is not None else ans.probabilities.get(ans.value, 1.0)
        rel = ans.value if (ans.value == "independent" or conf >= min_confidence) else "independent"
        relations[cid] = (rel, float(conf))
    superseded = sum(1 for rel, _ in relations.values() if rel in _SUPERSEDING)
    return JevOutcome(value=relations, detail={"n": len(candidates), "superseded": superseded}, response=resp)


def _superseded_ids(candidates: list[tuple[str, str, str]], relations: Mapping[str, tuple[str, float]]) -> list[str]:
    return [cid for cid, _, _ in candidates if relations.get(cid, ("independent", 0.0))[0] in _SUPERSEDING]


def judge_supersession(
    new_fact: str, candidates: list[tuple[str, str, str]], *, legacy: Callable[[], list[str]],
    runtime: JudgmentRuntime | None = None, new_fact_date: str = "",
) -> list[str]:
    """Return the candidate fact ids that ``new_fact`` supersedes, duplicates, or contradicts."""
    if not candidates:
        return legacy()

    def jev(client: JevClient) -> JevOutcome | None:
        out = jev_supersession(client, new_fact, candidates, new_fact_date=new_fact_date)
        if out is None or out.fallback_reason:
            return out
        return JevOutcome(value=_superseded_ids(candidates, out.value), detail=out.detail, response=out.response)

    return decide("supersession", legacy, jev, runtime=runtime,
                  agree=lambda a, b: set(a) == set(b), describe=lambda ids: ",".join(ids) or "-")


def judge_fact_links(
    new_fact_id: str, new_fact: str, candidates: list[tuple[str, str, str]],
    *, legacy: Callable[[], tuple[list[FactLink], list[str]]],
    runtime: JudgmentRuntime | None = None, new_fact_date: str = "",
) -> tuple[list[FactLink], list[str]]:
    """Return ``(links from the new fact, superseded candidate ids)``."""
    if not candidates:
        return legacy()

    def jev(client: JevClient) -> JevOutcome | None:
        out = jev_supersession(client, new_fact, candidates, new_fact_date=new_fact_date)
        if out is None or out.fallback_reason:
            return out
        links: list[FactLink] = []
        for cid, _, _ in candidates:
            rel, conf = out.value[cid]
            if rel in _LINK_RELATION:
                links.append(FactLink(
                    source_fact_id=new_fact_id, target_fact_id=cid, relation_type=_LINK_RELATION[rel],
                    confidence=round(conf, 3), context=f"jev: {rel} (p={round(conf, 3)})",
                    created_by="supersession",
                ))
        return JevOutcome(value=(links, _superseded_ids(candidates, out.value)), detail=out.detail,
                          response=out.response)

    def describe(value: tuple[list[FactLink], list[str]]) -> str:
        links, superseded = value
        return (f"sup={','.join(superseded) or '-'}|links="
                f"{';'.join(f'{l.target_fact_id}:{l.relation_type}' for l in links) or '-'}")

    return decide("supersession", legacy, jev, runtime=runtime,
                  agree=lambda a, b: set(a[1]) == set(b[1]), describe=describe)


# --- seam S8: tag consolidation ----------------------------------------------

def candidate_tag_pairs(
    tags: list[str], *, limit: int = 200, min_similarity: float = 0.6,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    knn: int = 8, min_cosine: float = 0.72, max_block: int = 200,
    strict_embed: bool = False,
) -> list[tuple[str, str]]:
    """Tag pairs worth asking about: shared name token, close spelling, or close embedding.

    Lexical candidates are generated by blocking (tags sharing a name token or a
    5-character prefix) so vocabularies of thousands of tags stay tractable;
    blocks larger than ``max_block`` are treated as uninformative and skipped.
    Embedding candidates are the ``knn`` nearest neighbours of each tag at
    cosine ``min_cosine`` or above. Pairs are ranked by their best score and
    capped at ``limit``.
    """
    unique = sorted({t for t in tags if t and not t.startswith("_")})
    if len(unique) < 2:
        return []
    tokens = {t: _tag_tokens(t) for t in unique}
    blocks: dict[str, set[str]] = {}
    for t in unique:
        for tok in tokens[t]:
            blocks.setdefault("tok:" + tok, set()).add(t)
        if len(t) >= 5:
            blocks.setdefault("pre:" + t[:5], set()).add(t)
    scored: dict[tuple[str, str], float] = {}
    for members in blocks.values():
        if len(members) < 2 or len(members) > max_block:
            continue
        ordered = sorted(members)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                key = (a, b)
                if key in scored:
                    continue
                shared = tokens[a] & tokens[b]
                if shared:
                    overlap = len(shared) / max(1, len(tokens[a] | tokens[b]))
                    score = max(0.5 + 0.5 * overlap, difflib.SequenceMatcher(None, a, b).ratio())
                else:
                    matcher = difflib.SequenceMatcher(None, a, b)
                    if matcher.quick_ratio() < min_similarity:
                        continue
                    score = matcher.ratio()
                    if score < min_similarity:
                        continue
                scored[key] = score
    if embed_fn is not None:
        try:
            import numpy as np
            vecs = np.asarray(embed_fn(unique), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1)
            norms[norms == 0] = 1.0
            unit = vecs / norms[:, None]
            k = min(knn, len(unique) - 1)
            for start_row in range(0, len(unique), 512):
                chunk = unit[start_row:start_row + 512]
                sims = chunk @ unit.T
                for offset in range(chunk.shape[0]):
                    i = start_row + offset
                    row = sims[offset]
                    row[i] = -1.0
                    nearest = np.argpartition(-row, k)[:k] if k < len(row) else np.arange(len(row))
                    for j in nearest:
                        sim = float(row[j])
                        if sim < min_cosine:
                            continue
                        a, b = (unique[i], unique[j]) if unique[i] < unique[j] else (unique[j], unique[i])
                        scored[(a, b)] = max(scored.get((a, b), 0.0), sim)
        except Exception:
            if strict_embed:
                raise
            logger.debug("embedding pair candidates unavailable", exc_info=True)
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pair for pair, _ in ranked[:limit]]


def groups_from_pairs(pairs: list[tuple[str, str]], rank: Mapping[str, int]) -> list[dict]:
    """Union-find the accepted pairs; the highest-ranked member (ties: alphabetical) is canonical."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    clusters: dict[str, set[str]] = {}
    for tag in parent:
        clusters.setdefault(find(tag), set()).add(tag)
    groups: list[dict] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda t: (-int(rank.get(t, 0)), t))
        groups.append({"canonical": canonical, "aliases": sorted(members - {canonical})})
    groups.sort(key=lambda g: g["canonical"])
    return groups


def jev_tag_consolidation(
    client: JevClient, pairs: list[tuple[str, str]], *, batch: int = MAX_QUESTIONS_PER_CALL,
) -> JevOutcome | None:
    """Value maps each pair to the probability that both names cover the same broad topic."""
    probs: dict[tuple[str, str], float] = {}
    responses: list[JevResponse] = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        state = {"pairs": {str(start + i): [a, b] for i, (a, b) in enumerate(chunk)}}
        questions = {
            f"same__{start + i}": noul_q(
                f"Do the two tag names in `pairs.{start + i}` refer to the same broad topic, so that "
                f"a search for either should also find content stored under the other?",
                true="both names describe the same domain or subject area, for example two scale-model hobby tags",
                false="the names share a word or spelling but describe different subjects",
            )
            for i in range(len(chunk))
        }
        resp = client.ask(seam="tag_consolidation", state=state, questions=questions)
        if resp is None:
            return None
        for i, pair in enumerate(chunk):
            ans = resp.answers.get(f"same__{start + i}")
            if ans is None or ans.kind != "noul":
                return JevOutcome.fallback("bad_answer", response=resp)
            probs[pair] = float(ans.value)
        responses.append(resp)
    return JevOutcome(value=probs, detail={"pairs": len(pairs)}, response=_merge_responses(responses))


def _group_pairs(groups: list[dict]) -> set[frozenset]:
    out: set[frozenset] = set()
    for g in groups:
        members = sorted({g["canonical"], *g["aliases"]})
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                out.add(frozenset((a, b)))
    return out


def judge_tag_consolidation(
    tags: list[str], *, legacy: Callable[[], list[dict]], runtime: JudgmentRuntime | None = None,
    canonical_rank: Mapping[str, int] | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None, max_pairs: int = 200,
    strict: bool = False,
) -> list[dict]:
    """Return consolidation groups as ``{"canonical", "aliases", "reason"}`` dicts.

    ``strict`` applies in jev mode: a missing or failed model answer raises
    ``JudgmentUnavailable`` instead of silently applying the legacy groups.
    """
    rt = runtime if runtime is not None else current()
    if not rt.enabled_for("tag_consolidation"):
        if strict and rt.mode_for("tag_consolidation") is JudgmentMode.JEV:
            raise JudgmentUnavailable("tag_consolidation: no judgment client configured")
        return legacy()
    pairs = candidate_tag_pairs(tags, limit=max_pairs, embed_fn=embed_fn, strict_embed=strict)
    rank = canonical_rank or {}
    threshold = rt.config.noul_threshold

    def jev(client: JevClient) -> JevOutcome | None:
        if not pairs:
            return JevOutcome(value=[], detail={"pairs": 0, "kept": 0, "groups": 0})
        out = jev_tag_consolidation(client, pairs)
        if out is None or out.fallback_reason:
            return out
        probs = dict(out.value)
        kept = [p for p in pairs if probs[p] >= threshold]
        groups = groups_from_pairs(kept, rank)
        # Closure round: a group joined through a chain (A~B, B~C) may hold
        # pairs no candidate list ever proposed (A~C). Judge those before
        # trusting the chain, so sparse candidates cannot hide a contradiction.
        missing: list[tuple[str, str]] = []
        for g in groups:
            members = sorted({g["canonical"], *g["aliases"]})
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if (a, b) not in probs:
                        missing.append((a, b))
        closure_resp = None
        if missing:
            closure = jev_tag_consolidation(client, missing)
            if closure is None or closure.fallback_reason:
                return closure
            probs.update(closure.value)
            closure_resp = closure.response
        accepted: list[dict] = []
        conflicted = 0
        for g in groups:
            members = {g["canonical"], *g["aliases"]}
            inside = [(p, pr) for p, pr in probs.items() if p[0] in members and p[1] in members]
            # A judged pair the model rejected must not be unified by transitivity
            # through its neighbours; the whole group is set aside instead.
            if any(pr < threshold for _, pr in inside):
                conflicted += 1
                logger.info("JUDGMENT_CONFLICT seam=tag_consolidation group=%s rejected_pairs=%d",
                            ",".join(sorted(members)), sum(1 for _, pr in inside if pr < threshold))
                continue
            ps = [pr for _, pr in inside]
            g["reason"] = f"jev: same broad topic (p={round(min(ps), 3)})"
            accepted.append(g)
        return JevOutcome(value=accepted, detail={"pairs": len(pairs), "kept": len(kept), "groups": len(accepted),
                                                  "conflicted": conflicted, "closure_pairs": len(missing)},
                          response=_merge_responses([r for r in (out.response, closure_resp) if r is not None]))

    if strict and rt.mode_for("tag_consolidation") is JudgmentMode.JEV:
        assert rt.client is not None
        outcome = _run_jev("tag_consolidation", jev, rt.client)
        if outcome is None or outcome.fallback_reason:
            raise JudgmentUnavailable(
                f"tag_consolidation: {outcome.fallback_reason if outcome else 'jev_unavailable'}",
            )
        return outcome.value

    return decide(
        "tag_consolidation", legacy, jev, runtime=rt,
        agree=lambda a, b: _group_pairs(a) == _group_pairs(b),
        describe=lambda gs: ";".join(f"{g['canonical']}<-{'+'.join(g['aliases'])}" for g in gs) or "-",
    )


# --- seam S9: fact curation --------------------------------------------------

def jev_fact_curation(
    client: JevClient, question: str, facts: list[str], *, batch: int | None = None,
) -> JevOutcome | None:
    """Value maps each fact index to the probability that it could help answer the question.

    Every fact is judged in one call unless *batch* splits them.
    """
    batch = batch or max(1, len(facts))
    probs: dict[int, float] = {}
    responses: list[JevResponse] = []
    for start in range(0, len(facts), batch):
        chunk = facts[start:start + batch]
        state = {"question": question, "facts": {str(start + i): text for i, text in enumerate(chunk)}}
        questions = {
            f"rel__{start + i}": noul_q(
                f"Could `facts.{start + i}` help answer `question`, even tangentially?",
                true="the fact bears on the question's subject, on the people or things it names, "
                     "or on context needed to answer it",
                false="the fact concerns an unrelated matter and could not inform the answer",
            )
            for i in range(len(chunk))
        }
        resp = client.ask(seam="fact_curation", state=state, questions=questions)
        if resp is None:
            return None
        for i in range(len(chunk)):
            ans = resp.answers.get(f"rel__{start + i}")
            if ans is None or ans.kind != "noul":
                return JevOutcome.fallback("bad_answer", response=resp)
            probs[start + i] = float(ans.value)
        responses.append(resp)
    return JevOutcome(value=probs, detail={"n": len(facts)}, response=_merge_responses(responses))


def judge_fact_curation(
    question: str, facts: list[str], *, legacy: Callable[[], list[int] | None],
    runtime: JudgmentRuntime | None = None,
) -> list[int] | None:
    """Return the indices of ``facts`` worth keeping for ``question``.

    A judged choice may be empty (nothing relevant). ``None`` comes only from
    *legacy* and means no choice was made.
    """
    rt = runtime if runtime is not None else current()
    if not facts:
        return legacy()
    floor = rt.config.curation_min_probability

    def jev(client: JevClient) -> JevOutcome | None:
        out = jev_fact_curation(client, question, facts)
        if out is None or out.fallback_reason:
            return out
        keep = [i for i in range(len(facts)) if out.value[i] >= floor]
        return JevOutcome(value=keep, detail={"n": len(facts), "kept": len(keep)}, response=out.response)

    return decide("fact_curation", legacy, jev, runtime=rt,
                  agree=lambda a, b: set(a or []) == set(b or []),
                  describe=lambda idx: ",".join(map(str, idx or [])) or "-")


# --- seam S10: tag split -----------------------------------------------------

def jev_tag_split(
    client: JevClient, tag: str, turns: list[str], *, threshold: float, max_chars: int = 60_000,
) -> JevOutcome | None:
    bounded: list[str] = []
    used = 0
    truncated = False
    for text in turns:
        if used + len(text) > max_chars:
            truncated = True
            break
        bounded.append(text)
        used += len(text)
    resp = client.ask(
        seam="tag_split",
        state={"tag": tag, "turns": bounded},
        questions={"multi_topic": noul_q(
            "Do the `turns`, all currently filed under the single tag `tag`, cover two or more "
            "clearly distinct sub-topics that a reader would want to retrieve separately?",
            true="the turns split into at least two coherent groups about different subjects",
            false="the turns are about one subject, or vary only in incidental detail within a single topic",
        )},
    )
    if resp is None:
        return None
    ans = resp.answers.get("multi_topic")
    if ans is None or ans.kind != "noul":
        return JevOutcome.fallback("bad_answer", response=resp)
    return JevOutcome(value=bool(ans.value >= threshold),
                      detail={"p": round(ans.value, 3), "turns": len(bounded), "truncated": truncated},
                      response=resp)


def judge_tag_split(
    tag: str, turns: list[str], *, legacy: Callable[[], bool], runtime: JudgmentRuntime | None = None,
) -> bool:
    """True when the turns under ``tag`` span more than one topic."""
    rt = runtime if runtime is not None else current()
    return decide("tag_split", legacy,
                  lambda c: jev_tag_split(c, tag, turns, threshold=rt.config.noul_threshold),
                  runtime=rt, describe=str)


# --- seam S11: summary grounding ---------------------------------------------

def jev_summary_grounding(
    client: JevClient, summary: str, source: str, *, threshold: float, max_state_bytes: int,
) -> JevOutcome | None:
    state = {"summary": summary, "source": source}
    size = _payload_bytes(state)
    if size > max_state_bytes:
        logger.info("JUDGMENT_SKIP seam=summary_grounding reason=state_too_large bytes=%d max_bytes=%d",
                    size, max_state_bytes)
        return JevOutcome.fallback("state_too_large")
    resp = client.ask(
        seam="summary_grounding",
        state=state,
        questions={"grounded": noul_q(
            "Is every claim in `summary` supported by `source`, with the right people, polarity, and specifics?",
            true="each statement in the summary can be traced to the source: no invented facts, no swapped "
                 "speakers, no reversed negation or intent, nothing imported from outside the source",
            false="the summary asserts something the source does not say, attributes a statement to the "
                  "wrong person, inverts a negation or intent, or describes material absent from the source",
        )},
    )
    if resp is None:
        return None
    ans = resp.answers.get("grounded")
    if ans is None or ans.kind != "noul":
        return JevOutcome.fallback("bad_answer", response=resp)
    return JevOutcome(value=bool(ans.value >= threshold), detail={"p": round(ans.value, 3), "bytes": size},
                      response=resp)


def judge_summary_grounding(
    summary: str, source: str, *, legacy: Callable[[], bool], runtime: JudgmentRuntime | None = None,
) -> bool:
    """True when ``summary`` is grounded in ``source``."""
    rt = runtime if runtime is not None else current()
    return decide(
        "summary_grounding", legacy,
        lambda c: jev_summary_grounding(c, summary, source, threshold=rt.config.noul_threshold,
                                        max_state_bytes=rt.config.grounding_max_state_bytes),
        runtime=rt, describe=str,
    )
