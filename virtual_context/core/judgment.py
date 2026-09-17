"""Typed-judgment layer: route selected engine decisions through TypeSafe Jev.

Three modes (``JudgmentConfig.mode`` / ``VC_JUDGMENT_MODE``):

* ``legacy`` - Jev is never called; every seam behaves exactly as before.
* ``shadow`` - the legacy answer is used; the Jev answer is logged as
  ``JUDGMENT_SHADOW`` for comparison.
* ``jev`` - the Jev answer is used; any failure falls back to legacy and logs
  ``JUDGMENT_FALLBACK``.

The runtime is installed process-wide by the engine at construction because
the seams are module-level functions called far from any engine handle.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import httpx

from ..types import JUDGMENT_MODES, JUDGMENT_SEAMS, JudgmentConfig

logger = logging.getLogger(__name__)

ENV_MODE = "VC_JUDGMENT_MODE"
T = TypeVar("T")


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
        resp = self._client().post(
            self.config.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.config.timeout_s,
        )
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
    logger.info("JUDGMENT_RUNTIME mode=%s model=%s seams=%s", runtime.mode.value, runtime.config.model,
                ",".join(f"{k}:{v.value}" for k, v in sorted(runtime.seam_modes.items())) or "-")


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


def judge_query_intent(query: str, legacy: Callable[[], str]) -> str:
    return decide("query_intent", legacy, lambda c: jev_query_intent(c, query), describe=str)


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


def judge_temporal_intent(message: str, legacy: Callable[[], bool]) -> bool:
    rt = current()
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


def judge_safety_critical(text: str, legacy: Callable[[], bool]) -> bool:
    rt = current()
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


def judge_admission(payload: dict, eligible: list[str]) -> AdmissionJudgment | None:
    """Return the Jev admission judgment in shadow or jev mode; None in legacy or on failure."""
    rt = current()
    if not rt.enabled_for("admission"):
        return None
    assert rt.client is not None
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
