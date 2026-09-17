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

from ..types import JUDGMENT_MODES, JudgmentConfig

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

    @property
    def enabled(self) -> bool:
        return self.mode is not JudgmentMode.LEGACY and self.client is not None


def build_runtime(
    config: JudgmentConfig,
    *,
    environ: Mapping[str, str] | None = None,
    http_client: httpx.Client | None = None,
) -> JudgmentRuntime:
    mode = JudgmentMode.resolve(config.mode, environ=environ)
    client = None
    if mode is not JudgmentMode.LEGACY:
        client = JevClient(config, http_client=http_client, environ=environ)
    return JudgmentRuntime(mode=mode, client=client, config=config)


_LEGACY_RUNTIME = JudgmentRuntime(JudgmentMode.LEGACY, None, JudgmentConfig())
_installed: JudgmentRuntime = _LEGACY_RUNTIME
_lock = threading.Lock()


def install(runtime: JudgmentRuntime) -> None:
    global _installed
    with _lock:
        _installed = runtime
    logger.info("JUDGMENT_RUNTIME mode=%s model=%s", runtime.mode.value, runtime.config.model)


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
    if not rt.enabled:
        return legacy()
    assert rt.client is not None
    if rt.mode is JudgmentMode.SHADOW:
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
