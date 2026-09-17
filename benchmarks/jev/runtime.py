from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from virtual_context.core.judgment import JevClient, JudgmentMode, JudgmentRuntime
from virtual_context.types import JudgmentConfig

CACHE_DIR = Path(__file__).parent / "cache"


class CachingJevClient(JevClient):
    """Caches every request body by SHA-256 so reruns never re-bill."""

    def __init__(self, config: JudgmentConfig, *, cache_dir: Path = CACHE_DIR, **kw) -> None:
        super().__init__(config, **kw)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.calls: list[tuple[float, int, bool]] = []  # (elapsed_ms, input_tokens, cached)

    def _post(self, body: dict) -> dict:
        key = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            data = json.loads(path.read_text())
            self.calls.append((0.0, int((data.get("usage") or {}).get("input_tokens", 0) or 0), True))
            return data
        started = time.monotonic()
        data = super()._post(body)
        elapsed = (time.monotonic() - started) * 1000.0
        self.misses += 1
        self.calls.append((elapsed, int((data.get("usage") or {}).get("input_tokens", 0) or 0), False))
        path.write_text(json.dumps(data))
        return data


class FakeJevClient(JevClient):
    """Offline stand-in for smoke tests: noul 0.5, choice = first criterion key."""

    def _post(self, body: dict) -> dict:
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.5}
            elif q["type"] == "choice":
                first = next(iter(q["criteria"]))
                answers[key] = {"type": "choice", "choice": first, "confidence": 1.0,
                                "probabilities": {k: (1.0 if k == first else 0.0) for k in q["criteria"]}}
            else:
                answers[key] = {"type": "score", "score": 0.0, "confidence": 1.0, "probabilities": {"0": 1.0},
                                "legend": {"0": q["criteria"][0]}}
        return {"model": "fake", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


def build_live_runtime(cache_dir: Path = CACHE_DIR, *, mode: str = "jev", timeout_s: float = 30.0) -> JudgmentRuntime:
    cfg = JudgmentConfig(mode=mode, timeout_s=timeout_s)
    if not os.environ.get(cfg.api_key_env):
        raise SystemExit(f"{cfg.api_key_env} is not set; export it before running the harness")
    return JudgmentRuntime(JudgmentMode(mode), CachingJevClient(cfg, cache_dir=cache_dir), cfg)


def build_fake_runtime() -> JudgmentRuntime:
    cfg = JudgmentConfig(mode="jev")
    return JudgmentRuntime(JudgmentMode.JEV, FakeJevClient(cfg, environ={"TYPESAFE_API_KEY": "fake"}), cfg)
