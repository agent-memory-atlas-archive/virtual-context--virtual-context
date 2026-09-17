"""Labeled predicate areas: intent (S2), temporal (S3), safety (S4)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from virtual_context.core.judgment import (
    JudgmentRuntime, jev_query_intent, jev_safety_critical, jev_temporal_intent,
)
from virtual_context.core.quote_search import _detect_query_intent_legacy
from virtual_context.core.structured_summary import is_safety_critical_personal_evidence_legacy
from virtual_context.core.tag_generator import detect_temporal_heuristic
from virtual_context.patterns import DEFAULT_TEMPORAL_PATTERNS

DATA_DIR = Path(__file__).parent / "data"
LONGMEMEVAL_500 = Path(__file__).resolve().parents[1] / "longmemeval" / "data" / "longmemeval_s_cleaned.json.original_500q"
_TEMPORAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DEFAULT_TEMPORAL_PATTERNS]


def _load_rows(area: str) -> list[dict]:
    rows = []
    with open(DATA_DIR / f"{area}.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _weak_rows(area: str, dataset_path: Path | None) -> list[dict]:
    path = dataset_path or LONGMEMEVAL_500
    if area == "safety" or not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for q in data:
        qt = q["question_type"]
        if area == "intent":
            label = "current_state" if qt == "knowledge-update" else "default"
        else:
            label = qt == "temporal-reasoning"
        rows.append({"id": f"lme:{q['question_id']}", "text": q["question"], "label": label, "weak": True,
                     "source": f"longmemeval:{qt}"})
    return rows


def _legacy(area: str, text: str):
    if area == "intent":
        return _detect_query_intent_legacy(text)
    if area == "temporal":
        return detect_temporal_heuristic(text, _TEMPORAL_PATTERNS)
    return is_safety_critical_personal_evidence_legacy(text)


def _jev(area: str, client, text: str, threshold: float):
    if area == "intent":
        return jev_query_intent(client, text, min_confidence=0.0), jev_query_intent(client, text, min_confidence=0.5)
    if area == "temporal":
        out = jev_temporal_intent(client, text, threshold=threshold)
        return out, out
    out = jev_safety_critical(client, text, threshold=threshold)
    return out, out


def _score(rows: list[dict], key: str) -> dict:
    def acc(subset):
        return (sum(1 for r in subset if r[key] == r["label"]) / len(subset)) if subset else None
    strong = [r for r in rows if not r.get("weak")]
    weak = [r for r in rows if r.get("weak")]
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        confusion.setdefault(str(r["label"]), {}).setdefault(str(r[key]), 0)
        confusion[str(r["label"])][str(r[key])] += 1
    return {"accuracy_all": acc(rows), "accuracy_strong": acc(strong), "accuracy_weak": acc(weak), "confusion": confusion}


def run_area(area: str, runtime: JudgmentRuntime, *, limit: int | None = None, dataset_path: Path | None = None) -> dict:
    assert area in ("intent", "temporal", "safety")
    rows = _load_rows(area) + _weak_rows(area, dataset_path)
    if limit is not None:
        rows = rows[:limit]
    client = runtime.client
    assert client is not None
    threshold = runtime.config.noul_threshold
    latencies, tokens = [], []
    for r in rows:
        text = r["text"]
        r["legacy"] = _legacy(area, text)
        raw, deployed = _jev(area, client, text, threshold)
        if raw is None or raw.fallback_reason:
            r["jev_raw"] = None
        else:
            r["jev_raw"] = raw.value
            r["jev_p"] = raw.detail.get("p")
            if raw.response is not None:
                latencies.append(raw.response.latency_ms)
                tokens.append(raw.response.input_tokens)
        r["jev_deployed"] = deployed.value if (deployed is not None and not deployed.fallback_reason) else r["legacy"]
    return {
        "area": area, "n": len(rows),
        "n_strong": sum(1 for r in rows if not r.get("weak")), "n_weak": sum(1 for r in rows if r.get("weak")),
        "legacy": _score(rows, "legacy"), "jev_raw": _score(rows, "jev_raw"), "jev_deployed": _score(rows, "jev_deployed"),
        "rows": rows,
        "jev_latency_ms": mean(latencies) if latencies else 0.0,
        "jev_tokens": mean(tokens) if tokens else 0.0,
    }
