"""S5 admission harness: legacy admission model vs Jev on labeled candidate sets."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean

from virtual_context.core.community.actor_card_admission import build_admission_request, parse_admission_response
from virtual_context.core.judgment import JudgmentRuntime, jev_admission
from virtual_context.core.llm_utils import parse_llm_json

DATA = Path(__file__).parent / "data" / "admission.jsonl"
DEFAULT_LEGACY_MODEL = "qwen/qwen3-235b-a22b-2507"


def load_sets() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def legacy_arm_label() -> str:
    provider = os.environ.get("VC_JEV_ADMISSION_PROVIDER", "openrouter")
    model = os.environ.get("VC_JEV_ADMISSION_MODEL", DEFAULT_LEGACY_MODEL if provider == "openrouter" else "k3")
    thinking = os.environ.get("VC_JEV_ADMISSION_THINKING", "off")
    return f"{provider}:{model}" + (f":thinking-{thinking}" if provider == "kimi" else "")


def _legacy_provider():
    """The model side of the comparison: OpenRouter (default) or the Kimi Code endpoint.

    VC_JEV_ADMISSION_PROVIDER=openrouter|kimi, VC_JEV_ADMISSION_MODEL, and for kimi
    VC_JEV_ADMISSION_THINKING=on|off (default off).
    """
    provider = os.environ.get("VC_JEV_ADMISSION_PROVIDER", "openrouter")
    if provider == "kimi":
        from virtual_context.providers.anthropic import AnthropicProvider
        key = os.environ.get("KIMI_API_KEY")
        if not key:
            raise SystemExit("KIMI_API_KEY is not set; needed for the Kimi Code admission model")
        return AnthropicProvider(
            api_key=key,
            model=os.environ.get("VC_JEV_ADMISSION_MODEL", "k3"),
            temperature=0.0,
            base_url="https://api.kimi.com/coding",
            disable_thinking=os.environ.get("VC_JEV_ADMISSION_THINKING", "off") != "on",
        )
    from virtual_context.providers.generic_openai import GenericOpenAIProvider
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set; needed for the legacy admission model")
    return GenericOpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        model=os.environ.get("VC_JEV_ADMISSION_MODEL", DEFAULT_LEGACY_MODEL),
        api_key=key,
        temperature=0.0,
    )


def _score(rows: list[dict], side: str) -> dict:
    cands = [c for r in rows for c in r["candidates_scored"] if c.get(side) is not None]
    if not cands:
        return {"reason_accuracy": None, "admit_accuracy": None, "coverage_accuracy": None, "mean_ms": 0.0, "mean_tokens": 0.0}
    sets = [r for r in rows if r.get(f"{side}_coverage") is not None]
    return {
        "reason_accuracy": sum(1 for c in cands if c[side] == c["expected"]) / len(cands),
        "admit_accuracy": sum(1 for c in cands if (c[side] == "durable") == (c["expected"] == "durable")) / len(cands),
        "coverage_accuracy": (sum(1 for r in sets if r[f"{side}_coverage"] == r["expected"]["coverage_reason"]) / len(sets)) if sets else None,
        "mean_ms": mean(r[f"{side}_ms"] for r in sets) if sets else 0.0,
        "mean_tokens": mean(r[f"{side}_tokens"] for r in sets) if sets else 0.0,
    }


def run_admission(runtime: JudgmentRuntime, *, limit: int | None = None, offline: bool = False) -> dict:
    sets = load_sets()
    if limit is not None:
        sets = sets[:limit]
    provider = None if offline else _legacy_provider()
    client = runtime.client
    assert client is not None
    rows = []
    for s in sets:
        eligible = [c["candidate_id"] for c in s["candidates"]]
        req = build_admission_request(candidates=s["candidates"], compact_facts=s["facts"], actor_turns=s["actor_turns"],
                                      evidence_segments=s["evidence_segments"], curator_substantive=s["curator_substantive"],
                                      as_of="2026-09-16T00:00:00+00:00")
        row = {"id": s["id"], "expected": s["expected"], "candidates_scored": [
            {"candidate_id": cid, "expected": s["expected"]["decisions"][cid]} for cid in eligible]}
        if provider is not None:
            started = time.monotonic()
            text, usage = provider.complete(system=req["system"], user=req["user"], max_tokens=req["max_tokens"])
            row["legacy_ms"] = (time.monotonic() - started) * 1000.0
            row["legacy_tokens"] = int((usage or {}).get("input_tokens") or (usage or {}).get("prompt_tokens") or 0)
            row["legacy_raw"] = text[:2000]
            try:
                substantive, decisions = parse_admission_response(text, parse_json=parse_llm_json, eligible=eligible)
                parsed = parse_llm_json(text)
                row["legacy_coverage"] = parsed.get("coverage_reason") if isinstance(parsed, dict) else None
                for c in row["candidates_scored"]:
                    c["legacy"] = decisions[c["candidate_id"]]["reason"]
            except Exception as exc:
                row["legacy_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        outcome = jev_admission(client, req["payload"], eligible)
        if outcome is not None and not outcome.fallback_reason:
            row["jev_coverage"] = outcome.value["coverage_reason"]
            row["jev_ms"] = outcome.response.latency_ms if outcome.response else 0.0
            row["jev_tokens"] = outcome.response.input_tokens if outcome.response else 0
            for c, d in zip(row["candidates_scored"], outcome.value["decisions"]):
                c["jev"] = d["reason"]
        rows.append(row)
    both = [c for r in rows for c in r["candidates_scored"] if c.get("legacy") is not None and c.get("jev") is not None]
    return {
        "area": "admission", "legacy_arm": (legacy_arm_label() if provider is not None else None),
        "n_sets": len(rows), "n_candidates": sum(len(r["candidates_scored"]) for r in rows),
        "legacy": _score(rows, "legacy"), "jev": _score(rows, "jev"),
        "agreement": (sum(1 for c in both if c["legacy"] == c["jev"]) / len(both)) if both else None,
        "rows": rows,
    }
