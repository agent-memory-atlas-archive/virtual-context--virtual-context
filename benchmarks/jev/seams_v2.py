"""Labeled areas for the second seam set: tag_reuse, supersession, consolidation,
curation, tag_split, grounding.

The legacy arm is the production prompt path run against a configurable model
(``VC_JEV_LEGACY_MODEL`` through OpenRouter, default the platform tagger and
summarizer model); the grounding legacy arm is the compactor heuristic and needs
no model. The Jev arm calls the seam functions directly. Rows are scored per
decision unit (one proposed tag, one candidate fact, one tag pair, one fact, one
tag, one summary) so the generic report table applies.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.judgment import (
    JudgmentRuntime, jev_fact_curation, jev_summary_grounding, jev_supersession, jev_tag_consolidation,
    jev_tag_reuse, jev_tag_split,
)
from virtual_context.core.tag_consolidator import _CONSOLIDATION_PROMPT, _SYSTEM, _merge_transitive_groups
from virtual_context.core.tag_consolidator import _parse_response as _parse_consolidation
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.core.tag_splitter import TagSplitter
from virtual_context.ingest.curator import FactCurator
from virtual_context.ingest.supersession import FactSupersessionChecker
from virtual_context.types import CurationConfig, Fact, SupersessionConfig, TagGeneratorConfig, TagSplittingConfig

from .labeled import _score

DATA_DIR = Path(__file__).parent / "data"
AREAS = ("tag_reuse", "supersession", "consolidation", "curation", "tag_split", "grounding")
DEFAULT_LEGACY_MODEL = "google/gemini-2.5-flash-lite"


def legacy_arm_label() -> str:
    return "openrouter:" + os.environ.get("VC_JEV_LEGACY_MODEL", DEFAULT_LEGACY_MODEL)


def _legacy_provider():
    from virtual_context.providers.generic_openai import GenericOpenAIProvider
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set; needed for the legacy model arm")
    return GenericOpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        model=os.environ.get("VC_JEV_LEGACY_MODEL", DEFAULT_LEGACY_MODEL),
        api_key=key,
        temperature=0.0,
    )


def _load(area: str) -> list[dict]:
    rows = []
    with open(DATA_DIR / f"{area}.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fact(spec: dict, fact_id: str = "") -> Fact:
    return Fact(
        id=fact_id or spec.get("id", ""), conversation_id="h", subject=spec["subject"], verb=spec["verb"],
        object=spec["object"], session_date=spec.get("session_date", ""),
    )


class _Timing:
    def __init__(self) -> None:
        self.legacy_ms: list[float] = []
        self.jev_ms: list[float] = []
        self.jev_tokens: list[int] = []

    def note(self, outcome) -> None:
        if outcome is not None and not outcome.fallback_reason and outcome.response is not None:
            self.jev_ms.append(outcome.response.latency_ms)
            self.jev_tokens.append(outcome.response.input_tokens)


def _timed(fn, sink: list[float]):
    started = time.monotonic()
    try:
        return fn()
    finally:
        sink.append((time.monotonic() - started) * 1000.0)


# --- tag_reuse: one row per proposed tag --------------------------------------

def _run_tag_reuse(rows, client, provider, threshold, timing):
    tagger = None
    if provider is not None:
        tagger = LLMTagGenerator(provider, TagGeneratorConfig(type="llm", min_tags=1, max_tags=6))
    for r in rows:
        existing = list(r["existing"])
        if tagger is not None:
            result = _timed(lambda: tagger.generate_tags(r["text"], existing_tags=existing), timing.legacy_ms)
            # The production tagger sees the existing tags in its prompt and may emit
            # several of them; it is credited when the labeled tag is among its output.
            reused = [t for t in result.tags if t in set(existing)]
            r["legacy"] = r["label"] if r["label"] in result.tags else (reused[0] if reused else "none")
            r["legacy_tags"] = list(result.tags)
            r["legacy_any_existing"] = bool(reused)
        else:
            r["legacy"] = "none"
        raw = jev_tag_reuse(client, r["text"], [r["proposed"]], {r["proposed"]: existing}, min_confidence=0.0)
        deployed = jev_tag_reuse(client, r["text"], [r["proposed"]], {r["proposed"]: existing}, min_confidence=0.5)
        timing.note(raw)
        r["jev_raw"] = (raw.value[r["proposed"]] or "none") if raw and not raw.fallback_reason else None
        r["jev_p"] = raw.detail.get("p0") if raw and not raw.fallback_reason else None
        r["jev_deployed"] = (deployed.value[r["proposed"]] or "none") if deployed and not deployed.fallback_reason else r["legacy"]
    return rows


# --- supersession: sets expand to one row per candidate -----------------------

def _run_supersession(rows, client, provider, threshold, timing):
    checker = None
    if provider is not None:
        checker = FactSupersessionChecker(provider, os.environ.get("VC_JEV_LEGACY_MODEL", DEFAULT_LEGACY_MODEL),
                                          store=None, config=SupersessionConfig(enabled=True))
    out = []
    for s in rows:
        new = _fact(s["new_fact"], "N0")
        cands = [_fact(c, c["id"]) for c in s["candidates"]]
        legacy_ids = None
        if checker is not None:
            legacy_ids = set(_timed(lambda: checker._llm_check_batch(new, cands), timing.legacy_ms))
        triples = [(c.id, c.format_for_prompt(), c.session_date) for c in cands]
        raw = jev_supersession(client, new.format_for_prompt(), triples, new_fact_date=new.session_date, min_confidence=0.0)
        deployed = jev_supersession(client, new.format_for_prompt(), triples, new_fact_date=new.session_date, min_confidence=0.5)
        timing.note(raw)
        for c in s["candidates"]:
            row = {"id": f"{s['id']}:{c['id']}", "set": s["id"], "candidate": c["id"], "relation": c["label"],
                   "label": c["label"] != "independent"}
            row["legacy"] = (c["id"] in legacy_ids) if legacy_ids is not None else False
            if raw and not raw.fallback_reason:
                rel, p = raw.value[c["id"]]
                row["jev_raw"] = rel != "independent"
                row["jev_relation"] = rel
                row["jev_p"] = round(p, 3)
            else:
                row["jev_raw"] = None
            if deployed and not deployed.fallback_reason:
                row["jev_deployed"] = deployed.value[c["id"]][0] != "independent"
            else:
                row["jev_deployed"] = row["legacy"]
            out.append(row)
    return out


# --- consolidation: one row per tag pair ---------------------------------------

def _run_consolidation(rows, client, provider, threshold, timing):
    pairs = [tuple(r["pair"]) for r in rows]
    same_group: set[frozenset] = set()
    if provider is not None:
        tags = sorted({t for p in pairs for t in p})
        prompt = _CONSOLIDATION_PROMPT.format(tag_list="\n".join(f"- {t}" for t in tags))
        response, _ = _timed(lambda: provider.complete(system=_SYSTEM, user=prompt, max_tokens=4096), timing.legacy_ms)
        groups = _merge_transitive_groups(_parse_consolidation(response))
        for g in groups:
            members = sorted({g.canonical, *g.aliases})
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    same_group.add(frozenset((a, b)))
    outcome = jev_tag_consolidation(client, pairs)
    timing.note(outcome)
    for r, pair in zip(rows, pairs):
        r["legacy"] = frozenset(pair) in same_group
        if outcome and not outcome.fallback_reason:
            p = outcome.value[pair]
            r["jev_p"] = round(p, 3)
            r["jev_raw"] = p >= 0.5
            r["jev_deployed"] = p >= threshold
        else:
            r["jev_raw"] = None
            r["jev_deployed"] = r["legacy"]
    return rows


# --- curation: sets expand to one row per fact ---------------------------------

def _run_curation(rows, client, provider, threshold, timing, *, min_probability: float):
    curator = None
    if provider is not None:
        curator = FactCurator(provider, os.environ.get("VC_JEV_LEGACY_MODEL", DEFAULT_LEGACY_MODEL),
                              CurationConfig(enabled=True))
    out = []
    for s in rows:
        facts = [_fact(f, f"f{i}") for i, f in enumerate(s["facts"])]
        kept = None
        if curator is not None:
            idx = _timed(lambda: curator._llm_curate(facts, s["question"]), timing.legacy_ms)
            kept = set(idx) if idx else set(range(len(facts)))  # empty answer keeps every fact
        outcome = jev_fact_curation(client, s["question"], [f.format_for_prompt() for f in facts])
        timing.note(outcome)
        for i, f in enumerate(s["facts"]):
            row = {"id": f"{s['id']}:f{i}", "set": s["id"], "label": bool(f["label"])}
            row["legacy"] = (i in kept) if kept is not None else True
            if outcome and not outcome.fallback_reason:
                p = outcome.value[i]
                row["jev_p"] = round(p, 3)
                row["jev_raw"] = p >= 0.5
                row["jev_deployed"] = p >= min_probability
            else:
                row["jev_raw"] = None
                row["jev_deployed"] = row["legacy"]
            out.append(row)
    return out


# --- tag_split: one row per tag ------------------------------------------------

def _run_tag_split(rows, client, provider, threshold, timing):
    splitter = TagSplitter(provider, TagSplittingConfig(enabled=True)) if provider is not None else None
    for r in rows:
        contents = [(i + 1, t) for i, t in enumerate(r["turns"])]
        if splitter is not None:
            result = _timed(lambda: splitter.split(r["tag"], contents, set(), len(contents) * 3), timing.legacy_ms)
            r["legacy"] = bool(result.splittable)
            r["legacy_reason"] = result.reason
        else:
            r["legacy"] = False
        lines = [f"[T{n}] {t[:200]}" for n, t in contents]
        outcome = jev_tag_split(client, r["tag"], lines, threshold=threshold)
        timing.note(outcome)
        if outcome and not outcome.fallback_reason:
            r["jev_p"] = outcome.detail["p"]
            r["jev_raw"] = outcome.detail["p"] >= 0.5
            r["jev_deployed"] = bool(outcome.value)
        else:
            r["jev_raw"] = None
            r["jev_deployed"] = r["legacy"]
    return rows


# --- grounding: one row per summary --------------------------------------------

def _run_grounding(rows, client, provider, threshold, timing):
    for r in rows:
        r["legacy"] = DomainCompactor._segment_summary_reject_reason(r["summary"], r["source"], None) is None
        outcome = jev_summary_grounding(client, r["summary"], r["source"], threshold=threshold, max_state_bytes=120_000)
        timing.note(outcome)
        if outcome and not outcome.fallback_reason:
            r["jev_p"] = outcome.detail["p"]
            r["jev_raw"] = outcome.detail["p"] >= 0.5
            r["jev_deployed"] = bool(outcome.value)
        else:
            r["jev_raw"] = None
            r["jev_deployed"] = r["legacy"]
    return rows


def run_area(area: str, runtime: JudgmentRuntime, *, limit: int | None = None, offline: bool = False) -> dict:
    assert area in AREAS
    rows = _load(area)
    if limit is not None:
        rows = rows[:limit]
    client = runtime.client
    assert client is not None
    provider = None if (offline or area == "grounding") else _legacy_provider()
    threshold = runtime.config.noul_threshold
    timing = _Timing()
    if area == "tag_reuse":
        scored = _run_tag_reuse(rows, client, provider, threshold, timing)
    elif area == "supersession":
        scored = _run_supersession(rows, client, provider, threshold, timing)
    elif area == "consolidation":
        scored = _run_consolidation(rows, client, provider, threshold, timing)
    elif area == "curation":
        scored = _run_curation(rows, client, provider, threshold, timing,
                               min_probability=runtime.config.curation_min_probability)
    elif area == "tag_split":
        scored = _run_tag_split(rows, client, provider, threshold, timing)
    else:
        scored = _run_grounding(rows, client, provider, threshold, timing)
    return {
        "area": area, "n": len(scored), "n_strong": len(scored), "n_weak": 0,
        "legacy_arm": ("heuristic" if area == "grounding" else (legacy_arm_label() if provider is not None else None)),
        "legacy": _score(scored, "legacy"), "jev_raw": _score(scored, "jev_raw"), "jev_deployed": _score(scored, "jev_deployed"),
        "rows": scored,
        "legacy_latency_ms": mean(timing.legacy_ms) if timing.legacy_ms else 0.0,
        "jev_latency_ms": mean(timing.jev_ms) if timing.jev_ms else 0.0,
        "jev_tokens": mean(timing.jev_tokens) if timing.jev_tokens else 0.0,
    }
