"""Seams S6-S11: tag reuse, supersession, tag consolidation, fact curation,
tag split, summary grounding, plus the admission state cap."""
import json
import logging

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import (
    SUPERSESSION_CRITERIA,
    build_runtime,
    candidate_tag_pairs,
    groups_from_pairs,
    judge_fact_curation,
    judge_fact_links,
    judge_summary_grounding,
    judge_supersession,
    judge_tag_consolidation,
    judge_tag_reuse,
    judge_tag_split,
    nearest_existing_tags,
    trim_admission_payload,
)
from virtual_context.types import JUDGMENT_SEAMS, FactLink, JudgmentConfig


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _runtime(mode, answers_fn, **cfg):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"model": "jev-t", "answers": answers_fn(body),
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    rt = build_runtime(JudgmentConfig(mode=mode, **cfg), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    return rt, seen


def _choice(key, choice, confidence, options):
    return {key: {"type": "choice", "choice": choice, "confidence": confidence,
                  "probabilities": {o: (confidence if o == choice else (1 - confidence) / max(1, len(options) - 1)) for o in options}}}


def _noul(key, p):
    return {key: {"type": "noul", "noul": p}}


def _answers(*parts):
    out = {}
    for part in parts:
        out.update(part)
    return out


# --- config ------------------------------------------------------------------

def test_new_seams_are_registered_and_parsed():
    for seam in ("tag_reuse", "supersession", "tag_consolidation", "fact_curation", "tag_split", "summary_grounding"):
        assert seam in JUDGMENT_SEAMS
    from virtual_context.config import _parse_judgment
    cfg = _parse_judgment({
        "mode": "legacy",
        "seams": {"tag_reuse": "shadow", "supersession": "shadow", "summary_grounding": "jev"},
        "admission_max_state_bytes": 5000, "tag_reuse_candidates": 5,
        "curation_min_probability": 0.25, "grounding_max_state_bytes": 7000,
    })
    assert cfg.seams == {"tag_reuse": "shadow", "supersession": "shadow", "summary_grounding": "jev"}
    assert (cfg.admission_max_state_bytes, cfg.tag_reuse_candidates, cfg.curation_min_probability,
            cfg.grounding_max_state_bytes) == (5000, 5, 0.25, 7000)
    rt = build_runtime(cfg, environ={"TYPESAFE_API_KEY": "k"})
    assert rt.mode_for("summary_grounding") is judgment.JudgmentMode.JEV
    assert rt.mode_for("rerank") is judgment.JudgmentMode.LEGACY


# --- S6 tag reuse ------------------------------------------------------------

def test_nearest_existing_tags_ranks_string_similarity_and_caps():
    existing = ["data-visualization", "database-migrations", "deployment", "model-kit", "visualization-libraries"]
    ranked = nearest_existing_tags("data-visualization-tools", existing, limit=3)
    assert ranked[0] == "data-visualization"
    assert len(ranked) == 3
    assert "data-visualization-tools" not in nearest_existing_tags("data-visualization-tools", existing + ["data-visualization-tools"], limit=3)


def test_tag_reuse_legacy_keeps_every_proposed_tag_new():
    assert judge_tag_reuse("text", ["new-tag"], {"new-tag": ["old-tag"]}, runtime=judgment.current()) == {"new-tag": None}


def test_tag_reuse_shadow_logs_and_returns_all_new(caplog):
    rt, seen = _runtime("shadow", lambda b: _choice("reuse__0", "data-visualization", 0.92, ["data-visualization", "none"]))
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        out = judge_tag_reuse("charts about sales", ["data-visualization-tools"],
                              {"data-visualization-tools": ["data-visualization"]}, runtime=rt)
    assert out == {"data-visualization-tools": None}
    assert seen[0]["state"]["proposed_tags"] == {"0": "data-visualization-tools"}
    assert seen[0]["state"]["existing_tags"] == {"0": ["data-visualization"]}
    assert set(seen[0]["questions"]["reuse__0"]["criteria"]) == {"data-visualization", "none"}
    assert any("JUDGMENT_SHADOW seam=tag_reuse agree=False" in r.getMessage() for r in caplog.records)


def test_tag_reuse_jev_maps_to_existing_when_confident_else_new():
    rt, _ = _runtime("jev", lambda b: _answers(
        _choice("reuse__0", "data-visualization", 0.9, ["data-visualization", "none"]),
        _choice("reuse__1", "deployment", 0.4, ["deployment", "none"]),
        _choice("reuse__2", "none", 0.95, ["model-kit", "none"]),
    ))
    out = judge_tag_reuse("t", ["data-visualization-tools", "deploy-notes", "unrelated-topic"],
                          {"data-visualization-tools": ["data-visualization"], "deploy-notes": ["deployment"],
                           "unrelated-topic": ["model-kit"]}, runtime=rt)
    assert out == {"data-visualization-tools": "data-visualization", "deploy-notes": None, "unrelated-topic": None}


def test_tag_reuse_jev_bad_answer_falls_back_to_new():
    rt, _ = _runtime("jev", lambda b: _choice("reuse__0", "not-an-option", 0.9, ["x", "none"]))
    assert judge_tag_reuse("t", ["p"], {"p": ["x"]}, runtime=rt) == {"p": None}


# --- S7 supersession ---------------------------------------------------------

def test_supersession_jev_marks_supersedes_duplicates_and_contradicts():
    opts = list(SUPERSESSION_CRITERIA)
    rt, seen = _runtime("jev", lambda b: _answers(
        _choice("rel__f1", "supersedes", 0.9, opts), _choice("rel__f2", "independent", 0.8, opts),
        _choice("rel__f3", "duplicates", 0.85, opts), _choice("rel__f4", "contradicts", 0.7, opts),
    ))
    cands = [("f1", "user | lives_in | Boston", "2024/01/01"), ("f2", "user | ran | marathon", ""),
             ("f3", "user | lives in | Boston MA", ""), ("f4", "user | lives_in | Austin", "")]
    got = judge_supersession("user | lives_in | Seattle", cands, legacy=lambda: ["f9"], runtime=rt)
    assert got == ["f1", "f3", "f4"]
    assert seen[0]["state"]["new_fact"] == "user | lives_in | Seattle"
    assert seen[0]["state"]["candidates"]["f1"]["session_date"] == "2024/01/01"
    assert set(seen[0]["questions"]["rel__f1"]["criteria"]) == set(opts)


def test_supersession_shadow_returns_legacy_and_logs_set_agreement(caplog):
    opts = list(SUPERSESSION_CRITERIA)
    rt, _ = _runtime("shadow", lambda b: _choice("rel__f1", "supersedes", 0.9, opts))
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        got = judge_supersession("n", [("f1", "c", "")], legacy=lambda: ["f1"], runtime=rt)
    assert got == ["f1"]
    assert any("JUDGMENT_SHADOW seam=supersession agree=True" in r.getMessage() for r in caplog.records)


def test_supersession_jev_low_confidence_relation_counts_as_independent():
    opts = list(SUPERSESSION_CRITERIA)
    rt, _ = _runtime("jev", lambda b: _choice("rel__f1", "supersedes", 0.4, opts))
    assert judge_supersession("n", [("f1", "c", "")], legacy=lambda: ["f1"], runtime=rt) == []


def test_fact_links_jev_builds_links_with_mapped_relations():
    opts = list(SUPERSESSION_CRITERIA)
    rt, _ = _runtime("jev", lambda b: _answers(
        _choice("rel__f1", "supersedes", 0.9, opts), _choice("rel__f2", "duplicates", 0.9, opts),
        _choice("rel__f3", "contradicts", 0.9, opts), _choice("rel__f4", "independent", 0.9, opts)))
    links, superseded = judge_fact_links("n0", "user | x | y",
                                         [("f1", "a", ""), ("f2", "b", ""), ("f3", "c", ""), ("f4", "d", "")],
                                         legacy=lambda: ([], []), runtime=rt)
    assert superseded == ["f1", "f2", "f3"]
    rel = {(l.source_fact_id, l.target_fact_id): l.relation_type for l in links}
    assert rel == {("n0", "f1"): "supersedes", ("n0", "f2"): "same_as", ("n0", "f3"): "contradicts"}
    assert all(isinstance(l, FactLink) and l.created_by == "supersession" for l in links)


# --- S8 tag consolidation ----------------------------------------------------

def test_candidate_tag_pairs_finds_lexical_neighbors_only():
    pairs = candidate_tag_pairs(["model-kit", "model-tanks", "data-model", "deployment", "deploy-notes"], limit=10)
    assert ("model-kit", "model-tanks") in pairs
    assert ("deploy-notes", "deployment") in pairs
    assert ("data-model", "deployment") not in pairs


def test_groups_from_pairs_unions_and_picks_highest_ranked_canonical():
    groups = groups_from_pairs([("a", "b"), ("b", "c"), ("x", "y")], {"a": 3, "b": 10, "c": 1, "x": 2, "y": 2})
    as_map = {g["canonical"]: sorted(g["aliases"]) for g in groups}
    assert as_map == {"b": ["a", "c"], "x": ["y"]}


def test_tag_consolidation_jev_groups_from_pairwise_nouls():
    def answers(body):
        out = {}
        for key, q in body["questions"].items():
            pair = body["state"]["pairs"][key.split("__", 1)[1]]
            out.update(_noul(key, 0.9 if set(pair) == {"model-kit", "model-tanks"} else 0.1))
        return out
    rt, seen = _runtime("jev", answers)
    groups = judge_tag_consolidation(["model-kit", "model-tanks", "data-model"], legacy=lambda: [],
                                     runtime=rt, canonical_rank={"model-kit": 5, "model-tanks": 2, "data-model": 9})
    assert groups == [{"canonical": "model-kit", "aliases": ["model-tanks"], "reason": "jev: same broad topic (p=0.9)"}]
    assert all(k.startswith("same__") for k in seen[0]["questions"])


def test_tag_consolidation_shadow_returns_legacy_groups(caplog):
    rt, _ = _runtime("shadow", lambda b: {k: {"type": "noul", "noul": 0.2} for k in b["questions"]})
    legacy_groups = [{"canonical": "model-kit", "aliases": ["model-tanks"], "reason": "llm"}]
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        got = judge_tag_consolidation(["model-kit", "model-tanks"], legacy=lambda: legacy_groups, runtime=rt,
                                      canonical_rank={})
    assert got == legacy_groups
    assert any("JUDGMENT_SHADOW seam=tag_consolidation agree=False" in r.getMessage() for r in caplog.records)


# --- S9 fact curation --------------------------------------------------------

def test_fact_curation_jev_keeps_facts_above_inclusive_floor():
    rt, seen = _runtime("jev", lambda b: _answers(_noul("rel__0", 0.9), _noul("rel__1", 0.35), _noul("rel__2", 0.05)),
                        curation_min_probability=0.3)
    got = judge_fact_curation("where do I live?", ["user | lives_in | Seattle", "user | likes | rain", "user | ran | 5k"],
                              legacy=lambda: [0], runtime=rt)
    assert got == [0, 1]
    assert seen[0]["state"]["question"] == "where do I live?"
    assert seen[0]["state"]["facts"]["2"] == "user | ran | 5k"


def test_fact_curation_shadow_returns_legacy(caplog):
    rt, _ = _runtime("shadow", lambda b: _answers(_noul("rel__0", 0.9)))
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        assert judge_fact_curation("q", ["f"], legacy=lambda: [], runtime=rt) == []
    assert any("JUDGMENT_SHADOW seam=fact_curation agree=False" in r.getMessage() for r in caplog.records)


# --- S10 tag split -----------------------------------------------------------

def test_tag_split_jev_and_shadow():
    rt, seen = _runtime("jev", lambda b: _noul("multi_topic", 0.8))
    assert judge_tag_split("cooking", ["[T1] pasta", "[T2] car repair"], legacy=lambda: False, runtime=rt) is True
    assert seen[0]["state"]["tag"] == "cooking"
    rt2, _ = _runtime("shadow", lambda b: _noul("multi_topic", 0.8))
    assert judge_tag_split("cooking", ["[T1] pasta"], legacy=lambda: False, runtime=rt2) is False


# --- S11 summary grounding ---------------------------------------------------

def test_summary_grounding_jev_uses_noul_and_shadow_returns_legacy():
    rt, seen = _runtime("jev", lambda b: _noul("grounded", 0.2))
    assert judge_summary_grounding("Alice moved to Paris.", "Alice said she may visit Paris.", legacy=lambda: True, runtime=rt) is False
    assert seen[0]["state"] == {"summary": "Alice moved to Paris.", "source": "Alice said she may visit Paris."}
    rt2, _ = _runtime("shadow", lambda b: _noul("grounded", 0.2))
    assert judge_summary_grounding("s", "src", legacy=lambda: True, runtime=rt2) is True


def test_summary_grounding_skips_oversized_state(caplog):
    rt, seen = _runtime("jev", lambda b: _noul("grounded", 0.2), grounding_max_state_bytes=200)
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        assert judge_summary_grounding("s", "x" * 1000, legacy=lambda: True, runtime=rt) is True
    assert seen == []
    assert any("JUDGMENT_SKIP seam=summary_grounding reason=state_too_large" in r.getMessage() for r in caplog.records)


# --- admission state cap -----------------------------------------------------

def test_trim_admission_payload_drops_evidence_then_old_turns():
    payload = {
        "as_of": "2026-09-20T00:00:00+00:00", "curator_substantive_claim": True,
        "candidates": [{"candidate_id": "c1", "body": "x"}], "facts": [],
        "actor_turns": [{"id": f"t{i}", "content": "y" * 1300} for i in range(10)],
        "evidence_segments": [{"ref": f"s{i}", "text": "z" * 2000} for i in range(5)],
    }
    trimmed, detail = trim_admission_payload(payload, 5000)
    assert trimmed is not None
    assert len(json.dumps(trimmed, separators=(",", ":"))) <= 5000
    assert trimmed["evidence_segments"] == []
    assert [t["id"] for t in trimmed["actor_turns"]] == ["t7", "t8", "t9"]
    assert detail["dropped_evidence"] == 5 and detail["dropped_turns"] == 7
    assert trimmed["candidates"] == payload["candidates"]
    untouched, detail2 = trim_admission_payload(payload, 10 ** 6)
    assert untouched is payload and detail2 == {}
    assert trim_admission_payload({"candidates": [{"body": "q" * 9000}], "actor_turns": [], "evidence_segments": []}, 5000)[0] is None
