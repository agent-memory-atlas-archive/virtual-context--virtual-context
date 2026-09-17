"""S1 rerank harness over cached LongMemEval stores.

For each cached question: copy store.db to a scratch dir, build the engine
over it exactly as the benchmark runner does, run retrieval in legacy and in
jev mode, and score the candidate order against the gold answer sessions.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from statistics import mean, median

from benchmarks.longmemeval.vc_runner import DEFAULT_CACHE_DIR, _build_vc_config
from virtual_context.config import load_config
from virtual_context.core import judgment
from virtual_context.core.judgment import JudgmentMode, JudgmentRuntime
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import JudgmentConfig

LONGMEMEVAL_500 = Path(__file__).resolve().parents[1] / "longmemeval" / "data" / "longmemeval_s_cleaned.json.original_500q"
_WS = re.compile(r"\s+")


def normalize_snippet(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def gold_segment_refs(store_path: Path, gold_sessions: list[list[dict]]) -> set[str]:
    snippets = []
    for session in gold_sessions:
        for turn in session:
            snip = normalize_snippet(turn.get("content", ""))[:200]
            if len(snip) >= 40:
                snippets.append(snip)
    if not snippets:
        return set()
    refs: set[str] = set()
    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        for ref, full_text in conn.execute("SELECT ref, full_text FROM segments"):
            hay = normalize_snippet(full_text)
            if any(s in hay for s in snippets):
                refs.add(ref)
    finally:
        conn.close()
    return refs


def first_gold_rank(order: list[str], gold: set[str]) -> int | None:
    for i, ref in enumerate(order, start=1):
        if ref in gold:
            return i
    return None


def _questions(dataset_path: Path | None, cache_root: Path) -> list[dict]:
    data = json.loads((dataset_path or LONGMEMEVAL_500).read_text())
    return [q for q in data if (cache_root / q["question_id"] / "store.db").exists()]


def _gold_sessions(q: dict) -> list[list[dict]]:
    wanted = set(q.get("answer_session_ids") or [])
    return [s for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]) if sid in wanted]


def _legacy_runtime() -> JudgmentRuntime:
    return JudgmentRuntime(JudgmentMode.LEGACY, None, JudgmentConfig())


def _build_engine(tmpdir: Path, qid: str, embedder) -> VirtualContextEngine:
    cfg_dict = _build_vc_config(storage_dir=str(tmpdir), session_id=f"bench-{qid}")
    cfg_dict["storage_root"] = str(tmpdir)
    cfg_dict["storage"]["sqlite"]["path"] = str(tmpdir / "store.db")
    cfg_dict["storage"]["filesystem"]["root"] = str(tmpdir / "store")
    config = load_config(config_dict=cfg_dict)
    return VirtualContextEngine(config=config, embedding_provider=embedder)


def _pin_temporal_to_legacy(engine: VirtualContextEngine) -> None:
    """Isolate S1: the inbound temporal flag (S3) stays on the pattern list in both modes."""
    from virtual_context.core.tag_generator import detect_temporal_heuristic
    retriever = engine._retriever
    retriever._detect_temporal = lambda message: detect_temporal_heuristic(message, retriever._temporal_patterns)


def _retrieve(engine: VirtualContextEngine, question: str, runtime: JudgmentRuntime) -> tuple[list[str], int, list[str]]:
    _pin_temporal_to_legacy(engine)
    with judgment.override(runtime):
        result = engine._retriever.retrieve(question, current_utilization=0.0)
    selected = [s.ref for s in result.summaries]
    overflow = [s.ref for s in result.overflow_summaries]
    tokens = sum(s.summary_tokens for s in result.summaries)
    return selected + overflow, tokens, selected


def run_rerank(runtime: JudgmentRuntime, *, limit: int | None = None, dataset_path: Path | None = None,
               cache_root: Path | None = None) -> dict:
    cache_root = cache_root or DEFAULT_CACHE_DIR
    questions = _questions(dataset_path, cache_root)
    if limit is not None:
        questions = questions[:limit]
    from virtual_context.core.embedding_provider import EmbeddingProvider
    embedder = EmbeddingProvider(model_name="all-MiniLM-L6-v2")
    client = runtime.client
    rows = []
    jev_calls: list[tuple[float, int]] = []
    for q in questions:
        qid = q["question_id"]
        src = cache_root / qid
        with tempfile.TemporaryDirectory(prefix=f"jev-rerank-{qid}-") as tmp:
            tmpdir = Path(tmp)
            for name in ("store.db", "store.db-wal", "store.db-shm"):
                if (src / name).exists():
                    shutil.copy2(src / name, tmpdir / name)
            gold = gold_segment_refs(tmpdir / "store.db", _gold_sessions(q))
            engine = _build_engine(tmpdir, qid, embedder)
            row = {"id": qid, "type": q["question_type"], "question": q["question"], "n_gold_segments": len(gold)}
            for mode, rt in (("legacy", _legacy_runtime()), ("jev", runtime)):
                n_before = len(getattr(client, "calls", []))
                order, tokens, selected = _retrieve(engine, q["question"], rt)
                if mode == "jev" and hasattr(client, "calls"):
                    for elapsed, in_tokens, cached in client.calls[n_before:]:
                        if not cached:
                            jev_calls.append((elapsed, in_tokens))
                row[mode] = {
                    "order": order[:30], "selected": selected, "selected_tokens": tokens,
                    "first_gold_rank": first_gold_rank(order, gold),
                    "gold_in_selected": any(r in gold for r in selected),
                }
            rows.append(row)

    def _agg(mode: str) -> dict:
        scored = [r for r in rows if r["n_gold_segments"] > 0]
        ranks = [r[mode]["first_gold_rank"] for r in scored if r[mode]["first_gold_rank"] is not None]
        out = {
            "n_scored": len(scored),
            "gold_in_selected_rate": (sum(1 for r in scored if r[mode]["gold_in_selected"]) / len(scored)) if scored else None,
            "gold_found_anywhere_rate": (len(ranks) / len(scored)) if scored else None,
            "mean_first_gold_rank": mean(ranks) if ranks else 0.0,
            "median_first_gold_rank": median(ranks) if ranks else 0.0,
            "mean_selected_tokens": mean(r[mode]["selected_tokens"] for r in rows) if rows else 0.0,
        }
        if mode == "jev":
            out["mean_jev_ms"] = mean(c[0] for c in jev_calls) if jev_calls else 0.0
            out["mean_jev_tokens"] = mean(c[1] for c in jev_calls) if jev_calls else 0.0
            out["jev_live_calls"] = len(jev_calls)
        return out

    by_type: dict[str, dict] = {}
    for r in rows:
        if r["n_gold_segments"] == 0:
            continue
        b = by_type.setdefault(r["type"], {"n": 0, "legacy_hits": 0, "jev_hits": 0})
        b["n"] += 1
        b["legacy_hits"] += int(r["legacy"]["gold_in_selected"])
        b["jev_hits"] += int(r["jev"]["gold_in_selected"])
    for b in by_type.values():
        b["legacy"] = b["legacy_hits"] / b["n"]
        b["jev"] = b["jev_hits"] / b["n"]
    return {"area": "rerank", "n": len(rows), "n_unscored_no_gold_match": sum(1 for r in rows if r["n_gold_segments"] == 0),
            "legacy": _agg("legacy"), "jev": _agg("jev"), "by_type": by_type, "rows": rows}
