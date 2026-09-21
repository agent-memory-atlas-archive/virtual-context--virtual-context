from __future__ import annotations

import argparse
from pathlib import Path

from . import report
from .runtime import build_fake_runtime, build_live_runtime

AREAS = ("rerank", "intent", "temporal", "safety", "admission",
         "tag_reuse", "supersession", "consolidation", "curation", "tag_split", "grounding")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="benchmarks.jev", description="Legacy vs Jev judgment harness")
    p.add_argument("area", choices=AREAS + ("all",))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=report.RESULTS_DIR)
    p.add_argument("--fake", action="store_true", help="offline fake Jev client (smoke)")
    p.add_argument("--no-legacy", action="store_true", help="skip the legacy model arm where one exists")
    p.add_argument("--dataset", type=Path, default=None, help="LongMemEval 500q file for weak labels / rerank")
    p.add_argument("--budgets", default=None, help="rerank grid: comma-separated absolute summary budgets in tokens")
    p.add_argument("--min-probs", default=None, help="rerank grid: comma-separated rerank_min_probability values")
    args = p.parse_args(argv)
    runtime = build_fake_runtime() if args.fake else build_live_runtime()
    areas = AREAS if args.area == "all" else (args.area,)
    for area in areas:
        if area in ("intent", "temporal", "safety"):
            from .labeled import run_area
            result = run_area(area, runtime, limit=args.limit, dataset_path=args.dataset)
        elif area == "rerank":
            if args.budgets or args.min_probs:
                from .rerank import run_rerank_grid
                budgets = tuple(int(x) for x in (args.budgets or "7500").split(","))
                min_probs = tuple(float(x) for x in (args.min_probs or "0").split(","))
                result = run_rerank_grid(runtime, budgets=budgets, min_probs=min_probs, limit=args.limit, dataset_path=args.dataset)
            else:
                from .rerank import run_rerank
                result = run_rerank(runtime, limit=args.limit, dataset_path=args.dataset)
        elif area == "admission":
            from .admission import run_admission
            result = run_admission(runtime, limit=args.limit, offline=args.fake or args.no_legacy)
        else:
            from .seams_v2 import run_area as run_seam_area
            result = run_seam_area(area, runtime, limit=args.limit, offline=args.fake or args.no_legacy)
        path = report.write(result, args.out)
        print(report.table(result))
        print(f"\nwrote {path}")
        client = runtime.client
        if hasattr(client, "hits"):
            print(f"jev cache: hits={client.hits} misses={client.misses}")
    return 0
