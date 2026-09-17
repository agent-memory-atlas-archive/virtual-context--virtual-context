from __future__ import annotations

import argparse
from pathlib import Path

from . import report
from .runtime import build_fake_runtime, build_live_runtime

AREAS = ("rerank", "intent", "temporal", "safety", "admission")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="benchmarks.jev", description="Legacy vs Jev judgment harness")
    p.add_argument("area", choices=AREAS + ("all",))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=report.RESULTS_DIR)
    p.add_argument("--fake", action="store_true", help="offline fake Jev client (smoke)")
    p.add_argument("--dataset", type=Path, default=None, help="LongMemEval 500q file for weak labels / rerank")
    args = p.parse_args(argv)
    runtime = build_fake_runtime() if args.fake else build_live_runtime()
    areas = AREAS if args.area == "all" else (args.area,)
    for area in areas:
        if area in ("intent", "temporal", "safety"):
            from .labeled import run_area
            result = run_area(area, runtime, limit=args.limit, dataset_path=args.dataset)
        elif area == "rerank":
            from .rerank import run_rerank
            result = run_rerank(runtime, limit=args.limit, dataset_path=args.dataset)
        else:
            from .admission import run_admission
            result = run_admission(runtime, limit=args.limit, offline=args.fake)
        path = report.write(result, args.out)
        print(report.table(result))
        print(f"\nwrote {path}")
        client = runtime.client
        if hasattr(client, "hits"):
            print(f"jev cache: hits={client.hits} misses={client.misses}")
    return 0
