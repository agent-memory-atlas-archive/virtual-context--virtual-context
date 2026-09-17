from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def write(result: dict, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{result['area']}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def table(result: dict) -> str:
    area = result["area"]
    if area == "rerank":
        lines = [f"## rerank (n={result['n']}, scored={result['legacy']['n_scored']}, unscored_no_gold_match={result['n_unscored_no_gold_match']})", "",
                 "| mode | gold in selected | gold found anywhere | mean rank of first gold | median rank | mean selected tokens | mean Jev ms | mean Jev input tokens |",
                 "|---|---|---|---|---|---|---|---|"]
        for mode in ("legacy", "jev"):
            m = result[mode]
            lines.append(f"| {mode} | {_pct(m['gold_in_selected_rate'])} | {_pct(m['gold_found_anywhere_rate'])} | {m['mean_first_gold_rank']:.2f} | {m['median_first_gold_rank']:.1f} | {m['mean_selected_tokens']:.0f} | {m.get('mean_jev_ms', 0.0):.0f} | {m.get('mean_jev_tokens', 0.0):.0f} |")
        lines += ["", "| question type | n | legacy gold-in-selected | jev gold-in-selected |", "|---|---|---|---|"]
        for qt, row in sorted(result["by_type"].items()):
            lines.append(f"| {qt} | {row['n']} | {_pct(row['legacy'])} | {_pct(row['jev'])} |")
        return "\n".join(lines)
    if area == "rerank_grid":
        arms = result["arms"]
        lines = [f"## rerank grid (n={result['n']}; cell = gold-in-selected % / mean selected tokens / mean summaries selected)", "",
                 "| budget tokens | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for b in result["budgets"]:
            row = result["grid"][str(b)]
            cells = [f"{_pct(row[a]['gold_in_selected_rate'])} / {row[a]['mean_selected_tokens']:.0f} / {row[a]['mean_n_selected']:.1f}" if a in row else "-" for a in arms]
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    if area == "admission":
        lines = [f"## admission (sets={result['n_sets']}, candidates={result['n_candidates']}, legacy arm={result.get('legacy_arm')})", "",
                 "| side | reason accuracy | admit/reject accuracy | coverage accuracy | mean ms | mean input tokens |", "|---|---|---|---|---|---|"]
        for side in ("legacy", "jev"):
            m = result[side]
            lines.append(f"| {side} | {_pct(m['reason_accuracy'])} | {_pct(m['admit_accuracy'])} | {_pct(m['coverage_accuracy'])} | {m['mean_ms']:.0f} | {m['mean_tokens']:.0f} |")
        lines.append(f"\nlegacy vs jev reason agreement: {_pct(result['agreement'])}")
        return "\n".join(lines)
    lines = [f"## {area} (n={result['n']}, strong={result['n_strong']}, weak={result['n_weak']})", "",
             "| side | accuracy all | accuracy strong | accuracy weak |", "|---|---|---|---|"]
    for side in ("legacy", "jev_raw", "jev_deployed"):
        m = result[side]
        lines.append(f"| {side} | {_pct(m['accuracy_all'])} | {_pct(m['accuracy_strong'])} | {_pct(m['accuracy_weak'])} |")
    lines.append(f"\nJev mean latency {result['jev_latency_ms']:.0f} ms, mean input tokens {result['jev_tokens']:.0f}")
    return "\n".join(lines)
