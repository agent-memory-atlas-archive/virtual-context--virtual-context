"""Summary of a reader's VC tool chain for payload logs and autopsy reports."""

from __future__ import annotations

import json
from typing import Any


def _result(call: dict) -> dict:
    raw = call.get("result")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _useful(result: dict) -> bool:
    if result.get("error"):
        return False
    if result.get("found") is not None:
        return bool(result.get("found"))
    if int(result.get("tokens_added") or 0) > 0:
        return True
    return bool(result.get("results") or result.get("facts") or result.get("summaries"))


def analyze_tool_chain(tool_calls: list[dict]) -> dict[str, Any]:
    """Counts, pattern and paging effects of one reader's tool calls."""
    calls = [c for c in tool_calls or [] if isinstance(c, dict)]
    names = [str(c.get("tool", "")) for c in calls]
    results = [_result(c) for c in calls]
    useful = [_useful(r) for r in results]

    pattern: list[str] = []
    for name in names:
        if pattern and pattern[-1].split(" x")[0] == name:
            base, _, count = pattern[-1].partition(" x")
            pattern[-1] = f"{base} x{int(count or 1) + 1}"
        else:
            pattern.append(name)

    pivot = any(
        not useful[i] and names[i + 1] != names[i]
        for i in range(len(calls) - 1)
    )
    collapse_then_expand = any(
        name == "vc_expand_topic" and (calls[i].get("input") or {}).get("collapse_tags")
        for i, name in enumerate(names)
    ) or any(
        names[i] == "vc_collapse_topic" and "vc_expand_topic" in names[i + 1:]
        for i in range(len(names))
    )
    unique = sorted(set(names))
    return {
        "total_calls": len(calls),
        "chain_pattern": " -> ".join(pattern) if pattern else "none",
        "unique_tool_types": unique,
        "unique_tool_type_count": len(unique),
        "useful_calls": sum(useful),
        "wasted_calls": len(calls) - sum(useful),
        "tokens_added_total": sum(int(r.get("tokens_added") or 0) for r in results),
        "tokens_freed_total": sum(
            int(r.get("total_tokens_freed") or r.get("tokens_freed") or 0) for r in results
        ),
        "has_strategy_pivot": pivot,
        "has_collapse_then_expand": collapse_then_expand,
    }
