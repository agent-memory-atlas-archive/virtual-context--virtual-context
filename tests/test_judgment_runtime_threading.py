"""Sentinel: every call to a seam-bearing function passes the engine runtime.

A call that omits ``runtime=`` / ``judgment_runtime=`` silently falls back to
the module registry (legacy unless a test overrides it), which is exactly the
process-wide behavior this layer no longer relies on.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "virtual_context"

SEAM_CALLS = (
    "render_summaries_for_model",
    "render_summary_items_for_model",
    "format_tag_section",
    "_detect_query_intent",
    "is_safety_critical_personal_evidence",
    "_build_structured_summary",
    "build_deterministic_structured_summary",
    "_contain_summary_results_for_speaker_context",
    "_find_quote",
    "_search_summaries",
    "_conditioned_find_quote",
    "fill_pass",
    "judge_admission",
    "judge_query_intent",
    "judge_temporal_intent",
    "judge_safety_critical",
    "rerank_summaries",
    "validate_tag_rollup_inputs",
    "_validated_tag_rollup_segment",
    "apply_tag_claim_safety_floor",
    "_select_tag_claims",
    "_rollup_structured_summary",
)

# Offline historical-repair CLI: no engine exists there, and it deliberately uses
# the legacy predicate for evidence migration (module default, never Jev).
ALLOWLIST = ("virtual_context/cli/structured_summary_migration_cmd.py",)


def _call_span(src: str, open_idx: int) -> str:
    depth = 0
    in_str = None
    i = open_idx
    while i < len(src):
        ch = src[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return src[open_idx:i + 1]
        i += 1
    return src[open_idx:]


def test_every_seam_call_passes_the_engine_runtime():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if str(path.relative_to(ROOT.parent)) in ALLOWLIST:
            continue
        src = path.read_text()
        for name in SEAM_CALLS:
            for m in re.finditer(r"(?<![\w.])" + re.escape(name) + r"\(", src):
                line_start = src.rfind("\n", 0, m.start()) + 1
                line = src[line_start:src.find("\n", m.start())]
                if line.lstrip().startswith(("def ", "async def ")):
                    continue
                span = _call_span(src, m.end() - 1)
                if span == "()":
                    continue  # a mention inside a string, not a call
                if "runtime=" not in span:
                    lineno = src.count("\n", 0, m.start()) + 1
                    offenders.append(f"{path.relative_to(ROOT.parent)}:{lineno} {name}(...)")
    assert not offenders, "seam calls without an engine runtime:\n" + "\n".join(offenders)
