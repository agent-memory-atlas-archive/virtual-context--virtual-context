"""Tag Consolidation Engine — post-compaction semantic clustering of tags.

Scans the full tag universe in a store, identifies semantically related tags
that should be treated as equivalent during retrieval, and writes alias
mappings + backfills segment_tags so retrieval covers both canonical and
variant tags.

Usage:
    from virtual_context.core.tag_consolidator import consolidate_tags
    result = consolidate_tags(store, llm_provider)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .judgment import judge_tag_consolidation
from .llm_utils import parse_llm_json
from .store import ContextStore
from ..types import LLMProvider

logger = logging.getLogger(__name__)


# ── prompt ──────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a tag taxonomy expert. Output valid JSON only. "
    "No markdown fences, no extra text."
)

_CONSOLIDATION_PROMPT = """\
Below is a list of tag names from a conversation memory store.

Your job: identify groups of tags that refer to THE SAME broad topic and
should be unified so that a search for any member of the group also finds
content stored under the other members.

Rules:
1. Only group tags that are truly about the same domain.  "model-kit" and
   "model-tanks" both refer to scale-model hobby projects — group them.
   But "model-kit" and "data-model" are unrelated — do NOT group them.
2. Each group must have exactly one "canonical" tag (the most general one)
   and one or more "aliases" (the more specific or variant tags).
3. A tag may appear in at most one group. If it doesn't belong to any group
   leave it out.
4. Only create groups where cross-referencing genuinely helps retrieval.
   Trivial morphological variants (plurals, hyphenation) are already handled
   elsewhere — focus on SEMANTIC relationships.
5. Keep the number of groups reasonable.  Quality over quantity.
6. Use only each tag NAME as a semantic signal: "model-tanks" clearly relates
   to scale modeling from its name alone.

Tag names:
{tag_list}

Respond with JSON:
{{
  "groups": [
    {{
      "canonical": "the-main-tag",
      "aliases": ["variant-1", "variant-2"],
      "reason": "one sentence explaining the grouping"
    }}
  ]
}}"""


# ── types ───────────────────────────────────────────────────────────────

@dataclass
class ConsolidationGroup:
    """A cluster of semantically equivalent tags."""
    canonical: str
    aliases: list[str]
    reason: str = ""


@dataclass
class ConsolidationResult:
    """Result of running tag consolidation on a store."""
    groups: list[ConsolidationGroup] = field(default_factory=list)
    aliases_written: int = 0
    segment_tags_added: int = 0
    # Exact writes made by an apply run, one entry per group, so a run can be
    # reverted: {"canonical", "aliases_written": [...], "segment_refs": [...]}.
    applied: list[dict] = field(default_factory=list)
    # Aliases left alone because they already map to a different canonical.
    skipped: list[dict] = field(default_factory=list)


# ── core logic ──────────────────────────────────────────────────────────

def _store_conversation_id(store: ContextStore) -> str:
    conversation_id = getattr(store, "conversation_id", "")
    return conversation_id if isinstance(conversation_id, str) else ""


def _get_store_aliases(store: ContextStore) -> dict[str, str]:
    getter = getattr(store, "get_tag_aliases", None)
    if not callable(getter):
        return {}
    conversation_id = _store_conversation_id(store)
    try:
        return getter(conversation_id=conversation_id or None)
    except TypeError:
        return getter()


def _set_store_alias(store: ContextStore, alias: str, canonical: str) -> None:
    setter = getattr(store, "set_tag_alias", None)
    if not callable(setter):
        return
    conversation_id = _store_conversation_id(store)
    try:
        setter(alias, canonical, conversation_id=conversation_id)
    except TypeError:
        setter(alias, canonical)

def consolidate_tags(
    store: ContextStore,
    llm: LLMProvider,
    *,
    dry_run: bool = False,
    batch_size: int = 500,
    judgment_runtime=None,
    embed_fn=None,
    max_pairs: int = 200,
    strict: bool = False,
    groups: list[ConsolidationGroup] | None = None,
    on_group_applied=None,
) -> ConsolidationResult:
    """Run tag consolidation on *store*.

    1. Load all tag names. Derived tag/segment prose is intentionally withheld
       because this stateless job has no audience-bound speaker proof.
    2. Send to LLM to identify semantic clusters.
    3. Write alias mappings to tag_aliases table.
    4. Backfill segment_tags: for every segment that has an alias tag,
       also add the canonical tag so retrieval covers both.

    Args:
        store: The context store to consolidate.
        llm: LLM provider for semantic clustering.
        dry_run: If True, compute groups but don't write to store.
        batch_size: Max tags per LLM call (for very large stores).
        judgment_runtime: Engine judgment runtime for the tag_consolidation seam.
        embed_fn: Optional tag embedding function used to propose candidate pairs.
        max_pairs: Cap on candidate pairs put to the judgment seam.
        strict: In jev mode, a missing model answer raises instead of
            applying legacy groups.
        groups: Pre-computed groups (a reviewed dry-run plan) to apply
            verbatim; no judgment runs and no regrouping happens when given.
        on_group_applied: Called with each group's provenance entry as soon as
            its writes are committed, so a caller can persist progress before
            a later group fails.

    Returns:
        ConsolidationResult with groups found and counts of writes.
    """
    # Layer-2 summaries and orphan segment snippets have no audience-bound
    # speaker proof at this stateless model boundary. Use names only.
    # Reads are scoped to the store's conversation: on a shared backend an
    # unscoped read would mix every conversation's vocabulary into one job.
    conversation_id = _store_conversation_id(store)
    if groups is not None:
        plan_groups = list(groups)
        _require_disjoint_groups(plan_groups)
        logger.info("Applying %d pre-computed consolidation groups verbatim.", len(plan_groups))
        return _apply_groups(
            store, plan_groups, conversation_id, dry_run=dry_run, on_group_applied=on_group_applied,
        )

    all_tags = store.get_all_tags(conversation_id=conversation_id or None)
    tag_names = [ts.tag for ts in all_tags]

    if not tag_names:
        logger.info("No tags found — nothing to consolidate.")
        return ConsolidationResult()

    logger.info("Consolidating %d tags...", len(tag_names))

    def llm_groups() -> list[dict]:
        # Batch tag entries and call LLM for each batch
        found: list[ConsolidationGroup] = []
        for batch_start in range(0, len(tag_names), batch_size):
            batch = tag_names[batch_start:batch_start + batch_size]
            tag_list = "\n".join(f"- {tag}" for tag in batch)
            prompt = _CONSOLIDATION_PROMPT.format(tag_list=tag_list)

            try:
                response, _ = llm.complete(system=_SYSTEM, user=prompt, max_tokens=4096)
                found.extend(_parse_response(response))
            except Exception as e:
                logger.error("LLM consolidation call failed: %s", e)
        return [{"canonical": g.canonical, "aliases": list(g.aliases), "reason": g.reason} for g in found]

    judged = judge_tag_consolidation(
        tag_names,
        legacy=llm_groups,
        runtime=judgment_runtime,
        canonical_rank={ts.tag: int(getattr(ts, "usage_count", 0) or 0) for ts in all_tags},
        embed_fn=embed_fn,
        max_pairs=max_pairs,
        strict=strict,
    )
    all_groups = [
        ConsolidationGroup(canonical=g["canonical"], aliases=list(g["aliases"]), reason=g.get("reason", ""))
        for g in judged
    ]

    if not all_groups:
        logger.info("No consolidation groups identified.")
        return ConsolidationResult()

    # Cross-batch merge: if a tag is canonical in one group
    # and an alias in another, merge the groups transitively.
    all_groups = _merge_transitive_groups(all_groups)

    logger.info("Found %d consolidation groups (after merge).", len(all_groups))
    for g in all_groups:
        logger.info("  %s ← %s (%s)", g.canonical, g.aliases, g.reason)

    return _apply_groups(
        store, all_groups, conversation_id, dry_run=dry_run, on_group_applied=on_group_applied,
    )


def _require_disjoint_groups(groups: list[ConsolidationGroup]) -> None:
    seen: dict[str, str] = {}
    for g in groups:
        for tag in (g.canonical, *g.aliases):
            if tag in seen and seen[tag] != g.canonical:
                raise ValueError(f"plan groups overlap on tag {tag!r} ({seen[tag]!r} and {g.canonical!r})")
            seen[tag] = g.canonical


def _create_alias(store: ContextStore, alias: str, canonical: str, conversation_id: str) -> bool:
    creator = getattr(store, "create_tag_alias_if_absent", None)
    if callable(creator):
        return bool(creator(alias, canonical, conversation_id=conversation_id or ""))
    _set_store_alias(store, alias, canonical)
    return True


def _apply_groups(
    store: ContextStore,
    all_groups: list[ConsolidationGroup],
    conversation_id: str,
    *,
    dry_run: bool,
    on_group_applied=None,
) -> ConsolidationResult:
    result = ConsolidationResult(groups=all_groups)
    if dry_run or not all_groups:
        return result

    existing_aliases = _get_store_aliases(store)
    for group in all_groups:
        written: list[str] = []
        backfill: list[str] = []
        for alias in group.aliases:
            current = existing_aliases.get(alias)
            if current is not None and current != group.canonical:
                # Another mapping owns this alias; touching its segments would
                # contradict the mapping retrieval already follows.
                result.skipped.append({"alias": alias, "canonical": group.canonical, "existing": current})
                continue
            if current == group.canonical:
                backfill.append(alias)
                continue
            # Conditional insert: only the run that created the row records it,
            # so a concurrent apply cannot claim (and later revert) our mapping.
            if _create_alias(store, alias, group.canonical, conversation_id):
                written.append(alias)
                backfill.append(alias)
            else:
                result.skipped.append({"alias": alias, "canonical": group.canonical, "existing": "concurrent"})
        result.aliases_written += len(written)
        # Set-based on segment_tags: nothing is read back and rewritten, so a
        # compaction landing on the same segment cannot be overwritten, and
        # every alias-tagged segment is reached, not a bounded page of them.
        refs = list(store.add_tag_to_segments_with_tags(
            group.canonical, backfill, conversation_id=conversation_id or "",
        )) if backfill else []
        result.segment_tags_added += len(refs)
        entry = {"canonical": group.canonical, "aliases_written": written, "segment_refs": refs}
        result.applied.append(entry)
        if on_group_applied is not None:
            on_group_applied(entry)

    logger.info("Wrote %d new aliases; backfilled %d segment_tags entries; skipped %d aliases.",
                result.aliases_written, result.segment_tags_added, len(result.skipped))
    return result


def revert_consolidation(store: ContextStore, applied: list[dict]) -> dict:
    """Undo the exact writes recorded in ``ConsolidationResult.applied``."""
    conversation_id = _store_conversation_id(store)
    aliases_deleted = 0
    segment_tags_removed = 0
    for entry in applied:
        canonical = str(entry.get("canonical", ""))
        refs = [str(r) for r in entry.get("segment_refs", [])]
        if canonical and refs:
            segment_tags_removed += int(store.remove_tag_from_segments(
                canonical, refs, conversation_id=conversation_id or "",
            ) or 0)
        for alias in entry.get("aliases_written", []):
            aliases_deleted += int(store.delete_tag_alias(str(alias), conversation_id=conversation_id or "") or 0)
    return {"aliases_deleted": aliases_deleted, "segment_tags_removed": segment_tags_removed}

def _merge_transitive_groups(
    groups: list[ConsolidationGroup],
) -> list[ConsolidationGroup]:
    """Merge groups that share tags transitively.

    If group A has canonical="scale-model" aliases=["model-kit"] and group B
    has canonical="model-kit" aliases=["model-kit-assembly"], merge them into
    one group with canonical="scale-model" and all others as aliases.

    Also handles the case where a canonical from one group appears as an alias
    in another, or two groups share an alias.
    """
    # Build union-find over all tag names mentioned in groups
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Register all tags and union within each group
    for g in groups:
        parent.setdefault(g.canonical, g.canonical)
        for alias in g.aliases:
            parent.setdefault(alias, alias)
            union(alias, g.canonical)

    # Collect clusters
    clusters: dict[str, set[str]] = {}
    all_tags_in_groups = set()
    for g in groups:
        all_tags_in_groups.add(g.canonical)
        all_tags_in_groups.update(g.aliases)

    for tag in all_tags_in_groups:
        root = find(tag)
        clusters.setdefault(root, set()).add(tag)

    # Pick canonical for each cluster: prefer the tag that was canonical
    # in the most groups, breaking ties by shortest name
    canonical_counts: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    for g in groups:
        canonical_counts[g.canonical] = canonical_counts.get(g.canonical, 0) + 1
        if g.reason:
            reasons.setdefault(find(g.canonical), []).append(g.reason)

    merged: list[ConsolidationGroup] = []
    for root, members in clusters.items():
        if len(members) <= 1:
            continue
        # Pick canonical: highest canonical_count, then shortest
        best = max(
            members,
            key=lambda t: (canonical_counts.get(t, 0), -len(t)),
        )
        aliases = sorted(members - {best})
        reason = (reasons.get(root, [""]))[0]
        merged.append(ConsolidationGroup(
            canonical=best,
            aliases=aliases,
            reason=reason,
        ))

    return merged


def _parse_response(response: str) -> list[ConsolidationGroup]:
    data = parse_llm_json(response)
    if not data:
        logger.error("Failed to parse consolidation response.")
        return []

    groups = []
    for g in data.get("groups", []):
        canonical = g.get("canonical", "")
        aliases = g.get("aliases", [])
        reason = g.get("reason", "")
        if canonical and aliases:
            groups.append(ConsolidationGroup(
                canonical=canonical,
                aliases=aliases,
                reason=reason,
            ))
    return groups
