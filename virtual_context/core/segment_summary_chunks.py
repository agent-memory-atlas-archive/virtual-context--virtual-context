"""Backfill the summary chunk for segments embedded before summaries were.

A segment's summary is embedded as its last chunk when its chunks are
written. Segments whose chunks predate that have raw-text chunks only; this
rewrites their chunks so the summary is among them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _chunk_texts(store, conversation_id: str) -> set[tuple[str, str]]:
    """Every (segment_ref, chunk text) stored for the conversation."""
    seen: set[tuple[str, str]] = set()
    after = None
    while True:
        page = store.get_segment_chunk_embedding_page(conversation_id=conversation_id, limit=200, after=after)
        if not page:
            return seen
        seen.update((str(row["segment_ref"]), str(row.get("text") or "")) for row in page)
        cursor = page[-1].get("cursor")
        if cursor is None or (after is not None and tuple(cursor) <= after):
            return seen
        after = tuple(cursor)


def backfill_segment_summary_chunks(store, semantic, conversation_id: str, *, dry_run: bool = False) -> dict:
    """Rewrite the chunks of every segment whose summary is not yet one of them.

    Returns counts of segments examined, segments missing the summary chunk,
    and segments rewritten. Rerunning after a full pass writes nothing.
    """
    existing = _chunk_texts(store, conversation_id)
    segments = store.get_all_segments(conversation_id=conversation_id)
    missing = [
        segment for segment in segments
        if (segment.summary or "").strip()
        and (segment.ref, segment.summary.strip()) not in existing
    ]
    written = 0
    if not dry_run:
        for segment in missing:
            try:
                semantic.embed_and_store_chunks(segment)
                written += 1
            except Exception:
                logger.warning("Summary chunk backfill failed for segment %s", segment.ref, exc_info=True)
    return {"segments": len(segments), "missing": len(missing), "written": written}
