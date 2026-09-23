"""A segment's summary is embedded as one of its chunks.

Chunk embeddings come from the raw turn text, which buries a figure in
conversation; the summary states it compactly. Embedding the summary as the
segment's last chunk lets query similarity find the segment either way.
"""

from __future__ import annotations

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import StoredSegment


def _manager(tmp_path):
    store = SQLiteStore(tmp_path / "vc.db")
    manager = SemanticSearchManager(store=store, config=VirtualContextConfig())
    embedded: list[str] = []

    def embed(texts):
        embedded.extend(texts)
        return [[1.0, float(len(t))] for t in texts]

    manager._embed_fn = embed
    return manager, store, embedded


def _segment(ref="seg-1", summary="Average muscle 81.7% and fat 14.0% on Aug 12."):
    return StoredSegment(
        ref=ref, conversation_id="conv", primary_tag="recomp", tags=["recomp"],
        summary=summary, summary_tokens=12,
        full_text="user: here are my numbers for the week, muscle and fat by day.\n" * 40
        + "assistant: noted, the stall continues and we will hold the doses.\n" * 40,
        full_tokens=20,
    )


def _chunks(store, ref):
    return [c for c in store.get_all_chunk_embeddings() if c.segment_ref == ref]


@pytest.mark.regression("BUG-081")
def test_the_summary_is_stored_as_the_last_chunk(tmp_path):
    manager, store, embedded = _manager(tmp_path)
    segment = _segment()
    store.store_segment(segment)
    manager.embed_and_store_chunks(segment)
    chunks = sorted(_chunks(store, "seg-1"), key=lambda c: c.chunk_index)
    assert chunks[-1].text == segment.summary
    assert len(chunks) >= 2
    assert segment.summary in embedded


@pytest.mark.regression("BUG-081")
def test_an_empty_summary_adds_no_chunk(tmp_path):
    manager, store, _ = _manager(tmp_path)
    segment = _segment(summary="  ")
    store.store_segment(segment)
    manager.embed_and_store_chunks(segment)
    assert all(c.text.strip() for c in _chunks(store, "seg-1"))
    assert not any(c.text == "  " for c in _chunks(store, "seg-1"))


@pytest.mark.regression("BUG-081")
def test_backfill_adds_the_summary_chunk_once(tmp_path):
    from virtual_context.core.segment_summary_chunks import backfill_segment_summary_chunks

    manager, store, embedded = _manager(tmp_path)
    old = _segment("seg-old")
    store.store_segment(old)
    # Chunks written before summaries were embedded: raw text only.
    summary, old.summary = old.summary, ""
    manager.embed_and_store_chunks(old)
    old.summary = summary
    done = _segment("seg-done")
    store.store_segment(done)
    manager.embed_and_store_chunks(done)

    assert backfill_segment_summary_chunks(store, manager, "conv", dry_run=True) == {"segments": 2, "missing": 1, "written": 0}
    assert not any(c.text == summary for c in _chunks(store, "seg-old"))
    assert backfill_segment_summary_chunks(store, manager, "conv") == {"segments": 2, "missing": 1, "written": 1}
    assert any(c.text == summary for c in _chunks(store, "seg-old"))
    assert backfill_segment_summary_chunks(store, manager, "conv") == {"segments": 2, "missing": 0, "written": 0}
