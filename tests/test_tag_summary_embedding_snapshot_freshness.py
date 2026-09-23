"""Every worker sees a newly saved tag-summary embedding snapshot.

Each worker process keeps the snapshot in memory. Compaction saves a new one
from whichever worker ran it; the others must pick it up on their next read
instead of serving the copy they loaded first.
"""

from __future__ import annotations

import fakeredis
import numpy as np
import pytest

from virtual_context.proxy.session_state import SessionStateProvider

CONV = "conv-1"


def _workers(n: int = 2):
    shared = fakeredis.FakeRedis(decode_responses=False)
    return [SessionStateProvider(redis_client=shared, store=None) for _ in range(n)]


@pytest.mark.regression("PROXY-037")
def test_a_snapshot_saved_by_one_worker_reaches_another_that_already_loaded():
    a, b = _workers()
    a.save_tag_summary_embedding_snapshot(CONV, {"old": [1.0, 0.0]})
    assert set(b.load_tag_summary_embedding_snapshot(CONV)) == {"old"}

    a.save_tag_summary_embedding_snapshot(CONV, {"old": [1.0, 0.0], "new": [0.0, 1.0]})

    assert set(b.load_tag_summary_embedding_snapshot(CONV)) == {"old", "new"}


@pytest.mark.regression("PROXY-037")
def test_the_embedding_matrix_follows_the_newest_snapshot():
    a, b = _workers()
    a.save_tag_summary_embedding_snapshot(CONV, {"old": [3.0, 4.0]})
    tags, matrix = b.load_tag_summary_embedding_matrix(CONV)
    assert tags == ["old"]
    assert np.allclose(matrix, [[0.6, 0.8]])
    assert b.load_tag_summary_embedding_matrix(CONV)[1] is matrix

    a.save_tag_summary_embedding_snapshot(CONV, {"old": [3.0, 4.0], "new": [0.0, 2.0]})

    tags, matrix = b.load_tag_summary_embedding_matrix(CONV)
    assert sorted(tags) == ["new", "old"]
    assert matrix.shape == (2, 2)


def test_no_snapshot_means_no_matrix():
    (a,) = _workers(1)
    assert a.load_tag_summary_embedding_matrix(CONV) is None


@pytest.mark.regression("PROXY-037")
def test_a_deleted_snapshot_is_dropped_by_other_workers():
    a, b = _workers()
    a.save_tag_summary_embedding_snapshot(CONV, {"old": [1.0, 0.0]})
    assert b.load_tag_summary_embedding_snapshot(CONV)
    a.delete_tag_summary_embedding_snapshot(CONV)
    assert b.load_tag_summary_embedding_snapshot(CONV) is None


class _ChunkStore:
    def __init__(self):
        self.rows = [{"segment_ref": "s1", "chunk_index": 0, "embedding": [1.0, 0.0]}]

    def get_segment_chunk_embedding_page(self, *, conversation_id=None, limit=200, after=None):
        rows = [dict(r, cursor=(r["segment_ref"], r["chunk_index"])) for r in self.rows]
        return [r for r in rows if after is None or r["cursor"] > after][:limit]


def test_segment_chunk_matrix_is_rebuilt_when_the_snapshot_version_moves():
    shared = fakeredis.FakeRedis(decode_responses=False)
    store = _ChunkStore()
    writer = SessionStateProvider(redis_client=shared, store=store)
    reader = SessionStateProvider(redis_client=shared, store=store)
    writer.save_tag_summary_embedding_snapshot(CONV, {"t": [1.0, 0.0]})
    refs, matrix = reader.load_segment_chunk_matrix(CONV)
    assert refs == ["s1"] and matrix.shape == (1, 2)

    store.rows.append({"segment_ref": "s2", "chunk_index": 0, "embedding": [0.0, 3.0]})
    assert reader.load_segment_chunk_matrix(CONV)[0] == ["s1"]
    writer.save_tag_summary_embedding_snapshot(CONV, {"t": [1.0, 0.0]})

    refs, matrix = reader.load_segment_chunk_matrix(CONV)
    assert refs == ["s1", "s2"]
    assert np.allclose(matrix[1], [0.0, 1.0])
