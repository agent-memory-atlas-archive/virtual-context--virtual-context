"""Retrieval's recent context comes from the stored conversation, not a worker's memory.

A deployment runs several workers, each with its own in-memory history for a
conversation. Calls of one message can land on different workers, so recent
context taken from that memory made tagging, and the retrieval memo keyed on
it, depend on which worker answered.
"""
from __future__ import annotations

import pytest

from tests.test_context_hint_prewarm import _ingest_pairs, _make_engine
from virtual_context.types import Message


@pytest.mark.regression("PROXY-031")
def test_two_workers_with_different_histories_get_the_same_context(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _ingest_pairs(engine, 6)
        n = engine.config.tag_generator.context_lookback_pairs
        stored = engine._retrieval._stored_recent_history(n)
        assert stored and stored[-1].content == "here is a long reply about topic 5"
        worker_a = [Message(role="user", content="q a"), Message(role="assistant", content="a a")]
        worker_b = [Message(role="user", content="q b"), Message(role="assistant", content="a b"),
                    Message(role="user", content="q c"), Message(role="assistant", content="a c")]
        context_a = engine._retrieval._get_recent_context(
            engine._retrieval._stored_recent_history(n) or worker_a, n, exclude_last=0)
        context_b = engine._retrieval._get_recent_context(
            engine._retrieval._stored_recent_history(n) or worker_b, n, exclude_last=0)
        assert context_a == context_b
        assert context_a[-1] == "here is a long reply about topic 5"
    finally:
        engine.close()


@pytest.mark.regression("PROXY-031")
def test_the_in_flight_message_is_not_treated_as_context(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _ingest_pairs(engine, 3)
        # the current request's user message is stored before retrieval runs
        from virtual_context.proxy.formats import detect_format
        body = {"messages": [{"role": "user", "content": "the question being asked now"}]}
        engine._ingest_reconciler.ingest_batch(
            engine.config.conversation_id, body=body, fmt=detect_format(body),
            expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
        )
        n = engine.config.tag_generator.context_lookback_pairs
        context = engine._retrieval._get_recent_context(
            engine._retrieval._stored_recent_history(n), n, exclude_last=0)
        assert "the question being asked now" not in (context or [])
    finally:
        engine.close()


@pytest.mark.regression("PROXY-031")
def test_no_stored_turns_falls_back_to_the_given_history(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        assert engine._retrieval._stored_recent_history(3) is None
    finally:
        engine.close()
