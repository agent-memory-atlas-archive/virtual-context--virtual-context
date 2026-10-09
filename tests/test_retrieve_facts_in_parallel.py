"""A request's facts are gathered while its summaries are being chosen."""
from __future__ import annotations

import threading
import time

import pytest

from tests.test_retriever import _make_retriever


@pytest.mark.regression("BUG-125")
def test_facts_are_prepared_while_topics_are_selected(tmp_sqlite_db, monkeypatch):
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        spans: dict[str, tuple[float, float, int]] = {}
        real_select = retriever._select_topics

        def slow_select(*args, **kwargs):
            start = time.monotonic()
            time.sleep(0.3)
            out = real_select(*args, **kwargs)
            spans["topics"] = (start, time.monotonic(), threading.get_ident())
            return out

        def slow_prepare(facts):
            start = time.monotonic()
            time.sleep(0.3)
            spans["facts"] = (start, time.monotonic(), threading.get_ident())
            return facts

        monkeypatch.setattr(retriever, "_select_topics", slow_select)
        result = retriever.retrieve("What about the court filing?", facts_transform=slow_prepare)

        assert result.summaries
        assert result.retrieval_metadata.get("facts_prepared") is True
        topics, facts = spans["topics"], spans["facts"]
        assert facts[2] != topics[2]
        assert facts[0] < topics[1] and topics[0] < facts[1], "the two must overlap in time"
    finally:
        store.close()


def test_without_a_transform_facts_are_returned_unprepared(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        result = retriever.retrieve("What about the court filing?")
        assert "facts_prepared" not in result.retrieval_metadata
    finally:
        store.close()
