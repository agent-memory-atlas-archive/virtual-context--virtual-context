"""The MCP topic tool and resources show only the bound conversation's topics.

``domain_status`` and the ``virtualcontext://domains`` resources read tags and
summaries with no conversation filter, so a store shared by several
conversations served every conversation's topics and summaries.
"""

from __future__ import annotations

import json

import pytest

from virtual_context.mcp import server
from virtual_context.types import SegmentMetadata, StoredSegment


@pytest.fixture
def bound_engine(tmp_path, monkeypatch):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": "conv-mine",
    }))
    raw = getattr(engine._store, "_store", engine._store)
    for conv, tag in (("conv-mine", "gardening"), ("conv-other", "salary")):
        raw.store_segment(StoredSegment(
            ref=f"{conv}-seg", conversation_id=conv, primary_tag=tag, tags=[tag],
            summary=f"{tag} summary", full_text=tag, metadata=SegmentMetadata(),
        ))
    monkeypatch.setattr(server, "_engine", engine)
    return engine


def _call(fn, *args):
    return getattr(fn, "fn", fn)(*args)


@pytest.mark.regression("BUG-110")
def test_domain_status_lists_only_this_conversations_topics(bound_engine):
    tags = {entry["tag"] for entry in json.loads(_call(server.domain_status))}
    assert tags == {"gardening"}


@pytest.mark.regression("BUG-110")
def test_domain_resources_do_not_serve_another_conversation(bound_engine):
    assert "salary" not in _call(server.list_domains)
    assert "salary summary" not in _call(server.get_domain_summaries, "salary")
    assert "gardening summary" in _call(server.get_domain_summaries, "gardening")
