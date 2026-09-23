"""A completed request/response pair is stored with its proved audience.

The turn-complete path appends a pair the payload reconcile has not stored
yet. Without the proved audience both rows are audience-ineligible, and
without a channel the assistant half is rejected by conversation-scoped
reads, so every summary built from the turn is withheld.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.core.tagging_pipeline import TaggingPipeline
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    SOURCE_CONVERSATION_KEY,
    Message,
    StorageConfig,
    TagGeneratorConfig,
    TurnTagEntry,
)

CONV = "sk:agent:demo:discord:guild:1"


def _pipeline(tmp_path: Path) -> tuple[TaggingPipeline, SQLiteStore]:
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t1", conversation_id=CONV)
    config = VirtualContextConfig(
        conversation_id=CONV,
        tenant_id="t1",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    pipeline = TaggingPipeline.__new__(TaggingPipeline)
    pipeline._store = store
    pipeline._semantic = semantic
    pipeline.config = config
    return pipeline, store


def _persist(pipeline: TaggingPipeline, route: str | None) -> None:
    metadata = {"conversation info": {"chat_id": "channel:7", "group_channel": "#a"}}
    if route is not None:
        metadata[SOURCE_CONVERSATION_KEY] = route
    pipeline._persist_canonical_turn(
        TurnTagEntry(turn_number=0, primary_tag="topic", tags=["topic"], session_date=""),
        Message(role="user", content="what did we decide?", metadata=metadata),
        Message(role="assistant", content="we chose the smaller dose"),
    )


@pytest.mark.regression("PROXY-036")
def test_a_proved_route_stamps_both_halves_of_the_completed_pair(tmp_path):
    pipeline, store = _pipeline(tmp_path)
    _persist(pipeline, CONV)
    user, assistant = store.get_all_canonical_turns(CONV)
    for row in (user, assistant):
        assert row.audience_conversation_id == CONV
        assert int(row.audience_attribution_version) == 1
        assert row.origin_channel_id == "7"
    assert assistant.origin_channel_label == "#a"
    assert not (assistant.sender_actor_id or "")


@pytest.mark.regression("PROXY-036")
def test_an_unproved_route_stays_ineligible(tmp_path):
    pipeline, store = _pipeline(tmp_path)
    _persist(pipeline, "sk:agent:demo:discord:guild:unknown")
    user, assistant = store.get_all_canonical_turns(CONV)
    for row in (user, assistant):
        assert not (row.audience_conversation_id or "")
        assert int(row.audience_attribution_version or 0) == 0
    assert user.origin_channel_id == "7"
    assert not (assistant.origin_channel_id or "")
