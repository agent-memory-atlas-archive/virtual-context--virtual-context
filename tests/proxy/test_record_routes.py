"""The dashboard's record routes review and edit a loaded conversation's memory."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from virtual_context.proxy.dashboard import register_dashboard_routes
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.types import Fact, SegmentMetadata, StoredSegment

CONV = "conv-dashboard-review"
TURN = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def setup(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": CONV,
    }))
    engine._store.save_canonical_turn(
        CONV, 0, "I keep my keys under the mat", "noted", canonical_turn_id=TURN,
        turn_group_number=0, sort_key=1.0, primary_tag="home", tags=["home"],
    )
    engine._store.store_segment(StoredSegment(
        ref="seg-home", conversation_id=CONV, primary_tag="home", tags=["home"], summary="keys",
        metadata=SegmentMetadata(canonical_turn_ids=[TURN], source_mapping_complete=True),
    ))
    engine._store.store_facts([Fact(id="keys", subject="user", verb="keeps", object="keys under the mat",
                                    what="keeps keys under the mat", segment_ref="seg-home",
                                    conversation_id=CONV, tags=["home"])])
    queued = []
    app = FastAPI()
    register_dashboard_routes(app, ProxyMetrics(), state=SimpleNamespace(engine=engine, submit_record_edit=queued.append))
    return TestClient(app), engine, queued


def test_review_lists_rejects_and_restores_a_fact(setup):
    client, _, _ = setup
    base = f"/dashboard/conversations/{CONV}/record"
    facts = client.get(base + "/facts").json()["facts"]
    assert facts[0]["fact"] == "keeps keys under the mat"
    assert facts[0]["source_turns"][0]["user"] == "I keep my keys under the mat"
    assert client.post(base + "/facts/keys/reject", json={"reason": "private"}).json()["verdict"] == "rejected"
    assert client.get(base + "/facts").json()["facts"][0]["trust_state"] == "rejected"
    verdicts = client.get(base + "/verdicts").json()["verdicts"]
    assert (verdicts[0]["actor"], verdicts[0]["reason"]) == ("dashboard", "private")
    assert client.post(base + "/facts/keys/restore").status_code == 200


def test_editing_a_turn_queues_its_rebuild_and_is_audited(setup):
    client, _, queued = setup
    base = f"/dashboard/conversations/{CONV}/record"
    resp = client.post(base + f"/turns/{TURN}/edit", json={"user_content": "I keep my keys in a lockbox"})
    assert resp.status_code == 202 and queued == [resp.json()["operation_id"]]
    edit = client.get(base + "/edits").json()["edits"][0]
    assert edit["before_user_content"] == "I keep my keys under the mat"
    assert client.get(base + "/operations").json()["operations"][0]["action"] == "edit_turn"


def test_an_unloaded_conversation_is_not_found(setup):
    client, _, _ = setup
    assert client.get("/dashboard/conversations/other/record/facts").status_code == 404
