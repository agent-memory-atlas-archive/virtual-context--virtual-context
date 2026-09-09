"""BUG-073: ordinary channel backfill preserves immutable source receipts."""

import json

import pytest

from tests.test_audience_reassignment import (
    ACTOR,
    ANSWER,
    BODY,
    CHANNEL,
    GUILD,
    MESSAGE,
    ORIGINAL,
    OWNER,
    TARGET,
    TENANT,
)
from tests.test_channel_backfill import _channel_envelope, _run_cli
from virtual_context.config import VirtualContextConfig
from virtual_context.core.composite_store import CompositeStore
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.engine import VirtualContextEngine
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import StorageConfig, TagGeneratorConfig

pytestmark = pytest.mark.regression("BUG-073")


@pytest.fixture
def backfill_stack(tmp_path):
    import hashlib

    store = SQLiteStore(tmp_path / "backfill-receipts.db")
    for key in (OWNER, ORIGINAL, TARGET):
        store.activate_conversation(key)
        store.upsert_conversation(tenant_id=TENANT, conversation_id=key)
    for alias in (ORIGINAL, TARGET):
        store.save_conversation_alias(alias, OWNER)
    store._get_conn().execute(
        "UPDATE conversations SET phase='active' WHERE conversation_id=?", (OWNER,)
    )
    store._get_conn().commit()
    config = VirtualContextConfig(
        conversation_id=OWNER,
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    rec = IngestReconciler(store=store, semantic=semantic)
    digest = hashlib.sha256(BODY.encode()).hexdigest()
    rec.ingest_single(
        OWNER,
        user_content=BODY,
        assistant_content=ANSWER,
        user_origin_channel_id=CHANNEL,
        user_origin_channel_label="#synthetic",
        user_sender_actor_id=f"actor:discord:{ACTOR}",
        user_reply_edge={
            "source_message_id": MESSAGE,
            "audience_conversation_id": ORIGINAL,
            "audience_attribution_version": 1,
        },
        user_source_claim={
            "version": 1,
            "agent_scope_id": "synthetic-agent",
            "platform": "discord",
            "account_id": "synthetic-account",
            "message_id": MESSAGE,
            "channel_id": CHANNEL,
            "guild_id": GUILD,
            "author_id": ACTOR,
            "transport_body_sha256": digest,
            "canonical_body_sha256": digest,
            "projection_version": "openclaw-current-user-v1",
            "reply_target_message_id": "",
        },
        expected_conversation_generation=store.get_conversation_generation(OWNER),
        expected_lifecycle_epoch=1,
    )
    conn = store._get_conn()
    assistant_id = conn.execute(
        "SELECT canonical_turn_id FROM canonical_turns WHERE assistant_content<>''"
    ).fetchone()[0]
    conn.execute(
        "UPDATE canonical_turns SET origin_channel_id='',origin_channel_label='',origin_conversation_id=? WHERE canonical_turn_id=?",
        ("discord:channel:" + CHANNEL, assistant_id),
    )
    conn.commit()
    engine = VirtualContextEngine.__new__(VirtualContextEngine)
    engine._store = CompositeStore(
        segments=store, facts=store, fact_links=store, state=store, search=store
    )
    yield store, engine, assistant_id
    store.close()


def _protect(store, assistant_id, mode):
    if mode in {"audience", "both"}:
        plan = store.plan_audience_reassignment(
            OWNER,
            ORIGINAL,
            TARGET,
            tenant_id=TENANT,
            expected_lifecycle_epoch=1,
            operation_id="synthetic-backfill-audience",
        )
        store.reassign_audience(plan, dry_run=False)
    if mode in {"enrichment", "both"}:
        plan = store.plan_assistant_channel_enrichment(
            OWNER,
            tenant_id=TENANT,
            audience_conversation_id=TARGET if mode == "both" else ORIGINAL,
            expected_lifecycle_epoch=1,
            operation_id="synthetic-backfill-enrichment",
            assistant_canonical_turn_ids=[assistant_id],
        )
        store.enrich_assistant_channels(plan, dry_run=False)


def _later(store, suffix="later", sort_key=100000.0):
    body = "A synthetic later eligible message."
    row_id = "synthetic-backfill-" + suffix
    raw = json.dumps(
        [
            {
                "type": "text",
                "text": _channel_envelope(
                    body, chat_id="channel:7000000000000000001", group_channel="#other-synthetic"
                ),
            }
        ]
    )
    store.save_canonical_turn(
        OWNER, -1, body, "", canonical_turn_id=row_id, sort_key=sort_key, user_raw_content=raw
    )
    return row_id


@pytest.mark.parametrize("mode", ["audience", "enrichment", "both"])
def test_receipted_rows_skip_before_limit_and_later_fill_matches_dry_run(backfill_stack, mode):
    store, engine, assistant_id = backfill_stack
    _protect(store, assistant_id, mode)
    later_id = _later(store)
    conn = store._get_conn()
    before = dict(
        conn.execute(
            "SELECT * FROM canonical_turns WHERE canonical_turn_id=?", (assistant_id,)
        ).fetchone()
    )
    snapshot = tuple(conn.iterdump())
    preview = engine.backfill_channels(OWNER, dry_run=True, limit=1)
    assert tuple(conn.iterdump()) == snapshot
    result = engine.backfill_channels(OWNER, limit=1)
    assert preview["updated"] == 1
    assert preview["skipped_receipted"] == 1
    assert result["updated"] == preview["updated"] == 1
    assert result["skipped_receipted"] == 1
    assert (
        dict(
            conn.execute(
                "SELECT * FROM canonical_turns WHERE canonical_turn_id=?", (assistant_id,)
            ).fetchone()
        )
        == before
    )
    assert tuple(
        conn.execute(
            "SELECT origin_channel_id,origin_channel_label FROM canonical_turns WHERE canonical_turn_id=?",
            (later_id,),
        ).fetchone()
    ) == ("7000000000000000001", "#other-synthetic")


def test_protected_existing_label_remains_evidence_for_unprotected_row(backfill_stack):
    store, engine, assistant_id = backfill_stack
    _protect(store, assistant_id, "both")
    store.save_canonical_turn(
        OWNER,
        -1,
        "Another synthetic message.",
        "",
        canonical_turn_id="synthetic-id-only",
        sort_key=100000.0,
        origin_channel_id=CHANNEL,
    )
    preview = engine.backfill_channels(OWNER, dry_run=True)
    assert preview["updated"] == 1 and preview["skipped_receipted"] == 1
    assert engine.backfill_channels(OWNER)["updated"] == 1
    row = (
        store._get_conn()
        .execute(
            "SELECT origin_channel_label FROM canonical_turns WHERE canonical_turn_id='synthetic-id-only'"
        )
        .fetchone()
    )
    assert row[0] == "#synthetic"


def test_unrelated_write_error_is_not_silently_treated_as_a_receipt_skip(
    backfill_stack, monkeypatch
):
    store, engine, assistant_id = backfill_stack
    _protect(store, assistant_id, "both")
    _later(store)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic storage failure")

    monkeypatch.setattr(store, "update_canonical_turn_channels_if_empty", fail)
    with pytest.raises(RuntimeError, match="synthetic storage failure"):
        engine.backfill_channels(OWNER)


def test_cli_reports_receipt_skips_in_tenant_totals(backfill_stack, monkeypatch, capsys):
    store, _engine, assistant_id = backfill_stack
    _protect(store, assistant_id, "both")
    _later(store)
    payload = _run_cli(
        monkeypatch,
        capsys,
        [
            "admin",
            "backfill-channels",
            "--all-convs-for-tenant",
            "--dry-run",
            "--tenant-id",
            TENANT,
            "--storage-backend",
            "sqlite",
            "--sqlite-path",
            str(store.db_path),
        ],
    )
    assert payload["status"] == "ok"
    assert payload["skipped_receipted"] == 1
    assert payload["updated"] == 1
    assert payload["results"][0]["skipped_receipted"] == 1
