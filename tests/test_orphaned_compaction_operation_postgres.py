"""PostgreSQL twin of test_orphaned_compaction_operation.py."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tests.pg_helpers import pg_dsn

_pg_required = pytest.mark.skipif(not pg_dsn(), reason="DATABASE_URL not set; skipping PG orphaned compaction")


@pytest.fixture(scope="module")
def store():
    from virtual_context.storage.postgres import PostgresStore
    return PostgresStore(pg_dsn())


def _seed(store, *, phase: str, hb_age_s: float) -> tuple[str, str]:
    conv = f"pg-orphan-{uuid.uuid4().hex[:8]}"
    op_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    heartbeat = now - timedelta(seconds=hb_age_s)
    with store.pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                """INSERT INTO conversation_lifecycle (conversation_id, generation, deleted, updated_at)
                   VALUES (%s, 0, FALSE, %s) ON CONFLICT (conversation_id) DO NOTHING""",
                (conv, now),
            )
            conn.execute(
                """INSERT INTO conversations (conversation_id, tenant_id, phase, lifecycle_epoch, created_at, updated_at)
                   VALUES (%s, 't-orphan', %s, 1, %s, %s)""",
                (conv, phase, now, now),
            )
            conn.execute(
                """INSERT INTO compaction_operation
                   (operation_id, conversation_id, lifecycle_epoch, phase_index, phase_count, phase_name,
                    status, started_at, heartbeat_ts, owner_worker_id, created_at)
                   VALUES (%s, %s, 1, 5, 7, 'tag_summaries', 'running', %s, %s, 'dead-worker', %s)""",
                (op_id, conv, heartbeat, heartbeat, heartbeat),
            )
    return conv, op_id


def _status(store, op_id):
    with store.pool.connection() as conn:
        return conn.execute(
            "SELECT status FROM compaction_operation WHERE operation_id = %s", (op_id,),
        ).fetchone()["status"]


def _retire(store, conv, op_id, **overrides):
    kwargs = dict(operation_id=op_id, conversation_id=conv, lifecycle_epoch=1,
                  stale_after_s=60.0, error_message="orphaned: conversation not compacting")
    kwargs.update(overrides)
    return store.fail_orphaned_compaction_operation(**kwargs)


@_pg_required
@pytest.mark.regression("BUG-080")
def test_stale_operation_on_an_active_conversation_is_failed_pg(store):
    conv, op_id = _seed(store, phase="active", hb_age_s=3600.0)
    assert _retire(store, conv, op_id) is True
    assert _status(store, op_id) == "failed"


@_pg_required
@pytest.mark.regression("BUG-080")
def test_compacting_fresh_or_other_operations_are_left_alone_pg(store):
    conv, op_id = _seed(store, phase="compacting", hb_age_s=3600.0)
    assert _retire(store, conv, op_id) is False
    conv2, op2 = _seed(store, phase="active", hb_age_s=5.0)
    assert _retire(store, conv2, op2) is False
    conv3, op3 = _seed(store, phase="active", hb_age_s=3600.0)
    assert _retire(store, conv3, str(uuid.uuid4())) is False
    assert {_status(store, op_id), _status(store, op2), _status(store, op3)} == {"running"}
