"""A stale compaction operation whose conversation is no longer compacting is retired.

The stale-lease sweeper takes over running compaction operations whose
heartbeat has aged out. When the conversation's phase is no longer
``compacting`` there is nothing to resume, and the row must leave the
running state or every sweeper tick finds it again.
"""

from __future__ import annotations

import pytest

from tests.test_find_stale_lease_helpers import _make_store, _seed_compaction, _seed_conversation


def _status(store, op_id):
    with store._get_conn() as c:
        return c.execute(
            "SELECT status, error_message FROM compaction_operation WHERE operation_id=?", (op_id,),
        ).fetchone()


def _retire(store, op_id, conv="conv-1", **overrides):
    kwargs = dict(
        operation_id=op_id, conversation_id=conv, lifecycle_epoch=1,
        stale_after_s=60.0, error_message="orphaned: conversation not compacting",
    )
    kwargs.update(overrides)
    return store.fail_orphaned_compaction_operation(**kwargs)


@pytest.mark.regression("BUG-080")
def test_stale_operation_on_an_active_conversation_is_failed(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-1", phase="active")
    op_id = _seed_compaction(store, "conv-1", hb_age_s=3600.0)

    assert _retire(store, op_id) is True
    status, message = _status(store, op_id)
    assert status == "failed" and "not compacting" in message
    assert not any(r["operation_id"] == op_id for r in store.find_stale_compaction_operations(grace_s=60.0))


@pytest.mark.regression("BUG-080")
def test_a_conversation_still_compacting_is_left_for_takeover(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-1", phase="compacting")
    op_id = _seed_compaction(store, "conv-1", hb_age_s=3600.0)
    assert _retire(store, op_id) is False
    assert _status(store, op_id)[0] == "running"


@pytest.mark.regression("BUG-080")
def test_a_fresh_heartbeat_is_left_alone(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-1", phase="active")
    op_id = _seed_compaction(store, "conv-1", hb_age_s=5.0)
    assert _retire(store, op_id) is False
    assert _status(store, op_id)[0] == "running"


@pytest.mark.regression("BUG-080")
def test_only_the_named_operation_and_epoch_are_touched(tmp_path):
    store = _make_store(tmp_path)
    _seed_conversation(store, "conv-1", phase="active")
    op_id = _seed_compaction(store, "conv-1", hb_age_s=3600.0)
    assert _retire(store, "some-other-operation") is False
    assert _retire(store, op_id, lifecycle_epoch=2) is False
    assert _status(store, op_id)[0] == "running"
