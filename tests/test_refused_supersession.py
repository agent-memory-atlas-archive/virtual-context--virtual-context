"""A supersession the ledger refused is not proposed again on the same sources.

``fact_decisions`` recorded every refused supersession and nothing read it, so
an unchanged pair was proposed, judged and refused again on every pass. The
candidate selection now consults the ledger and drops a pair refused under the
current policy for the same fact versions and source versions.
"""

from __future__ import annotations

import pytest

from virtual_context.core.fact_lifecycle import AdmissionDecision
from virtual_context.ingest import supersession
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact

CONV = "conv-refusals"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = SQLiteStore(tmp_path / "s.db")
    # The candidate pre-check and the ledger apply the same policy, so in
    # normal operation a refused pair rarely reaches the ledger twice; open the
    # pre-check to observe the ledger consultation on its own.
    monkeypatch.setattr(supersession, "decide_supersession",
                        lambda *a, **k: AdmissionDecision(True, "test"))
    yield s
    s.close()


def _facts(store):
    old = Fact(id="old", subject="alice", verb="lives in", object="Boston", conversation_id=CONV)
    new = Fact(id="new", subject="bob", verb="lives in", object="Denver", conversation_id=CONV)
    store.store_facts([old, new])
    return _fact(store, "old"), _fact(store, "new")


def _fact(store, fact_id):
    return next(f for f in store.query_facts(conversation_id=CONV, limit=10) if f.id == fact_id)


def _candidates(store, new, old):
    _snapshot, admitted, _ = supersession._admitted_snapshots(store, new, [old])
    return [fact.id for fact in admitted]


@pytest.mark.regression("BUG-111")
def test_a_refused_pair_is_not_proposed_again(store):
    old, new = _facts(store)
    assert _candidates(store, new, old) == ["old"]
    assert store.set_fact_superseded("old", "new") is False  # subject_mismatch, recorded
    assert _candidates(store, new, old) == []


@pytest.mark.regression("BUG-111")
def test_a_changed_fact_is_a_new_proposal(store):
    old, new = _facts(store)
    store.set_fact_superseded("old", "new")
    store.update_fact_fields("old", "lives in", "Cambridge", "active", "")
    assert _candidates(store, new, _fact(store, "old")) == ["old"]


@pytest.mark.regression("BUG-111")
def test_a_stale_snapshot_refusal_does_not_block(store):
    old, new = _facts(store)
    assert store.set_fact_superseded("old", "new", expected_old_version="stale") is False
    assert _candidates(store, new, old) == ["old"]
