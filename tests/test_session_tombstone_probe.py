"""Tombstone checks read the stored state's tail, not the whole state.

Lifecycle checks only need the ``deleted`` flag, but loading the
authoritative state fetched and parsed every conversation's full session
state (several MB for large conversations) to read it. ``to_json`` writes
``deleted`` as the last key, so the stored value always ends in the flag.
"""

from __future__ import annotations

import fakeredis
import pytest

from virtual_context.proxy.session_state import SessionState, SessionStateProvider


class _CountingRedis(fakeredis.FakeRedis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_reads = 0

    def get(self, name):
        self.full_reads += 1
        return super().get(name)


def _provider():
    client = _CountingRedis(decode_responses=False)
    return SessionStateProvider(redis_client=client, store=None), client


@pytest.mark.regression("BUG-092")
def test_a_live_state_is_not_deleted_without_a_full_read():
    provider, client = _provider()
    client.set(
        provider._key("conv"),
        SessionState(session_state="x" * 100_000, version=3).to_json(),
    )
    client.full_reads = 0
    assert provider.is_deleted_authoritative("conv") is False
    assert client.full_reads == 0


@pytest.mark.regression("BUG-092")
def test_a_tombstone_is_deleted_without_a_full_read():
    provider, client = _provider()
    provider.publish_tombstone("conv")
    client.full_reads = 0
    assert provider.is_deleted_authoritative("conv") is True
    assert client.full_reads == 0


@pytest.mark.regression("BUG-092")
def test_a_missing_state_is_not_deleted():
    provider, _client = _provider()
    assert provider.is_deleted_authoritative("absent") is False


@pytest.mark.regression("BUG-092")
@pytest.mark.parametrize("deleted", [True, False])
def test_an_unrecognized_layout_falls_back_to_the_full_state(deleted):
    import json

    provider, client = _provider()
    body = json.loads(SessionState(deleted=deleted, version=1).to_json())
    reordered = {"deleted": body.pop("deleted"), **body}
    client.set(provider._key("conv"), json.dumps(reordered).encode())
    assert provider.is_deleted_authoritative("conv") is deleted


@pytest.mark.regression("BUG-092")
def test_a_redis_failure_is_raised_not_read_as_live():
    provider, client = _provider()

    def _fail(*args, **kwargs):
        raise ConnectionError("redis down")

    client.getrange = _fail
    with pytest.raises(ConnectionError):
        provider.is_deleted_authoritative("conv")


@pytest.mark.regression("BUG-092")
@pytest.mark.parametrize("deleted", [True, False])
def test_a_compact_layout_is_read_from_the_tail(deleted):
    import json

    provider, client = _provider()
    body = json.loads(SessionState(deleted=deleted, version=1).to_json())
    client.set(
        provider._key("conv"),
        json.dumps(body, separators=(",", ":")).encode(),
    )
    client.full_reads = 0
    assert provider.is_deleted_authoritative("conv") is deleted
    assert client.full_reads == 0
