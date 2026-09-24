"""A hydrated engine saves its session state back under the loaded version.

The provider's save is a compare-and-swap on the stored version. An
extracted snapshot that always carried version 0 was rejected by every save
after a conversation's first, so the shared state froze and every request
replayed every turn stored since.
"""

from __future__ import annotations

import fakeredis
import pytest

from virtual_context.engine import VirtualContextEngine
from virtual_context.proxy.session_state import SessionState, SessionStateProvider
from virtual_context.types import VirtualContextConfig

CONV = "conv-version"


def _engine(tmp_path, provider):
    config = VirtualContextConfig()
    config.conversation_id = CONV
    config.storage.backend = "sqlite"
    config.storage.sqlite_path = str(tmp_path / "store.db")
    return VirtualContextEngine(config=config, session_state_provider=provider)


@pytest.mark.regression("BUG-089")
def test_repeated_saves_from_hydrated_engines_are_accepted(tmp_path):
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)
    assert provider.save(CONV, SessionState(last_indexed_turn=10)) == 1

    engine = _engine(tmp_path, provider)
    for expected_indexed in (11, 12):
        loaded = provider.load(CONV)
        engine.hydrate_from_session_state(loaded)
        engine._engine_state.last_indexed_turn = expected_indexed
        saved = provider.save(CONV, engine.extract_session_state())
        assert saved is not None, "save rejected as stale"
        engine.note_session_state_saved(saved)
        assert provider.load(CONV).last_indexed_turn == expected_indexed


@pytest.mark.regression("BUG-089")
def test_a_second_save_without_rehydrating_is_accepted(tmp_path):
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)
    provider.save(CONV, SessionState(last_indexed_turn=10))
    engine = _engine(tmp_path, provider)
    engine.hydrate_from_session_state(provider.load(CONV))
    engine.note_session_state_saved(provider.save(CONV, engine.extract_session_state()))
    engine._engine_state.last_indexed_turn = 20
    assert provider.save(CONV, engine.extract_session_state()) is not None
    assert provider.load(CONV).last_indexed_turn == 20


@pytest.mark.regression("BUG-089")
def test_a_save_from_a_worker_behind_a_newer_save_is_still_rejected(tmp_path):
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)
    provider.save(CONV, SessionState(last_indexed_turn=10))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = _engine(tmp_path / "a", provider), _engine(tmp_path / "b", provider)
    a.hydrate_from_session_state(provider.load(CONV))
    b.hydrate_from_session_state(provider.load(CONV))
    a.note_session_state_saved(provider.save(CONV, a.extract_session_state()))
    assert provider.save(CONV, b.extract_session_state()) is None


@pytest.mark.regression("BUG-089")
def test_hydrating_a_version_the_engine_already_holds_keeps_its_newer_work(tmp_path):
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)
    provider.save(CONV, SessionState(last_indexed_turn=10))
    engine = _engine(tmp_path, provider)
    engine.hydrate_from_session_state(provider.load(CONV))
    engine._engine_state.last_indexed_turn = 15  # background work not yet saved
    engine.hydrate_from_session_state(provider.load(CONV))
    assert engine._engine_state.last_indexed_turn == 15

    other = SessionState(last_indexed_turn=30, version=provider.load(CONV).version)
    provider.save(CONV, other)  # another worker saves a newer version
    engine.hydrate_from_session_state(provider.load(CONV))
    assert engine._engine_state.last_indexed_turn == 30
