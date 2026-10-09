"""A process prepares a database's schema once, not once per store."""
from __future__ import annotations

import pytest

from virtual_context.storage import postgres


class _FakePool:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


@pytest.mark.regression("BUG-122")
def test_schema_is_prepared_once_per_database(monkeypatch):
    calls = []
    monkeypatch.setattr(postgres, "ConnectionPool", _FakePool)
    monkeypatch.setattr(postgres.PostgresStore, "_ensure_schema", lambda self: calls.append(self.dsn))
    monkeypatch.setattr(postgres, "_SCHEMA_READY", set())

    postgres.PostgresStore(dsn="postgresql://a/db1")
    postgres.PostgresStore(dsn="postgresql://a/db1")
    postgres.PostgresStore(dsn="postgresql://a/db2")

    assert calls == ["postgresql://a/db1", "postgresql://a/db2"]


@pytest.mark.regression("BUG-122")
def test_a_failed_schema_preparation_is_retried(monkeypatch):
    attempts = []

    def ensure(self):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(postgres, "ConnectionPool", _FakePool)
    monkeypatch.setattr(postgres.PostgresStore, "_ensure_schema", ensure)
    monkeypatch.setattr(postgres, "_SCHEMA_READY", set())

    with pytest.raises(RuntimeError):
        postgres.PostgresStore(dsn="postgresql://a/db1")
    postgres.PostgresStore(dsn="postgresql://a/db1")
    assert len(attempts) == 2
