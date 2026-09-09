"""BUG-073: explicit, private and non-migrating channel enrichment commands."""

from __future__ import annotations

import json
import sqlite3

import pytest

from virtual_context.cli.main import main

pytestmark = pytest.mark.regression("BUG-073")


@pytest.fixture
def database(tmp_path):
    from tests.test_audience_reassignment_cli import database as source_fixture

    yield from source_fixture.__wrapped__(tmp_path)


def _assistant_id(store):
    return (
        store._get_conn()
        .execute(
            "SELECT canonical_turn_id FROM canonical_turns WHERE assistant_content <> ''",
        )
        .fetchone()[0]
    )


def _run(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["virtual-context", "admin", *map(str, args)])
    main()


def _plan_args(path, manifest, assistant_id):
    from tests.test_audience_reassignment_cli import OWNER, SOURCE, TENANT

    return (
        "plan-assistant-channel-enrichment",
        OWNER,
        SOURCE,
        "--tenant-id",
        TENANT,
        "--expected-lifecycle-epoch",
        "1",
        "--operation-id",
        "synthetic-channel-enrichment-cli",
        "--assistant-canonical-turn-id",
        assistant_id,
        "--manifest",
        manifest,
        "--sqlite-db",
        path,
    )


def test_plan_and_default_verify_preserve_non_wal_bytes(database, tmp_path, monkeypatch, capsys):
    path, store = database
    assistant_id = _assistant_id(store)
    store._get_conn().execute(
        "UPDATE canonical_turns SET origin_channel_id = '' WHERE canonical_turn_id = ?",
        (assistant_id,),
    )
    store.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    before = path.read_bytes()
    manifest = tmp_path / "private-manifest.json"
    _run(monkeypatch, *_plan_args(path, manifest, assistant_id))
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert manifest.stat().st_mode & 0o777 == 0o600
    _run(monkeypatch, "enrich-assistant-channels", "--manifest", manifest, "--sqlite-db", path)
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True and report["updated"] == 0
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not path.with_name(path.name + "-wal").exists()


def test_apply_requires_flag_and_preserves_assistant_bytes(database, tmp_path, monkeypatch, capsys):
    path, store = database
    assistant_id = _assistant_id(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET origin_channel_id = '' WHERE canonical_turn_id = ?",
        (assistant_id,),
    )
    before = dict(
        conn.execute(
            "SELECT * FROM canonical_turns WHERE canonical_turn_id = ?", (assistant_id,)
        ).fetchone()
    )
    manifest = tmp_path / "manifest.json"
    _run(monkeypatch, *_plan_args(path, manifest, assistant_id))
    capsys.readouterr()
    _run(
        monkeypatch,
        "enrich-assistant-channels",
        "--manifest",
        manifest,
        "--sqlite-db",
        path,
        "--apply",
    )
    report = json.loads(capsys.readouterr().out)
    assert report["updated"] == 1 and report["dry_run"] is False
    after = dict(
        conn.execute(
            "SELECT * FROM canonical_turns WHERE canonical_turn_id = ?", (assistant_id,)
        ).fetchone()
    )
    assert after["origin_channel_id"]
    for field in before:
        if field != "origin_channel_id":
            assert after[field] == before[field], field


def test_plan_refuses_legacy_schema_without_migrating(tmp_path, monkeypatch, capsys):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE retained(value TEXT)")
        conn.execute("INSERT INTO retained VALUES ('synthetic')")
    before = path.read_bytes()
    manifest = tmp_path / "not-created.json"
    with pytest.raises(SystemExit) as failure:
        _run(monkeypatch, *_plan_args(path, manifest, "synthetic-id"))
    assert failure.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert path.read_bytes() == before and not manifest.exists()


def test_invalid_json_fails_before_database_open(tmp_path, monkeypatch, capsys):
    from virtual_context.cli import assistant_channel_enrichment_cmd as command

    manifest = tmp_path / "duplicate.json"
    manifest.write_text('{"version":1,"version":2}')
    monkeypatch.setattr(command, "_open_store", lambda _args: pytest.fail("database opened"))
    with pytest.raises(SystemExit) as failure:
        _run(
            monkeypatch,
            "enrich-assistant-channels",
            "--manifest",
            manifest,
            "--sqlite-db",
            tmp_path / "unused.db",
        )
    assert failure.value.code == 1
    assert json.loads(capsys.readouterr().out)["stage"] == "manifest"


def test_no_implicit_database_selection(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as failure:
        _run(monkeypatch, "enrich-assistant-channels", "--manifest", tmp_path / "manifest.json")
    assert failure.value.code == 2
    assert "required" in capsys.readouterr().err
