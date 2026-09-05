"""BUG-070: explicit, private administrative audience manifests."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from virtual_context.cli import audience_reassignment_cmd as command
from virtual_context.cli.main import main
from virtual_context.storage.sqlite import SQLiteStore


pytestmark = pytest.mark.regression("BUG-070")
OWNER = "synthetic-cli-owner"
SOURCE = "synthetic-cli-source"
TENANT = "synthetic-cli-tenant"


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source.db"
    store = SQLiteStore(path)
    for conversation in (OWNER, SOURCE):
        store.activate_conversation(conversation)
        store.upsert_conversation(tenant_id=TENANT, conversation_id=conversation)
    store._get_conn().execute(
        "UPDATE conversations SET phase = 'active' WHERE conversation_id = ?", (OWNER,),
    )
    store.save_conversation_alias(SOURCE, OWNER)
    from tests.test_audience_reassignment import _ingest
    from virtual_context.config import VirtualContextConfig
    from virtual_context.core.ingest_reconciler import IngestReconciler
    from virtual_context.core.semantic_search import SemanticSearchManager
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id=OWNER, storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    _ingest(IngestReconciler(store=store, semantic=semantic), owner=OWNER,
            audience=SOURCE, body="Synthetic historical observation.")
    yield path, store
    store.close()


def _run(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["virtual-context", "admin", *map(str, args)])
    main()


def _plan_args(database, manifest):
    return (
        "plan-audience-reassignment", OWNER, SOURCE, OWNER,
        "--tenant-id", TENANT, "--expected-lifecycle-epoch", "1",
        "--operation-id", "synthetic-cli-operation", "--manifest", manifest,
        "--sqlite-db", database,
    )


def test_plan_cli_creates_private_manifest_without_source_changes(
    database, tmp_path, monkeypatch, capsys,
):
    path, store = database
    manifest = tmp_path / "manifest.json"
    before = tuple(store._get_conn().iterdump())
    _run(monkeypatch, *_plan_args(path, manifest))
    report = json.loads(capsys.readouterr().out)
    assert report["selected"] == 2
    assert report["updated"] == 0
    assert report["status"] == "planned"
    assert manifest.stat().st_mode & 0o777 == 0o600
    saved = json.loads(manifest.read_text())
    assert saved["owner_conversation_id"] == OWNER
    assert saved["from_audience"] == SOURCE
    assert tuple(store._get_conn().iterdump()) == before


def test_apply_cli_defaults_to_dry_run_and_requires_apply_flag(
    database, tmp_path, monkeypatch, capsys,
):
    path, store = database
    manifest = tmp_path / "manifest.json"
    _run(monkeypatch, *_plan_args(path, manifest))
    capsys.readouterr()
    before = tuple(store._get_conn().iterdump())
    arguments = ("reassign-audience", "--manifest", manifest, "--sqlite-db", path)
    _run(monkeypatch, *arguments)
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True and report["updated"] == 0
    assert tuple(store._get_conn().iterdump()) == before
    _run(monkeypatch, *arguments, "--apply")
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is False and report["updated"] == 2
    row = store._get_conn().execute(
        "SELECT audience_conversation_id,user_content FROM canonical_turns "
        "WHERE user_content <> ''",
    ).fetchone()
    assert tuple(row) == (OWNER, "Synthetic historical observation.")


def test_plan_cli_refuses_existing_manifest_without_overwriting(
    database, tmp_path, monkeypatch, capsys,
):
    path, store = database
    manifest = tmp_path / "manifest.json"
    manifest.write_text("existing private artifact")
    before = tuple(store._get_conn().iterdump())
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, *_plan_args(path, manifest))
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert manifest.read_text() == "existing private artifact"
    assert tuple(store._get_conn().iterdump()) == before


def test_invalid_manifest_is_refused_before_database_connection(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "invalid.json"
    manifest.write_text('{"version":1,"version":2}')
    monkeypatch.setattr(command, "_open_store", lambda _args: pytest.fail("database opened"))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "reassign-audience", "--manifest", manifest,
             "--sqlite-db", tmp_path / "unused.db")
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_cli_requires_explicit_database_selection(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "reassign-audience", "--manifest", tmp_path / "manifest.json")
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_database_error_output_never_contains_connection_secret(
    tmp_path, monkeypatch, capsys,
):
    secret = "postgresql://synthetic:synthetic-password@example.invalid/db"

    def fail(_args):
        raise RuntimeError(f"cannot connect to {secret}")

    monkeypatch.setattr(command, "_open_store", fail)
    arguments = list(_plan_args(tmp_path / "unused.db", tmp_path / "manifest.json"))
    arguments[-2:] = ["--postgres-dsn", secret]
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, *arguments)
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert "synthetic-password" not in output.out + output.err
    assert json.loads(output.out)["stage"] == "database"


def test_postgres_environment_selector_passes_dsn_only_to_store(monkeypatch):
    from virtual_context.storage import postgres

    seen = []
    sentinel = object()

    def factory(dsn, *, initialize_schema):
        assert initialize_schema is False
        seen.append(dsn)
        return sentinel

    monkeypatch.setattr(postgres, "PostgresStore", factory)
    monkeypatch.setenv("SYNTHETIC_AUDIENCE_DSN", "postgresql://example.invalid/synthetic")
    result = command._open_store(SimpleNamespace(
        sqlite_db=None, postgres_dsn=None, postgres_dsn_env="SYNTHETIC_AUDIENCE_DSN",
    ))
    assert result is sentinel
    assert seen == ["postgresql://example.invalid/synthetic"]


def test_sqlite_selector_refuses_missing_database(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="existing"):
        command._open_store(SimpleNamespace(
            sqlite_db=str(missing), postgres_dsn=None, postgres_dsn_env=None,
        ))
    assert not missing.exists()


def test_cli_refuses_missing_schema_without_running_migrations(tmp_path, monkeypatch, capsys):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE existing_data(value TEXT)")
        conn.execute("INSERT INTO existing_data VALUES ('synthetic retained row')")
        before = tuple(conn.iterdump())
    before_bytes = path.read_bytes()
    manifest = tmp_path / "manifest.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, *_plan_args(path, manifest))
    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "database"
    assert not manifest.exists()
    assert path.read_bytes() == before_bytes
    with sqlite3.connect(path) as conn:
        assert tuple(conn.iterdump()) == before
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_plan_and_verify_keep_non_wal_database_bytes_and_journal_mode(
    database, tmp_path, monkeypatch, capsys,
):
    path, store = database
    store.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    before = path.read_bytes()
    manifest = tmp_path / "non-wal-manifest.json"
    _run(monkeypatch, *_plan_args(path, manifest))
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    _run(monkeypatch, "reassign-audience", "--manifest", manifest, "--sqlite-db", path)
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not path.with_name(path.name + "-wal").exists()


@pytest.mark.parametrize("trigger", [
    "trg_guard_attested_canonical_turn_update",
    "trg_invalidate_actor_card_turn_source_update",
])
def test_cli_refuses_stale_scope_trigger_without_migrating(
    database, tmp_path, monkeypatch, capsys, trigger,
):
    path, store = database
    conn = store._get_conn()
    conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute(f"CREATE TRIGGER {trigger} BEFORE UPDATE ON canonical_turns "
                 "BEGIN SELECT 1; END")
    before = tuple(conn.iterdump())
    manifest = tmp_path / "stale-trigger.json"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, *_plan_args(path, manifest))
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["stage"] == "database"
    assert not manifest.exists()
    assert tuple(conn.iterdump()) == before


def test_no_bootstrap_store_refuses_missing_file_without_creating_parent(tmp_path):
    missing = tmp_path / "absent-parent" / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteStore(missing, initialize_schema=False)
    assert not missing.parent.exists()
