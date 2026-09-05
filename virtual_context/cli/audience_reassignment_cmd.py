"""Explicit database commands for private, audited audience manifests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def configure_parser(admin_sub):
    plan = admin_sub.add_parser(
        "plan-audience-reassignment",
        help="Write a private, exact audience reassignment manifest",
    )
    plan.add_argument("owner_conversation_id")
    plan.add_argument("from_audience")
    plan.add_argument("to_audience")
    plan.add_argument("--tenant-id", required=True)
    plan.add_argument("--expected-lifecycle-epoch", type=_nonnegative_int, required=True)
    plan.add_argument("--operation-id", required=True)
    apply = admin_sub.add_parser(
        "reassign-audience",
        help="Verify an audience manifest; write only with --apply",
    )
    apply.add_argument("--apply", action="store_true", help="Apply the audited manifest")
    for parser in (plan, apply):
        parser.add_argument("--manifest", required=True, help="Private JSON manifest path")
        database = parser.add_mutually_exclusive_group(required=True)
        database.add_argument("--sqlite-db", help="Existing SQLite database path")
        database.add_argument("--postgres-dsn", help="Explicit PostgreSQL connection string")
        database.add_argument(
            "--postgres-dsn-env",
            help="Name of an environment variable holding the PostgreSQL connection string",
        )


def _open_store(args):
    if args.sqlite_db:
        from ..storage.sqlite import SQLiteStore

        path = Path(args.sqlite_db).expanduser()
        if not path.is_file():
            raise ValueError("SQLite selection must name an existing database file")
        return SQLiteStore(path, initialize_schema=False)
    from ..storage.postgres import PostgresStore

    dsn = args.postgres_dsn
    if args.postgres_dsn_env:
        dsn = os.environ.get(args.postgres_dsn_env)
    if not dsn:
        raise ValueError("PostgreSQL selection must provide a nonempty connection string")
    return PostgresStore(dsn, initialize_schema=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("manifest contains a duplicate JSON key")
        result[key] = value
    return result


def _load_manifest(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_unique_object)
    if type(value) is not dict:
        raise ValueError("manifest must be a JSON object")
    return value


def _write_manifest(path, manifest):
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def cmd_admin_audience_reassignment(args):
    """Use relational storage directly, without an engine or model provider."""
    store = None
    stage = "manifest"
    try:
        path = Path(args.manifest).expanduser()
        planning = args.admin_command == "plan-audience-reassignment"
        if planning:
            if path.exists() or path.is_symlink():
                raise FileExistsError("manifest already exists")
            manifest = None
        else:
            manifest = _load_manifest(path)
        stage = "database"
        store = _open_store(args)
        if planning:
            stage = "plan"
            manifest = store.plan_audience_reassignment(
                args.owner_conversation_id, args.from_audience, args.to_audience,
                tenant_id=args.tenant_id,
                expected_lifecycle_epoch=args.expected_lifecycle_epoch,
                operation_id=args.operation_id,
            )
            stage = "manifest"
            _write_manifest(path, manifest)
            report = {
                "status": "planned", "manifest": str(path.resolve()),
                "manifest_digest": manifest["manifest_digest"],
                "selected": len(manifest["rows"]), "updated": 0,
            }
        else:
            stage = "apply"
            report = store.reassign_audience(manifest, dry_run=not args.apply)
            report = {"status": "verified" if report["dry_run"] else "applied", **report}
    except Exception as exc:
        report = {"status": "error", "stage": stage, "error_type": type(exc).__name__}
        # Connection errors may embed credentials. Only the administrative
        # contract's controlled validation messages are safe to print here.
        if stage in {"plan", "apply"} and isinstance(exc, (ValueError, RuntimeError)):
            report["reason"] = str(exc)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(1) from None
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                print("Audience command database cleanup failed.", file=sys.stderr)
    print(json.dumps(report, sort_keys=True, indent=2))
