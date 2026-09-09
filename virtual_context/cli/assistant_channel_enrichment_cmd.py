"""Explicit database commands for source-proved assistant channel enrichment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from .audience_reassignment_cmd import (
    _load_manifest,
    _nonnegative_int,
    _open_store,
    _write_manifest,
)


def configure_parser(admin_sub):
    plan = admin_sub.add_parser(
        "plan-assistant-channel-enrichment",
        help="Write a private manifest for explicitly selected assistant source rows",
    )
    plan.add_argument("owner_conversation_id")
    plan.add_argument("audience_conversation_id")
    plan.add_argument("--tenant-id", required=True)
    plan.add_argument("--expected-lifecycle-epoch", type=_nonnegative_int, required=True)
    plan.add_argument("--operation-id", required=True)
    plan.add_argument(
        "--assistant-canonical-turn-id",
        action="append",
        required=True,
        help="Exact assistant canonical ID; repeat for each approved row",
    )
    verify = admin_sub.add_parser(
        "verify-assistant-channel-enrichment",
        help="Verify an existing manifest without writes",
    )
    apply = admin_sub.add_parser(
        "enrich-assistant-channels",
        help="Verify by default; change channels only with --apply",
    )
    apply.add_argument("--apply", action="store_true", help="Apply this exact approved manifest")
    for parser in (plan, verify, apply):
        parser.add_argument("--manifest", required=True, help="Private JSON manifest path")
        database = parser.add_mutually_exclusive_group(required=True)
        database.add_argument("--sqlite-db", help="Existing SQLite database path")
        database.add_argument("--postgres-dsn", help="Explicit PostgreSQL connection string")
        database.add_argument(
            "--postgres-dsn-env",
            help="Environment variable containing the PostgreSQL connection string",
        )


def cmd_admin_assistant_channel_enrichment(args):
    """Use an existing guarded database directly, without providers or migration."""
    store = None
    stage = "manifest"
    try:
        path = Path(args.manifest).expanduser()
        planning = args.admin_command == "plan-assistant-channel-enrichment"
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
            manifest = store.plan_assistant_channel_enrichment(
                args.owner_conversation_id,
                tenant_id=args.tenant_id,
                audience_conversation_id=args.audience_conversation_id,
                expected_lifecycle_epoch=args.expected_lifecycle_epoch,
                operation_id=args.operation_id,
                assistant_canonical_turn_ids=args.assistant_canonical_turn_id,
            )
            stage = "manifest"
            _write_manifest(path, manifest)
            report = {
                "status": "planned",
                "manifest": str(path.resolve()),
                "manifest_digest": manifest["manifest_digest"],
                "selected": len(manifest["rows"]),
                "updated": 0,
            }
        else:
            stage = "apply"
            report = store.enrich_assistant_channels(
                manifest,
                dry_run=not getattr(args, "apply", False),
            )
            report = {"status": "verified" if report["dry_run"] else "applied", **report}
    except Exception as exc:
        report = {"status": "error", "stage": stage, "error_type": type(exc).__name__}
        # Database/open errors may contain credentials; only controlled
        # administrative proof failures can be included in command output.
        if stage in {"plan", "apply"} and isinstance(exc, (ValueError, RuntimeError)):
            report["reason"] = str(exc)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(1) from None
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                print("Channel enrichment database cleanup failed.", file=sys.stderr)
    print(json.dumps(report, sort_keys=True, indent=2))
