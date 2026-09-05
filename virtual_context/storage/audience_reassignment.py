"""Audited administrative changes to effective canonical memory audiences.

Audience names are opaque client choices. A reassignment changes placement,
never the original source ledger. Plans pin exact rows and immutable evidence;
the same transaction records the authorization before changing canonical scope.
Only exact attested pairs and one reassignment per canonical row are supported.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
import hashlib
import json
import uuid

from ..core.canonical_turns import utcnow_iso
from ..types import AUDIENCE_ATTRIBUTION_VERSION


_VERSION = 1
_HEADERS = (
    "operation_id", "tenant_id", "owner_conversation_id", "from_audience",
    "to_audience", "expected_lifecycle_epoch",
)
_ROW_FIELDS = (
    "canonical_turn_id", "from_audience", "to_audience", "turn_hash",
    "from_attribution_version", "to_attribution_version", "row_fingerprint",
    "source_fingerprint",
)
# Owner, ordinal, sort key, tag/compaction state and timestamps may change during
# normal maintenance. Audience/version are pinned separately because they are
# precisely the fields this operation is authorized to change.
_BODY_FIELDS = (
    "canonical_turn_id", "turn_hash", "hash_version", "normalized_user_text",
    "normalized_assistant_text", "user_content", "assistant_content",
    "user_raw_content", "assistant_raw_content", "sender", "origin_channel_id",
    "origin_channel_label", "sender_actor_id", "source_message_id",
    "reply_target_message_id", "origin_conversation_id",
)


def _placeholder(dialect):
    if dialect not in {"sqlite", "postgres"}:
        raise ValueError("unsupported audience reassignment dialect")
    return "%s" if dialect == "postgres" else "?"


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record(row):
    return {key: str(value) if isinstance(value, uuid.UUID) else value
            for key, value in dict(row).items()}


def canonical_row_fingerprint(row):
    """Hash immutable canonical body and physical-source projection fields."""
    values = dict(row)
    return _digest({key: values.get(key) for key in _BODY_FIELDS})


def source_membership_fingerprint(memberships):
    """Hash original source records, excluding their replay observation time."""
    records = [
        {key: value for key, value in dict(row).items() if key != "observed_at"}
        for row in memberships
    ]
    records.sort(key=lambda row: (
        row["tenant_id"], row["agent_scope_id"], row["platform"],
        row["account_id"], row["message_id"],
    ))
    return _digest(records)


def source_fingerprint(membership):
    """Fingerprint one immutable source membership for normal proof readers."""
    return source_membership_fingerprint([membership])


@contextmanager
def _transaction(store, dialect):
    _placeholder(dialect)
    if dialect == "postgres":
        with store.pool.connection() as conn, conn.transaction():
            yield conn
        return
    conn = store._get_conn()
    owned = not conn.in_transaction
    savepoint = "audience_reassignment_" + uuid.uuid4().hex
    conn.execute("BEGIN IMMEDIATE" if owned else f"SAVEPOINT {savepoint}")
    try:
        yield conn
        if owned:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except BaseException:
        if owned:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


def ensure_audience_reassignment_schema(conn, dialect):
    """Install audit tables and immutable guards without committing the caller."""
    _placeholder(dialect)
    conn.execute("""CREATE TABLE IF NOT EXISTS audience_reassignment_operations (
        operation_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        owner_conversation_id TEXT NOT NULL,
        from_audience TEXT NOT NULL,
        to_audience TEXT NOT NULL,
        expected_lifecycle_epoch INTEGER NOT NULL,
        manifest_digest TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )""")
    canonical_id_type = "UUID" if dialect == "postgres" else "TEXT"
    conn.execute(f"""CREATE TABLE IF NOT EXISTS canonical_audience_reassignments (
        canonical_turn_id {canonical_id_type} PRIMARY KEY
            REFERENCES canonical_turns(canonical_turn_id) ON DELETE CASCADE,
        operation_id TEXT NOT NULL
            REFERENCES audience_reassignment_operations(operation_id),
        tenant_id TEXT NOT NULL,
        owner_conversation_id TEXT NOT NULL,
        from_audience TEXT NOT NULL,
        to_audience TEXT NOT NULL,
        from_attribution_version INTEGER NOT NULL,
        to_attribution_version INTEGER NOT NULL,
        turn_hash TEXT NOT NULL,
        source_fingerprint TEXT NOT NULL,
        row_fingerprint TEXT NOT NULL,
        manifest_digest TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_audience_reassignments_operation
        ON canonical_audience_reassignments(operation_id)""")
    if dialect == "sqlite":
        statements = (
            """CREATE TRIGGER IF NOT EXISTS trg_audience_reassignment_update
                BEFORE UPDATE ON canonical_audience_reassignments
                BEGIN SELECT RAISE(ABORT, 'audience reassignment is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS trg_audience_reassignment_delete
                BEFORE DELETE ON canonical_audience_reassignments
                WHEN EXISTS (SELECT 1 FROM canonical_turns
                    WHERE canonical_turn_id = OLD.canonical_turn_id)
                BEGIN SELECT RAISE(ABORT, 'live audience reassignment is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS trg_audience_operation_update
                BEFORE UPDATE ON audience_reassignment_operations
                BEGIN SELECT RAISE(ABORT, 'audience operation is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS trg_audience_operation_delete
                BEFORE DELETE ON audience_reassignment_operations
                WHEN EXISTS (SELECT 1 FROM canonical_audience_reassignments
                    WHERE operation_id = OLD.operation_id)
                BEGIN SELECT RAISE(ABORT, 'live audience operation is immutable'); END""",
        )
        for statement in statements:
            conn.execute(statement)
        return
    conn.execute("""CREATE OR REPLACE FUNCTION guard_audience_reassignment()
        RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'UPDATE' OR EXISTS (
                SELECT 1 FROM canonical_turns
                WHERE canonical_turn_id = OLD.canonical_turn_id
            ) THEN
                RAISE EXCEPTION 'audience reassignment is immutable';
            END IF;
            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql""")
    conn.execute("""CREATE OR REPLACE FUNCTION guard_audience_operation()
        RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'UPDATE' OR EXISTS (
                SELECT 1 FROM canonical_audience_reassignments
                WHERE operation_id = OLD.operation_id
            ) THEN
                RAISE EXCEPTION 'audience operation is immutable';
            END IF;
            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql""")
    for name, table, function in (
        ("trg_audience_reassignment_guard", "canonical_audience_reassignments",
         "guard_audience_reassignment"),
        ("trg_audience_operation_guard", "audience_reassignment_operations",
         "guard_audience_operation"),
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name} ON {table}")
        conn.execute(f"CREATE TRIGGER {name} BEFORE UPDATE OR DELETE ON {table} "
                     f"FOR EACH ROW EXECUTE FUNCTION {function}()")


def assert_audience_reassignment_schema(conn, dialect):
    """Fail closed when a best-effort bootstrap left incomplete audit guards."""
    _placeholder(dialect)
    required = {
        "audience_reassignment_operations": {
            *_HEADERS, "manifest_digest", "row_count", "created_at",
        },
        "canonical_audience_reassignments": {
            *_ROW_FIELDS, "operation_id", "tenant_id", "owner_conversation_id",
            "manifest_digest", "created_at",
        },
    }
    expected_keys = {
        "audience_reassignment_operations": "operation_id",
        "canonical_audience_reassignments": "canonical_turn_id",
    }
    for table, columns in required.items():
        if dialect == "sqlite":
            schema = conn.execute(f"PRAGMA table_info({table})").fetchall()
            present = {row["name"] for row in schema}
            primary = {row["name"] for row in schema if row["pk"]}
            if primary != {expected_keys[table]}:
                raise RuntimeError(f"{table} has an unsafe primary key")
        else:
            schema = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s", (table,),
            ).fetchall()
            present = {row["column_name"] for row in schema}
            keys = conn.execute(
                "SELECT a.attname FROM pg_constraint c "
                "JOIN pg_attribute a ON a.attrelid = c.conrelid "
                "AND a.attnum = ANY(c.conkey) "
                "WHERE c.conrelid = %s::regclass AND c.contype = 'p'", (table,),
            ).fetchall() if present else []
            if {row["attname"] for row in keys} != {expected_keys[table]}:
                raise RuntimeError(f"{table} has an unsafe primary key")
        if not columns.issubset(present):
            raise RuntimeError(f"{table} is missing required audit columns")
    if dialect == "sqlite":
        foreign = conn.execute(
            "PRAGMA foreign_key_list(canonical_audience_reassignments)",
        ).fetchall()
        canonical_fk = any(
            row["from"] == "canonical_turn_id" and row["to"] == "canonical_turn_id"
            and row["table"] == "canonical_turns" and row["on_delete"].upper() == "CASCADE"
            for row in foreign
        )
        operation_fk = any(
            row["from"] == "operation_id" and row["to"] == "operation_id"
            and row["table"] == "audience_reassignment_operations"
            and row["on_delete"].upper() in {"NO ACTION", "RESTRICT"}
            for row in foreign
        )
        expected_triggers = {
            "trg_audience_reassignment_update", "trg_audience_reassignment_delete",
            "trg_audience_operation_update", "trg_audience_operation_delete",
        }
        triggers = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND tbl_name IN ('canonical_audience_reassignments', "
            "'audience_reassignment_operations')",
        ).fetchall()}
    else:
        foreign = conn.execute(
            "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid = 'canonical_audience_reassignments'::regclass "
            "AND contype = 'f'",
        ).fetchall()
        definitions = {row["definition"] for row in foreign}
        canonical_fk = any(
            "FOREIGN KEY (canonical_turn_id) REFERENCES canonical_turns(canonical_turn_id)"
            in definition and "ON DELETE CASCADE" in definition for definition in definitions
        )
        operation_fk = any(
            "FOREIGN KEY (operation_id) REFERENCES audience_reassignment_operations(operation_id)"
            in definition and "ON DELETE CASCADE" not in definition
            for definition in definitions
        )
        expected_triggers = {"trg_audience_reassignment_guard", "trg_audience_operation_guard"}
        triggers = {row["tgname"] for row in conn.execute(
            "SELECT tgname FROM pg_trigger WHERE tgrelid IN "
            "('canonical_audience_reassignments'::regclass, "
            "'audience_reassignment_operations'::regclass) "
            "AND NOT tgisinternal AND tgenabled <> 'D'",
        ).fetchall()}
    if not canonical_fk or not operation_fk:
        raise RuntimeError("audience reassignment foreign keys are missing or unsafe")
    if not expected_triggers.issubset(triggers):
        raise RuntimeError("audience reassignment immutability guards are missing")
    # Table presence alone does not establish migration capability: an older
    # canonical guard can refuse the scope change, and an older card trigger
    # can silently delete durable evidence. Inspect the installed bodies before
    # exposing the administrative API without normal schema bootstrap.
    if dialect == "sqlite":
        scope_triggers = (
            "trg_guard_attested_canonical_turn_update",
            "trg_invalidate_actor_card_turn_source_update",
        )
        definitions = {
            row["name"]: row["sql"] or "" for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
                "AND tbl_name='canonical_turns' AND name IN (?, ?)", scope_triggers,
            ).fetchall()
        }
    else:
        scope_triggers = (
            "trg_guard_attested_canonical_turn_update",
            "trg_invalidate_actor_card_turn_source",
        )
        definitions = {
            row["tgname"]: row["definition"] or "" for row in conn.execute(
                "SELECT tgname, pg_get_functiondef(tgfoid) AS definition "
                "FROM pg_trigger WHERE tgrelid='canonical_turns'::regclass "
                "AND tgname IN (%s, %s) AND NOT tgisinternal AND tgenabled <> 'D'",
                scope_triggers,
            ).fetchall()
        }
    if any("canonical_audience_reassignments" not in definitions.get(name, "").lower()
           for name in scope_triggers):
        raise RuntimeError("canonical audience reassignment trigger capability is missing")


def get_audience_reassignment(conn, canonical_turn_id, dialect):
    """Return the immutable authorization for an exact canonical row, if any."""
    p = _placeholder(dialect)
    row = conn.execute(
        f"SELECT * FROM canonical_audience_reassignments WHERE canonical_turn_id = {p}",
        (canonical_turn_id,),
    ).fetchone()
    return _record(row) if row is not None else None


def _validate_headers(headers):
    for key in _HEADERS[:-1]:
        value = headers.get(key)
        if type(value) is not str or not value or value != value.strip():
            raise ValueError(f"{key} must be a nonempty exact identifier")
    epoch = headers.get("expected_lifecycle_epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("expected_lifecycle_epoch must be a nonnegative integer")
    if headers["from_audience"] == headers["to_audience"]:
        raise ValueError("source and target audiences must differ")


def _fence(store, conn, headers, dialect):
    p = _placeholder(dialect)
    owner = headers["owner_conversation_id"]
    ids = sorted({owner, headers["from_audience"], headers["to_audience"]})
    placeholders = ",".join(p for _ in ids)
    locking = " FOR UPDATE" if dialect == "postgres" else ""
    lifecycle = conn.execute(
        f"SELECT * FROM conversation_lifecycle WHERE conversation_id IN ({placeholders}) "
        f"ORDER BY conversation_id{locking}", ids,
    ).fetchall()
    if {row["conversation_id"] for row in lifecycle} != set(ids):
        raise ValueError("audience reassignment requires existing lifecycle records")
    records = conn.execute(
        f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders}) "
        f"ORDER BY conversation_id{locking}", ids,
    ).fetchall()
    by_id = {row["conversation_id"]: dict(row) for row in records}
    if set(by_id) != set(ids):
        raise ValueError("audience reassignment requires existing conversations")
    for row in by_id.values():
        if row["tenant_id"] != headers["tenant_id"]:
            raise ValueError("audience reassignment cannot cross tenants")
        if row["deleted_at"] is not None or row["phase"] == "deleted":
            raise ValueError("audience reassignment cannot use deleted conversations")
    owner_row = by_id[owner]
    if (owner_row["phase"] != "active"
            or int(owner_row["lifecycle_epoch"]) != headers["expected_lifecycle_epoch"]):
        raise ValueError("owner lifecycle epoch or active phase changed")
    if any(bool(row["deleted"]) for row in lifecycle):
        raise ValueError("owner or audience lifecycle is deleted")
    if store._resolve_owner(conn, owner) != owner:
        raise ValueError("audience reassignment owner must be a terminal owner")
    # A direct retained alias has one lockable identity. Supporting arbitrary
    # chains would require fencing each intermediate lifecycle and alias epoch;
    # reject those rather than accepting an unlocked concurrent redirection.
    for audience in ids:
        if audience == owner:
            continue
        alias = conn.execute(
            f"SELECT target_id FROM conversation_aliases WHERE alias_id = {p}{locking}",
            (audience,),
        ).fetchone()
        if alias is None or alias["target_id"] != owner:
            raise ValueError("source and target audiences must be direct aliases of the owner")
    if int(owner_row["pending_raw_payload_entries"] or 0):
        raise RuntimeError("audience reassignment requires drained raw ingestion")
    for table, predicate in (
        ("compaction_operation", "status IN ('queued','running')"),
        ("ingestion_episode", "status = 'running'"),
    ):
        active = conn.execute(
            f"SELECT 1 FROM {table} WHERE conversation_id IN ({placeholders}) "
            f"AND {predicate} LIMIT 1", ids,
        ).fetchone()
        if active is not None:
            raise RuntimeError(f"audience reassignment has an active {table}")
    merging = conn.execute(
        f"SELECT 1 FROM merge_audit WHERE tenant_id = {p} AND status = 'in_progress' "
        f"AND (source_conversation_id IN ({placeholders}) "
        f"OR target_conversation_id IN ({placeholders})) LIMIT 1",
        [headers["tenant_id"], *ids, *ids],
    ).fetchone()
    if merging is not None:
        raise RuntimeError("audience reassignment has an active merge")


def _memberships(conn, ids, dialect):
    p = _placeholder(dialect)
    found = {}
    ordered = sorted(ids)
    for start in range(0, len(ordered), 400):
        chunk = ordered[start:start + 400]
        placeholders = ",".join(p for _ in chunk)
        rows = conn.execute(
            f"SELECT * FROM canonical_message_sources "
            f"WHERE canonical_turn_id IN ({placeholders}) "
            f"OR assistant_canonical_turn_id IN ({placeholders})",
            [*chunk, *chunk],
        ).fetchall()
        for row in rows:
            record = _record(row)
            key = tuple(record[name] for name in (
                "tenant_id", "agent_scope_id", "platform", "account_id", "message_id",
            ))
            found[key] = record
    return list(found.values())


def _load_rows(conn, ids, dialect):
    p = _placeholder(dialect)
    found = {}
    ordered = sorted(ids)
    locking = " FOR UPDATE" if dialect == "postgres" else ""
    for start in range(0, len(ordered), 400):
        chunk = ordered[start:start + 400]
        placeholders = ",".join(p for _ in chunk)
        for row in conn.execute(
            f"SELECT * FROM canonical_turns WHERE canonical_turn_id IN ({placeholders}) "
            f"ORDER BY canonical_turn_id{locking}", chunk,
        ).fetchall():
            record = _record(row)
            found[record["canonical_turn_id"]] = record
    if set(found) != set(ids):
        raise ValueError("manifest or source pair references missing canonical rows")
    return found


def _validate_memberships(rows, memberships, headers):
    owner = headers["owner_conversation_id"]
    paired_ids = set()
    for source in memberships:
        user = rows.get(source["canonical_turn_id"])
        assistant = rows.get(source["assistant_canonical_turn_id"])
        if user is None or assistant is None or user is assistant:
            raise ValueError("audience reassignment requires the complete exact source pair")
        ids = {source["canonical_turn_id"], source["assistant_canonical_turn_id"]}
        if paired_ids.intersection(ids):
            raise ValueError("canonical source belongs to overlapping pairs")
        paired_ids.update(ids)
        if source["tenant_id"] != headers["tenant_id"]:
            raise ValueError("source membership belongs to another tenant")
        if source["audience_conversation_id"] != headers["from_audience"]:
            raise ValueError("source pair does not originate in the selected audience")
        if user["conversation_id"] != owner or assistant["conversation_id"] != owner:
            raise ValueError("source pair belongs to another owner")
        if (assistant["audience_conversation_id"] or "") not in {
            "", headers["from_audience"], headers["to_audience"],
        }:
            raise ValueError("paired assistant has an unrelated audience")
        user_body = user["user_content"] or ""
        if (not user_body.strip() or (user["assistant_content"] or "").strip()
                or (assistant["user_content"] or "").strip()
                or not (assistant["assistant_content"] or "").strip()
                or int(source["pair_version"]) != 1
                or user["turn_hash"] != source["canonical_turn_hash"]
                or assistant["turn_hash"] != source["assistant_turn_hash"]
                or int(user["turn_group_number"]) != int(assistant["turn_group_number"])
                or hashlib.sha256(user_body.encode("utf-8")).hexdigest()
                != source["canonical_body_sha256"]
                or (user["sender_actor_id"] or "") != source["source_actor_id"]
                or (user["source_message_id"] or "") != source["message_id"]
                or (user["origin_channel_id"] or "") != source["channel_id"]
                or (user["reply_target_message_id"] or "")
                != source["reply_target_message_id"]):
            raise ValueError("source pair body, identity, or immutable proof disagrees")


def _source_map(rows, memberships):
    result = {canonical_id: [] for canonical_id in rows}
    for source in memberships:
        result[source["canonical_turn_id"]].append(source)
        result[source["assistant_canonical_turn_id"]].append(source)
    return result


def _fresh_rows(conn, headers, dialect):
    p = _placeholder(dialect)
    selected = conn.execute(
        f"SELECT canonical_turn_id FROM canonical_turns WHERE conversation_id = {p} "
        f"AND audience_conversation_id = {p} ORDER BY canonical_turn_id",
        (headers["owner_conversation_id"], headers["from_audience"]),
    ).fetchall()
    ids = {str(row["canonical_turn_id"]) for row in selected}
    memberships = _memberships(conn, ids, dialect)
    for source in memberships:
        if not source["assistant_canonical_turn_id"]:
            raise ValueError("source membership is missing its exact assistant pair")
        ids.add(source["canonical_turn_id"])
        ids.add(source["assistant_canonical_turn_id"])
    # This also detects a malformed row that is simultaneously one pair's user
    # and another's assistant; do not silently ignore its second membership.
    complete_memberships = _memberships(conn, ids, dialect)
    if {row["canonical_turn_id"] for row in memberships} != {
        row["canonical_turn_id"] for row in complete_memberships
    }:
        raise ValueError("source pairs overlap across reassignment membership")
    rows = _load_rows(conn, ids, dialect)
    _validate_memberships(rows, memberships, headers)
    source_map = _source_map(rows, memberships)
    planned = []
    for canonical_id in sorted(rows):
        row = rows[canonical_id]
        if row["conversation_id"] != headers["owner_conversation_id"]:
            raise ValueError("canonical row belongs to another owner")
        if get_audience_reassignment(conn, canonical_id, dialect) is not None:
            raise ValueError("a canonical row already has an audience reassignment")
        audience = row["audience_conversation_id"] or ""
        version = int(row["audience_attribution_version"] or 0)
        sources = source_map[canonical_id]
        if not sources:
            # A legacy row could later acquire a new immutable membership,
            # invalidating the receipt's pinned source proof. It must first
            # complete ordinary exact-source admission, never be guessed here.
            raise ValueError("unattested canonical row cannot be reassigned")
        is_paired_assistant = bool(sources) and all(
            source["assistant_canonical_turn_id"] == canonical_id for source in sources
        )
        if audience not in {headers["from_audience"], headers["to_audience"]}:
            if not (is_paired_assistant and audience == ""):
                raise ValueError("paired source has an unsupported audience")
        if version != AUDIENCE_ATTRIBUTION_VERSION:
            # Older exact-pair receipts can name an assistant whose audience
            # was filled without its version. The attested user proves this
            # assistant's placement; an unrelated audience was refused above.
            if not (is_paired_assistant and version == 0):
                raise ValueError("canonical row has stale audience attribution")
        for source in sources:
            if source["canonical_turn_id"] == canonical_id:
                if source["audience_conversation_id"] != audience:
                    raise ValueError("canonical audience disagrees with original source")
        planned.append({
            "canonical_turn_id": canonical_id,
            "from_audience": audience,
            "to_audience": headers["to_audience"],
            "from_attribution_version": version,
            "to_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
            "turn_hash": row["turn_hash"],
            "row_fingerprint": canonical_row_fingerprint(row),
            "source_fingerprint": source_membership_fingerprint(sources),
        })
    return planned


def _operation(conn, operation_id, dialect):
    p = _placeholder(dialect)
    row = conn.execute(
        f"SELECT * FROM audience_reassignment_operations WHERE operation_id = {p}",
        (operation_id,),
    ).fetchone()
    return _record(row) if row is not None else None


def _invalidate_cards(conn, headers, dialect):
    """Hide stale cards while retaining immutable claims for re-admission.

    Reassignment changes placement, not the supporting source. Destructive
    cache invalidation would erase durable carryovers outside the fresh input
    window. Their exact source pointers must be reprojected before rebuilding.
    """
    p = _placeholder(dialect)
    affected = conn.execute(
        "SELECT DISTINCT e.id, e.actor_id FROM actor_card_entries e JOIN ("
        "SELECT entry_id, tenant_id, owner_conversation_id FROM actor_card_entry_sources "
        "UNION ALL SELECT entry_id, tenant_id, owner_conversation_id "
        "FROM actor_card_turn_sources) s ON s.entry_id=e.id AND s.tenant_id=e.tenant_id "
        f"WHERE e.tenant_id={p} AND s.owner_conversation_id={p}",
        (headers["tenant_id"], headers["owner_conversation_id"]),
    ).fetchall()
    for actor in sorted({row["actor_id"] for row in affected}):
        conn.execute(
            "UPDATE actor_profiles SET card_dirty=1, card_invalid=1, card_build_marker='' "
            f"WHERE tenant_id={p} AND actor_id={p}", (headers["tenant_id"], actor),
        )
    return len(affected)


def _manifest(headers, rows):
    manifest = {"version": _VERSION, **headers, "rows": rows}
    manifest["manifest_digest"] = _digest(manifest)
    return manifest


def _replay_rows(conn, headers, operation, dialect):
    if any(operation[name] != headers[name] for name in _HEADERS):
        raise ValueError("operation id already belongs to another audience manifest")
    p = _placeholder(dialect)
    receipts = [_record(row) for row in conn.execute(
        f"SELECT * FROM canonical_audience_reassignments WHERE operation_id = {p} "
        "ORDER BY canonical_turn_id", (headers["operation_id"],),
    ).fetchall()]
    if len(receipts) != operation["row_count"]:
        raise ValueError("operation source rows were removed; replay is not complete")
    planned = [{name: receipt[name] for name in _ROW_FIELDS} for receipt in receipts]
    manifest = _manifest(headers, planned)
    if manifest["manifest_digest"] != operation["manifest_digest"]:
        raise ValueError("audience operation manifest does not match its receipts")
    rows = _load_rows(conn, {row["canonical_turn_id"] for row in planned}, dialect)
    memberships = _memberships(conn, rows, dialect)
    _validate_memberships(rows, memberships, headers)
    source_map = _source_map(rows, memberships)
    for receipt in receipts:
        canonical_id = receipt["canonical_turn_id"]
        row = rows[canonical_id]
        if not source_map[canonical_id]:
            raise ValueError("unattested canonical row cannot be reassigned")
        if (receipt["tenant_id"] != headers["tenant_id"]
                or receipt["owner_conversation_id"] != headers["owner_conversation_id"]
                or receipt["manifest_digest"] != operation["manifest_digest"]
                or row["conversation_id"] != headers["owner_conversation_id"]
                or (row["audience_conversation_id"] or "") != receipt["to_audience"]
                or int(row["audience_attribution_version"] or 0)
                != receipt["to_attribution_version"]
                or row["turn_hash"] != receipt["turn_hash"]
                or canonical_row_fingerprint(row) != receipt["row_fingerprint"]
                or source_membership_fingerprint(source_map[canonical_id])
                != receipt["source_fingerprint"]):
            raise ValueError("audience reassignment replay found changed source evidence")
    # New old-audience writes after a cutover are not silently included in an
    # earlier immutable operation. A fresh operation must inventory them.
    return planned


def plan_audience_reassignment(
    store, owner_conversation_id, from_audience, to_audience, *, tenant_id,
    expected_lifecycle_epoch, operation_id, dialect,
):
    """Return an exact JSON manifest without changing canonical or audit data."""
    headers = dict(
        owner_conversation_id=owner_conversation_id, from_audience=from_audience,
        to_audience=to_audience, tenant_id=tenant_id,
        expected_lifecycle_epoch=expected_lifecycle_epoch, operation_id=operation_id,
    )
    _validate_headers(headers)
    with _transaction(store, dialect) as conn:
        _fence(store, conn, headers, dialect)
        operation = _operation(conn, operation_id, dialect)
        rows = (_replay_rows(conn, headers, operation, dialect) if operation is not None
                else _fresh_rows(conn, headers, dialect))
        return _manifest(headers, rows)


def reassign_audience(store, manifest, *, dry_run=True, dialect):
    """Verify and atomically apply an exact plan, or verify an earlier apply."""
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be a boolean")
    expected_keys = {"version", "manifest_digest", "rows", *_HEADERS}
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise ValueError("invalid audience reassignment manifest shape")
    if type(manifest["version"]) is not int or manifest["version"] != _VERSION:
        raise ValueError("unsupported audience reassignment manifest version")
    headers = {name: manifest[name] for name in _HEADERS}
    _validate_headers(headers)
    if type(manifest["rows"]) is not list:
        raise ValueError("manifest rows must be an array")
    for row in manifest["rows"]:
        if type(row) is not dict or set(row) != set(_ROW_FIELDS):
            raise ValueError("invalid audience reassignment row shape")
    if _manifest(headers, manifest["rows"]) != manifest:
        raise ValueError("audience reassignment manifest digest mismatch")
    p = _placeholder(dialect)
    with _transaction(store, dialect) as conn:
        _fence(store, conn, headers, dialect)
        operation = _operation(conn, headers["operation_id"], dialect)
        rows = (_replay_rows(conn, headers, operation, dialect) if operation is not None
                else _fresh_rows(conn, headers, dialect))
        if _manifest(headers, rows) != manifest:
            raise ValueError("audience reassignment manifest is stale or incomplete")
        changed = sum(
            row["from_audience"] != row["to_audience"]
            or row["from_attribution_version"] != row["to_attribution_version"]
            for row in rows
        )
        report = {
            "operation_id": headers["operation_id"],
            "manifest_digest": manifest["manifest_digest"],
            "selected": len(rows), "updated": 0, "would_update": changed,
            "cards_invalidated": 0, "dry_run": dry_run,
            "already_applied": operation is not None,
        }
        remaining = conn.execute(
            f"SELECT COUNT(*) AS n FROM canonical_turns WHERE conversation_id = {p} "
            f"AND audience_conversation_id = {p}",
            (headers["owner_conversation_id"], headers["from_audience"]),
        ).fetchone()
        report["remaining_source_rows"] = int(remaining["n"])
        if operation is not None:
            report["would_update"] = 0
            return report
        if dry_run:
            return report
        now = utcnow_iso()
        op_fields = (*_HEADERS, "manifest_digest", "row_count", "created_at")
        op_values = {**headers, "manifest_digest": manifest["manifest_digest"],
                     "row_count": len(rows), "created_at": now}
        conn.execute(
            f"INSERT INTO audience_reassignment_operations ({','.join(op_fields)}) "
            f"VALUES ({','.join(p for _ in op_fields)})",
            [op_values[name] for name in op_fields],
        )
        receipt_fields = (
            *_ROW_FIELDS, "operation_id", "tenant_id", "owner_conversation_id",
            "manifest_digest", "created_at",
        )
        for row in rows:
            receipt = {**row, **{name: headers[name] for name in (
                "operation_id", "tenant_id", "owner_conversation_id",
            )}, "manifest_digest": manifest["manifest_digest"], "created_at": now}
            conn.execute(
                f"INSERT INTO canonical_audience_reassignments ({','.join(receipt_fields)}) "
                f"VALUES ({','.join(p for _ in receipt_fields)})",
                [receipt[name] for name in receipt_fields],
            )
        for row in rows:
            if (row["from_audience"] == row["to_audience"]
                    and row["from_attribution_version"] == row["to_attribution_version"]):
                continue
            result = conn.execute(
                f"UPDATE canonical_turns SET audience_conversation_id = {p}, "
                f"audience_attribution_version = {p}, updated_at = {p} "
                f"WHERE canonical_turn_id = {p} AND conversation_id = {p} "
                f"AND audience_conversation_id = {p} AND audience_attribution_version = {p} "
                f"AND turn_hash = {p}",
                (row["to_audience"], row["to_attribution_version"], now,
                 row["canonical_turn_id"], headers["owner_conversation_id"],
                 row["from_audience"], row["from_attribution_version"], row["turn_hash"]),
            )
            if result.rowcount != 1:
                raise RuntimeError("audience reassignment lost its canonical compare-and-set")
            report["updated"] += 1
        _replay_rows(conn, headers, _operation(conn, headers["operation_id"], dialect), dialect)
        if changed:
            report["cards_invalidated"] = _invalidate_cards(conn, headers, dialect)
        report["remaining_source_rows"] = int(conn.execute(
            f"SELECT COUNT(*) AS n FROM canonical_turns WHERE conversation_id = {p} "
            f"AND audience_conversation_id = {p}",
            (headers["owner_conversation_id"], headers["from_audience"]),
        ).fetchone()["n"])
        if report["remaining_source_rows"]:
            raise RuntimeError("source audience changed while applying its manifest")
        return report
