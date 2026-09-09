"""Audited, append-only source-channel enrichment for exact assistant pairs.

Only a missing assistant channel can change. Immutable source memberships and
earlier audience receipts remain untouched; replay composes their original
fingerprints with this separately authorized transition.
"""
from __future__ import annotations

import hashlib

from ..core.canonical_turns import compute_turn_hash_from_raw, utcnow_iso
from ..types import AUDIENCE_ATTRIBUTION_VERSION
from .audience_reassignment import (
    _digest, _fence, _load_rows, _memberships, _placeholder, _record,
    _transaction, assert_audience_reassignment_schema, canonical_row_fingerprint,
    get_audience_reassignment, source_fingerprint,
)


_VERSION = 1
_HEADERS = (
    "operation_id", "tenant_id", "owner_conversation_id",
    "audience_conversation_id", "expected_lifecycle_epoch",
)
_ROW_FIELDS = (
    "assistant_canonical_turn_id", "user_canonical_turn_id", "user_turn_hash",
    "assistant_turn_hash", "user_row_fingerprint", "source_fingerprint",
    "from_channel_id", "to_channel_id", "before_row_fingerprint",
    "after_row_fingerprint", "prior_audience_operation_id",
    "prior_audience_manifest_digest", "prior_audience_row_fingerprint",
)
_OP_TABLE = "assistant_channel_enrichment_operations"
_ROW_TABLE = "canonical_assistant_channel_enrichments"
_SQLITE_GUARDS = {
    "trg_assistant_channel_enrichment_update": f"""CREATE TRIGGER
        trg_assistant_channel_enrichment_update BEFORE UPDATE ON {_ROW_TABLE}
        BEGIN SELECT RAISE(ABORT, 'assistant channel enrichment is immutable'); END""",
    "trg_assistant_channel_enrichment_delete": f"""CREATE TRIGGER
        trg_assistant_channel_enrichment_delete BEFORE DELETE ON {_ROW_TABLE}
        WHEN EXISTS (SELECT 1 FROM canonical_turns
                     WHERE canonical_turn_id=OLD.assistant_canonical_turn_id)
        BEGIN SELECT RAISE(ABORT, 'live assistant channel enrichment is immutable'); END""",
    "trg_assistant_channel_operation_update": f"""CREATE TRIGGER
        trg_assistant_channel_operation_update BEFORE UPDATE ON {_OP_TABLE}
        BEGIN SELECT RAISE(ABORT, 'assistant channel operation is immutable'); END""",
    "trg_assistant_channel_operation_delete": f"""CREATE TRIGGER
        trg_assistant_channel_operation_delete BEFORE DELETE ON {_OP_TABLE}
        WHEN EXISTS (SELECT 1 FROM {_ROW_TABLE} WHERE operation_id=OLD.operation_id)
        BEGIN SELECT RAISE(ABORT, 'live assistant channel operation is immutable'); END""",
}
_PG_FUNCTIONS = {
    "guard_assistant_channel_enrichment": """BEGIN
        IF TG_OP = 'UPDATE' OR EXISTS (SELECT 1 FROM canonical_turns
            WHERE canonical_turn_id=OLD.assistant_canonical_turn_id) THEN
            RAISE EXCEPTION 'assistant channel enrichment is immutable';
        END IF;
        RETURN OLD;
    END;""",
    "guard_assistant_channel_operation": f"""BEGIN
        IF TG_OP = 'UPDATE' OR EXISTS (SELECT 1 FROM {_ROW_TABLE}
            WHERE operation_id=OLD.operation_id) THEN
            RAISE EXCEPTION 'assistant channel operation is immutable';
        END IF;
        RETURN OLD;
    END;""",
}


def _sql(text):
    return " ".join(str(text).split()).rstrip(";").lower()


def ensure_assistant_channel_enrichment_schema(conn, dialect):
    """Install the additive audit schema inside the caller's transaction."""
    _placeholder(dialect)
    canonical_type = "UUID" if dialect == "postgres" else "TEXT"
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {_OP_TABLE} (
        operation_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
        owner_conversation_id TEXT NOT NULL, audience_conversation_id TEXT NOT NULL,
        expected_lifecycle_epoch INTEGER NOT NULL, version INTEGER NOT NULL,
        manifest_digest TEXT NOT NULL, row_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {_ROW_TABLE} (
        assistant_canonical_turn_id {canonical_type} PRIMARY KEY
            REFERENCES canonical_turns(canonical_turn_id) ON DELETE CASCADE,
        user_canonical_turn_id {canonical_type} NOT NULL
            REFERENCES canonical_turns(canonical_turn_id) DEFERRABLE INITIALLY DEFERRED,
        operation_id TEXT NOT NULL REFERENCES {_OP_TABLE}(operation_id),
        tenant_id TEXT NOT NULL, owner_conversation_id TEXT NOT NULL,
        audience_conversation_id TEXT NOT NULL, audience_attribution_version INTEGER NOT NULL,
        user_turn_hash TEXT NOT NULL, assistant_turn_hash TEXT NOT NULL,
        user_row_fingerprint TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
        from_channel_id TEXT NOT NULL CHECK (from_channel_id=''),
        to_channel_id TEXT NOT NULL CHECK (to_channel_id<>''),
        before_row_fingerprint TEXT NOT NULL, after_row_fingerprint TEXT NOT NULL,
        prior_audience_operation_id TEXT NOT NULL,
        prior_audience_manifest_digest TEXT NOT NULL,
        prior_audience_row_fingerprint TEXT NOT NULL,
        manifest_digest TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_assistant_channel_enrichment_operation "
                 f"ON {_ROW_TABLE}(operation_id)")
    if dialect == "sqlite":
        for statement in _SQLITE_GUARDS.values():
            conn.execute(statement.replace("CREATE TRIGGER", "CREATE TRIGGER IF NOT EXISTS", 1))
    else:
        for name, body in _PG_FUNCTIONS.items():
            conn.execute(f"CREATE OR REPLACE FUNCTION {name}() RETURNS trigger "
                         f"AS $$ {body} $$ LANGUAGE plpgsql")
        for table, function in (
            (_ROW_TABLE, "guard_assistant_channel_enrichment"),
            (_OP_TABLE, "guard_assistant_channel_operation"),
        ):
            name = "trg_" + function
            conn.execute(f"DROP TRIGGER IF EXISTS {name} ON {table}")
            conn.execute(f"CREATE TRIGGER {name} BEFORE UPDATE OR DELETE ON {table} "
                         f"FOR EACH ROW EXECUTE FUNCTION {function}()")


def assert_assistant_channel_enrichment_schema(conn, dialect):
    """Refuse incomplete schema or stale guards without installing anything."""
    from .actor_card_transition_guards import (
        postgres_actor_card_turn_source_function_body,
        sqlite_actor_card_turn_source_update_sql,
    )
    from .channel_enrichment_guards import (
        receipted_channel_guard_function_body,
        receipted_channel_guard_predicate,
    )
    assert_audience_reassignment_schema(conn, dialect)
    required = {
        _OP_TABLE: {*_HEADERS, "version", "manifest_digest", "row_count", "created_at"},
        _ROW_TABLE: {*_ROW_FIELDS, "operation_id", "tenant_id", "owner_conversation_id",
                     "audience_conversation_id", "audience_attribution_version",
                     "manifest_digest", "created_at"},
    }
    for table, columns in required.items():
        expected_pk = "operation_id" if table == _OP_TABLE else "assistant_canonical_turn_id"
        if dialect == "sqlite":
            records = conn.execute(f"PRAGMA table_info({table})").fetchall()
            names = {row["name"] for row in records}
            primary = {row["name"] for row in records if row["pk"]}
        else:
            records = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name=%s", (table,),
            ).fetchall()
            names = {row["column_name"] for row in records}
            keys = conn.execute(
                "SELECT a.attname FROM pg_constraint c JOIN pg_attribute a "
                "ON a.attrelid=c.conrelid AND a.attnum=ANY(c.conkey) "
                "WHERE c.conrelid=%s::regclass AND c.contype='p'", (table,),
            ).fetchall() if names else []
            primary = {row["attname"] for row in keys}
        if not columns.issubset(names) or primary != {expected_pk}:
            raise RuntimeError("assistant channel enrichment schema is missing or unsafe")
    if dialect == "sqlite":
        fks = conn.execute(f"PRAGMA foreign_key_list({_ROW_TABLE})").fetchall()
        actual = {(row["from"], row["table"], row["to"], row["on_delete"].upper()) for row in fks}
        expected = {
            ("assistant_canonical_turn_id", "canonical_turns", "canonical_turn_id", "CASCADE"),
            ("user_canonical_turn_id", "canonical_turns", "canonical_turn_id", "NO ACTION"),
            ("operation_id", _OP_TABLE, "operation_id", "NO ACTION"),
        }
        if not expected.issubset(actual):
            raise RuntimeError("assistant channel enrichment foreign keys are unsafe")
        guards = {row["name"]: row["sql"] for row in conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name IN (?,?)", (_ROW_TABLE, _OP_TABLE),
        ).fetchall()}
        if any(_sql(guards.get(name, "")) != _sql(sql) for name, sql in _SQLITE_GUARDS.items()):
            raise RuntimeError("assistant channel enrichment immutability guards are stale")
        card = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='trg_invalidate_actor_card_turn_source_update'",
        ).fetchone()
        definition = str(card["sql"] or "") if card else ""
        if _sql(definition) != _sql(sqlite_actor_card_turn_source_update_sql()):
            raise RuntimeError("assistant channel enrichment card-preservation guard is missing or stale")
    else:
        foreign = conn.execute(
            "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid=%s::regclass AND contype='f'", (_ROW_TABLE,),
        ).fetchall()
        definitions = " ".join(row["definition"] for row in foreign)
        for fragment in (
            "FOREIGN KEY (assistant_canonical_turn_id) REFERENCES canonical_turns(canonical_turn_id) ON DELETE CASCADE",
            "FOREIGN KEY (user_canonical_turn_id) REFERENCES canonical_turns(canonical_turn_id) DEFERRABLE INITIALLY DEFERRED",
            f"FOREIGN KEY (operation_id) REFERENCES {_OP_TABLE}(operation_id)",
        ):
            if fragment not in definitions:
                raise RuntimeError("assistant channel enrichment foreign keys are unsafe")
        functions = {row["proname"]: row["prosrc"] for row in conn.execute(
            "SELECT p.proname,p.prosrc FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=current_schema() AND p.proname IN (%s,%s)", tuple(_PG_FUNCTIONS),
        ).fetchall()}
        if any(_sql(functions.get(name, "")) != _sql(body) for name, body in _PG_FUNCTIONS.items()):
            raise RuntimeError("assistant channel enrichment immutability guards are stale")
        triggers = conn.execute(
            "SELECT t.tgname,t.tgtype,t.tgenabled,c.relname,p.proname, "
            "p.pronamespace=c.relnamespace AS local_function, "
            "t.tgqual IS NULL AS unconditional,t.tgnargs,t.tgattr::text AS columns "
            "FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_proc p ON p.oid=t.tgfoid WHERE t.tgrelid IN "
            "(%s::regclass,%s::regclass) AND NOT t.tgisinternal", (_ROW_TABLE, _OP_TABLE),
        ).fetchall()
        active = {(row["tgname"], row["relname"], row["proname"]) for row in triggers
                  if row["tgtype"] == 27 and row["tgenabled"] in {"O", "A"}
                  and row["local_function"] and row["unconditional"]
                  and row["tgnargs"] == 0 and not row["columns"]}
        expected = {("trg_guard_assistant_channel_enrichment", _ROW_TABLE, "guard_assistant_channel_enrichment"),
                    ("trg_guard_assistant_channel_operation", _OP_TABLE, "guard_assistant_channel_operation")}
        if not expected.issubset(active):
            raise RuntimeError("assistant channel enrichment immutability guards are missing")
        card = conn.execute(
            "SELECT p.prosrc,p.proname,t.tgtype,t.tgenabled, "
            "p.pronamespace=c.relnamespace AS local_function, "
            "t.tgqual IS NULL AS unconditional,t.tgnargs, "
            "ARRAY(SELECT a.attname FROM pg_attribute a WHERE a.attrelid=t.tgrelid "
            "AND a.attnum=ANY(t.tgattr::smallint[]) ORDER BY a.attname) AS update_columns "
            "FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid "
            "JOIN pg_class c ON c.oid=t.tgrelid "
            "WHERE t.tgrelid='canonical_turns'::regclass "
            "AND t.tgname='trg_invalidate_actor_card_turn_source' AND NOT t.tgisinternal",
        ).fetchone()
        expected_columns = sorted([
            "conversation_id", "user_content", "sender_actor_id", "audience_conversation_id",
            "audience_attribution_version", "origin_channel_id", "created_at", "first_seen_at",
        ])
        if (card is None or card["tgtype"] != 27 or card["tgenabled"] not in {"O", "A"}
                or card["proname"] != "vc_invalidate_actor_card_turn_source"
                or not card["local_function"] or not card["unconditional"] or card["tgnargs"] != 0
                or list(card["update_columns"]) != expected_columns
                or _sql(card["prosrc"]) != _sql(postgres_actor_card_turn_source_function_body())):
            raise RuntimeError("assistant channel enrichment card-preservation guard is missing or stale")
    guard_name = "trg_guard_receipted_assistant_channel_update"
    if dialect == "sqlite":
        guard = conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (guard_name,)).fetchone()
        expected = f"""CREATE TRIGGER {guard_name}
            BEFORE UPDATE OF origin_channel_id, origin_channel_label ON canonical_turns
            FOR EACH ROW WHEN {receipted_channel_guard_predicate(dialect)}
            BEGIN SELECT RAISE(ABORT, 'receipted channel enrichment requires authorization'); END"""
        if guard is None or _sql(guard["sql"]) != _sql(expected):
            raise RuntimeError("assistant channel enrichment canonical guard is missing or stale")
    else:
        guard = conn.execute(
            "SELECT p.prosrc,p.proname,t.tgtype,t.tgenabled, "
            "p.pronamespace=c.relnamespace AS local_function, "
            "t.tgqual IS NULL AS unconditional,t.tgnargs, "
            "ARRAY(SELECT a.attname FROM pg_attribute a WHERE a.attrelid=t.tgrelid "
            "AND a.attnum=ANY(t.tgattr::smallint[]) ORDER BY a.attname) AS update_columns "
            "FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid "
            "JOIN pg_class c ON c.oid=t.tgrelid "
            "WHERE t.tgrelid='canonical_turns'::regclass AND t.tgname=%s AND NOT t.tgisinternal",
            (guard_name,),
        ).fetchone()
        if (guard is None or guard["tgtype"] != 19 or guard["tgenabled"] not in {"O", "A"}
                or guard["proname"] != "vc_guard_receipted_assistant_channel_update"
                or not guard["local_function"] or not guard["unconditional"] or guard["tgnargs"] != 0
                or list(guard["update_columns"]) != ["origin_channel_id", "origin_channel_label"]
                or _sql(guard["prosrc"]) != _sql(receipted_channel_guard_function_body(dialect))):
            raise RuntimeError("assistant channel enrichment canonical guard is missing or stale")


def _headers(values):
    for key in _HEADERS[:-1]:
        value = values.get(key)
        if (type(value) is not str or not value or value != value.strip() or len(value) > 1024
                or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError(f"{key} must be a nonempty exact identifier")
    epoch = values.get("expected_lifecycle_epoch")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("expected_lifecycle_epoch must be a positive integer")


def _ids(values):
    if type(values) is not list or not values or len(values) > 10000:
        raise ValueError("assistant selection must be a nonempty bounded array")
    if any(type(value) is not str or not value or value != value.strip() or len(value) > 1024 for value in values):
        raise ValueError("assistant selection contains an invalid identifier")
    if len(set(values)) != len(values):
        raise ValueError("assistant selection contains duplicate identifiers")
    return sorted(values)


def _manifest(headers, rows):
    value = {"version": _VERSION, **headers, "rows": rows}
    return {**value, "manifest_digest": _digest(value)}


def _operation(conn, operation_id, dialect):
    p = _placeholder(dialect)
    row = conn.execute(f"SELECT * FROM {_OP_TABLE} WHERE operation_id={p}", (operation_id,)).fetchone()
    return _record(row) if row is not None else None


def _receipt(conn, assistant_id, dialect):
    p = _placeholder(dialect)
    row = conn.execute(
        f"SELECT * FROM {_ROW_TABLE} WHERE assistant_canonical_turn_id={p}", (assistant_id,),
    ).fetchone()
    return _record(row) if row is not None else None


def _fenced(store, conn, headers, dialect):
    assert_assistant_channel_enrichment_schema(conn, dialect)
    # Existing administrative fencing proves the terminal owner, current
    # audience/alias, tenant, lifecycle and absence of active writers. Unlike
    # reassignment, enrichment does not authorize a change between audiences.
    _fence(store, conn, {
        **headers, "from_audience": headers["owner_conversation_id"],
        "to_audience": headers["audience_conversation_id"],
    }, dialect)


def _pair(conn, assistant_id, headers, dialect, *, allow_audience_successor=False):
    from .audience_proof import effective_attested_audience

    sources = _memberships(conn, {assistant_id}, dialect)
    if len(sources) != 1 or sources[0]["assistant_canonical_turn_id"] != assistant_id:
        raise ValueError("assistant channel enrichment requires one exact source pair")
    source = sources[0]
    user_id = source["canonical_turn_id"]
    rows = _load_rows(conn, {assistant_id, user_id}, dialect)
    if len(rows) != 2:
        raise ValueError("assistant channel enrichment requires distinct physical rows")
    user, assistant = rows[user_id], rows[assistant_id]
    if len(_memberships(conn, rows, dialect)) != 1:
        raise ValueError("assistant channel enrichment found overlapping source pairs")
    if source["tenant_id"] != headers["tenant_id"] or int(source["pair_version"]) != 1:
        raise ValueError("assistant channel enrichment has invalid source authority")
    if (not (source["channel_id"] or "").strip()
            or (source["channel_id"] or "") != source["channel_id"].strip()
            or (user["origin_channel_id"] or "") != source["channel_id"]
            or (user["sender_actor_id"] or "") != source["source_actor_id"]
            or (user["source_message_id"] or "") != source["message_id"]
            or (user["reply_target_message_id"] or "") != source["reply_target_message_id"]
            or user["turn_hash"] != source["canonical_turn_hash"]
            or assistant["turn_hash"] != source["assistant_turn_hash"]
            or int(user["turn_group_number"]) < 0
            or int(user["turn_group_number"]) != int(assistant["turn_group_number"])):
        raise ValueError("assistant channel enrichment source projection disagrees")
    if (not (user["user_content"] or "").strip()
            or (user["assistant_content"] or "") != ""
            or (assistant["user_content"] or "") != ""
            or not (assistant["assistant_content"] or "").strip()
            or hashlib.sha256((user["user_content"] or "").encode()).hexdigest()
            != source["canonical_body_sha256"]):
        raise ValueError("assistant channel enrichment source body or role disagrees")
    for key in ("sender", "sender_actor_id", "source_message_id", "reply_target_message_id",
                "reply_subject_actor_id", "reply_subject_label", "reply_target_body"):
        if (assistant[key] or "") != "":
            raise ValueError("assistant channel enrichment refuses human or reply identity")
    if int(assistant["reply_attribution_version"] or 0) != 0:
        raise ValueError("assistant channel enrichment refuses reply attribution")
    if any(int(row["audience_attribution_version"] or 0) != AUDIENCE_ATTRIBUTION_VERSION
           or not (row["audience_conversation_id"] or "").strip() for row in (user, assistant)):
        raise ValueError("assistant channel enrichment refuses unproved audience attribution; a separate attribution repair is required")
    effective = effective_attested_audience(conn, source, dialect=dialect)
    if effective != headers["audience_conversation_id"]:
        successor = get_audience_reassignment(conn, assistant_id, dialect)
        if (not allow_audience_successor or successor is None
                or successor["from_audience"] != headers["audience_conversation_id"]
                or successor["to_audience"] != effective):
            raise ValueError("assistant channel enrichment owner or audience disagrees")
    for row in (user, assistant):
        if (row["conversation_id"] != headers["owner_conversation_id"]
                or (row["audience_conversation_id"] or "") != effective):
            raise ValueError("assistant channel enrichment owner or audience disagrees")
        actual = compute_turn_hash_from_raw(
            row["user_content"], row["assistant_content"], version=row["hash_version"],
        )
        if actual != (row["turn_hash"], row["normalized_user_text"], row["normalized_assistant_text"]):
            raise ValueError("assistant channel enrichment canonical hash disagrees")
    return user, assistant, source


def _prior(conn, user, assistant, source, dialect, *, before_fingerprint):
    receipt = get_audience_reassignment(conn, assistant["canonical_turn_id"], dialect)
    user_receipt = get_audience_reassignment(conn, user["canonical_turn_id"], dialect)
    if receipt is None and user_receipt is None:
        return {name: "" for name in _ROW_FIELDS if name.startswith("prior_audience_")}
    if receipt is None or user_receipt is None:
        raise ValueError("assistant channel enrichment requires both audience receipts")
    # effective_attested_audience already verifies pair/tenant/source/target
    # linkage; pin the original physical projections as an additional gate.
    if (receipt["row_fingerprint"] != before_fingerprint
            or user_receipt["row_fingerprint"] != canonical_row_fingerprint(user)
            or receipt["source_fingerprint"] != source_fingerprint(source)):
        raise ValueError("assistant channel enrichment found changed audience receipt evidence")
    return {
        "prior_audience_operation_id": receipt["operation_id"],
        "prior_audience_manifest_digest": receipt["manifest_digest"],
        "prior_audience_row_fingerprint": receipt["row_fingerprint"],
    }


def _planned_row(conn, assistant_id, headers, dialect):
    if _receipt(conn, assistant_id, dialect) is not None:
        raise ValueError("assistant row already has channel enrichment authorization")
    user, assistant, source = _pair(conn, assistant_id, headers, dialect)
    if (assistant["origin_channel_id"] or "") != "":
        raise ValueError("assistant channel enrichment requires an empty channel")
    before = canonical_row_fingerprint(assistant)
    after = canonical_row_fingerprint({**assistant, "origin_channel_id": source["channel_id"]})
    return {
        "assistant_canonical_turn_id": assistant_id,
        "user_canonical_turn_id": user["canonical_turn_id"],
        "user_turn_hash": user["turn_hash"], "assistant_turn_hash": assistant["turn_hash"],
        "user_row_fingerprint": canonical_row_fingerprint(user),
        "source_fingerprint": source_fingerprint(source),
        "from_channel_id": "", "to_channel_id": source["channel_id"],
        "before_row_fingerprint": before, "after_row_fingerprint": after,
        **_prior(conn, user, assistant, source, dialect, before_fingerprint=before),
    }


def _verify_prior_operations(conn, rows, dialect):
    from . import audience_reassignment

    # Audience correction may precede the enrichment or follow it. Verify
    # the entire existing audience operation in either direction, without
    # recursively invoking its administrative fence or mutating receipts.
    operations = {row["prior_audience_operation_id"] for row in rows if row["prior_audience_operation_id"]}
    for row in rows:
        current = get_audience_reassignment(conn, row["assistant_canonical_turn_id"], dialect)
        if current is not None:
            operations.add(current["operation_id"])
    for operation_id in sorted(operations):
        operation = audience_reassignment._operation(conn, operation_id, dialect)
        if operation is None:
            raise ValueError("prior audience operation is missing")
        headers = {name: operation[name] for name in audience_reassignment._HEADERS}
        audience_reassignment._replay_rows(conn, headers, operation, dialect)


def _fence_successor_audiences(store, conn, headers, rows, dialect):
    """Fence a later effective audience without rewriting original headers."""
    targets = set()
    for row in rows:
        if row["prior_audience_operation_id"]:
            continue
        current = get_audience_reassignment(conn, row["assistant_canonical_turn_id"], dialect)
        if current is not None and current["to_audience"] != headers["audience_conversation_id"]:
            targets.add(current["to_audience"])
    for target in sorted(targets):
        _fence(store, conn, {**headers, "from_audience": headers["owner_conversation_id"],
                            "to_audience": target}, dialect)


def _operation_receipts(conn, operation, dialect):
    p = _placeholder(dialect)
    receipts = [_record(row) for row in conn.execute(
        f"SELECT * FROM {_ROW_TABLE} WHERE operation_id={p} ORDER BY assistant_canonical_turn_id",
        (operation["operation_id"],),
    ).fetchall()]
    headers = {name: operation[name] for name in _HEADERS}
    rows = [{name: receipt[name] for name in _ROW_FIELDS} for receipt in receipts]
    if (int(operation["version"]) != _VERSION or not receipts
            or len(receipts) != int(operation["row_count"])
            or _manifest(headers, rows)["manifest_digest"] != operation["manifest_digest"]):
        raise ValueError("assistant channel enrichment operation is incomplete or changed")
    for receipt in receipts:
        if (any(receipt[name] != headers[name] for name in _HEADERS if name != "expected_lifecycle_epoch")
                or receipt["manifest_digest"] != operation["manifest_digest"]
                or int(receipt["audience_attribution_version"]) != AUDIENCE_ATTRIBUTION_VERSION):
            raise ValueError("assistant channel enrichment receipt disagrees with its operation")
    return headers, receipts


def _verify_receipt(conn, receipt, headers, dialect):
    had_prior_audience = any(receipt[name] for name in _ROW_FIELDS if name.startswith("prior_audience_"))
    user, assistant, source = _pair(
        conn, receipt["assistant_canonical_turn_id"], headers, dialect,
        allow_audience_successor=not had_prior_audience,
    )
    if (receipt["user_canonical_turn_id"] != user["canonical_turn_id"]
            or receipt["user_turn_hash"] != user["turn_hash"]
            or receipt["assistant_turn_hash"] != assistant["turn_hash"]
            or receipt["user_row_fingerprint"] != canonical_row_fingerprint(user)
            or receipt["source_fingerprint"] != source_fingerprint(source)
            or receipt["from_channel_id"] != ""
            or receipt["to_channel_id"] != source["channel_id"]
            or receipt["to_channel_id"] != assistant["origin_channel_id"]
            or receipt["after_row_fingerprint"] != canonical_row_fingerprint(assistant)):
        raise ValueError("assistant channel enrichment replay found changed source evidence")
    restored = {**assistant, "origin_channel_id": receipt["from_channel_id"]}
    if canonical_row_fingerprint(restored) != receipt["before_row_fingerprint"]:
        raise ValueError("assistant channel enrichment does not preserve the original projection")
    if had_prior_audience:
        prior = _prior(conn, user, assistant, source, dialect,
                       before_fingerprint=receipt["before_row_fingerprint"])
        if any(receipt[name] != value for name, value in prior.items()):
            raise ValueError("assistant channel enrichment prior audience receipt disagrees")
    else:
        # An audience correction after enrichment pins the channel's AFTER
        # projection. Its immutable source/from-audience proof is checked by
        # _pair; it cannot replace the earlier enrichment's original headers.
        user_audience = get_audience_reassignment(conn, user["canonical_turn_id"], dialect)
        assistant_audience = get_audience_reassignment(conn, assistant["canonical_turn_id"], dialect)
        if user_audience is not None or assistant_audience is not None:
            if (user_audience is None or assistant_audience is None
                    or any(row["from_audience"] != headers["audience_conversation_id"]
                           for row in (user_audience, assistant_audience))
                    or any(int(row["from_attribution_version"]) != AUDIENCE_ATTRIBUTION_VERSION
                           for row in (user_audience, assistant_audience))
                    or user_audience["row_fingerprint"] != receipt["user_row_fingerprint"]
                    or assistant_audience["row_fingerprint"] != receipt["after_row_fingerprint"]):
                raise ValueError("assistant channel enrichment successor audience receipt disagrees")


def _verify_existing_receipt(conn, receipt, dialect, operation_cache):
    operation_id = receipt["operation_id"]
    if operation_id not in operation_cache:
        operation = _operation(conn, operation_id, dialect)
        if operation is None:
            raise ValueError("assistant channel enrichment operation is missing")
        headers, receipts = _operation_receipts(conn, operation, dialect)
        operation_cache[operation_id] = (
            headers, {row["assistant_canonical_turn_id"]: row for row in receipts},
        )
    headers, by_id = operation_cache[operation_id]
    if by_id.get(receipt["assistant_canonical_turn_id"]) != receipt:
        raise ValueError("assistant channel enrichment receipt is outside its operation")
    _verify_receipt(conn, receipt, headers, dialect)
    return receipt


def verify_assistant_channel_enrichment(conn, assistant_canonical_turn_id, *, dialect):
    """Verify a stored authorization on an existing caller-owned transaction."""
    receipt = _receipt(conn, assistant_canonical_turn_id, dialect)
    if receipt is None:
        raise ValueError("assistant channel enrichment authorization is missing")
    return _verify_existing_receipt(conn, receipt, dialect, {})


def canonical_row_matches_audience_receipt(
    conn, row, audience_receipt, *, dialect, verification_cache=None,
):
    """Compose an unchanged audience proof with one audited channel-only fill."""
    current = canonical_row_fingerprint(row)
    cache = verification_cache if verification_cache is not None else {}
    # No table on an older read-only database means no authorized enrichment.
    if "schema_exists" not in cache:
        if dialect == "sqlite":
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_ROW_TABLE,)).fetchone()
        else:
            exists = conn.execute("SELECT to_regclass(%s) AS name", (_ROW_TABLE,)).fetchone()
            exists = exists and exists["name"]
        cache["schema_exists"] = bool(exists)
    if not cache["schema_exists"]:
        return current == audience_receipt["row_fingerprint"]
    receipt = _receipt(conn, str(row["canonical_turn_id"]), dialect)
    if receipt is None:
        return current == audience_receipt["row_fingerprint"]
    try:
        _verify_existing_receipt(conn, receipt, dialect, cache.setdefault("operations", {}))
    except (ValueError, KeyError, TypeError):
        return False
    if receipt["after_row_fingerprint"] != current:
        return False
    if receipt["prior_audience_operation_id"]:
        return (
            receipt["prior_audience_operation_id"] == audience_receipt["operation_id"]
            and receipt["prior_audience_manifest_digest"] == audience_receipt["manifest_digest"]
            and receipt["prior_audience_row_fingerprint"] == audience_receipt["row_fingerprint"]
        )
    return (
        not receipt["prior_audience_manifest_digest"]
        and not receipt["prior_audience_row_fingerprint"]
        and audience_receipt == get_audience_reassignment(conn, str(row["canonical_turn_id"]), dialect)
        and audience_receipt["from_audience"] == receipt["audience_conversation_id"]
        and audience_receipt["row_fingerprint"] == receipt["after_row_fingerprint"]
    )


def plan_assistant_channel_enrichment(
    store, owner_conversation_id, *, tenant_id, audience_conversation_id,
    expected_lifecycle_epoch, operation_id, assistant_canonical_turn_ids, dialect,
):
    """Plan an explicit set without modifying canonical rows or audit state."""
    headers = dict(
        owner_conversation_id=owner_conversation_id, tenant_id=tenant_id,
        audience_conversation_id=audience_conversation_id,
        expected_lifecycle_epoch=expected_lifecycle_epoch, operation_id=operation_id,
    )
    _headers(headers)
    ids = _ids(assistant_canonical_turn_ids)
    with _transaction(store, dialect) as conn:
        _fenced(store, conn, headers, dialect)
        operation = _operation(conn, operation_id, dialect)
        if operation is not None:
            recorded, receipts = _operation_receipts(conn, operation, dialect)
            if recorded != headers or [row["assistant_canonical_turn_id"] for row in receipts] != ids:
                raise ValueError("operation id already belongs to another channel manifest")
            for receipt in receipts:
                _verify_receipt(conn, receipt, headers, dialect)
            rows = [{name: row[name] for name in _ROW_FIELDS} for row in receipts]
        else:
            rows = [_planned_row(conn, assistant_id, headers, dialect) for assistant_id in ids]
        _fence_successor_audiences(store, conn, headers, rows, dialect)
        _verify_prior_operations(conn, rows, dialect)
        return _manifest(headers, rows)


def enrich_assistant_channels(store, manifest, *, dry_run=True, dialect):
    """Verify or atomically apply exactly one private channel-enrichment plan."""
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be a boolean")
    if type(manifest) is not dict or set(manifest) != {"version", "rows", "manifest_digest", *_HEADERS}:
        raise ValueError("invalid assistant channel enrichment manifest shape")
    if type(manifest["version"]) is not int or manifest["version"] != _VERSION:
        raise ValueError("unsupported assistant channel enrichment version")
    headers = {name: manifest[name] for name in _HEADERS}
    _headers(headers)
    if type(manifest["rows"]) is not list:
        raise ValueError("assistant channel enrichment rows must be an array")
    for row in manifest["rows"]:
        if type(row) is not dict or set(row) != set(_ROW_FIELDS) or any(type(value) is not str for value in row.values()):
            raise ValueError("invalid assistant channel enrichment row shape")
    ids = _ids([row["assistant_canonical_turn_id"] for row in manifest["rows"]])
    if ids != [row["assistant_canonical_turn_id"] for row in manifest["rows"]] or _manifest(headers, manifest["rows"]) != manifest:
        raise ValueError("assistant channel enrichment manifest digest or order disagrees")
    p = _placeholder(dialect)
    with _transaction(store, dialect) as conn:
        _fenced(store, conn, headers, dialect)
        operation = _operation(conn, headers["operation_id"], dialect)
        if operation is not None:
            recorded, receipts = _operation_receipts(conn, operation, dialect)
            if recorded != headers:
                raise ValueError("operation id already belongs to another channel manifest")
            rows = [{name: row[name] for name in _ROW_FIELDS} for row in receipts]
            for receipt in receipts:
                _verify_receipt(conn, receipt, headers, dialect)
        else:
            rows = [_planned_row(conn, assistant_id, headers, dialect) for assistant_id in ids]
        if _manifest(headers, rows) != manifest:
            raise ValueError("assistant channel enrichment manifest is stale or incomplete")
        _fence_successor_audiences(store, conn, headers, rows, dialect)
        _verify_prior_operations(conn, rows, dialect)
        report = {
            "operation_id": headers["operation_id"], "manifest_digest": manifest["manifest_digest"],
            "selected": len(rows), "updated": 0, "would_update": 0 if operation else len(rows),
            "dry_run": dry_run, "already_applied": operation is not None,
        }
        if operation is not None or dry_run:
            return report
        now = utcnow_iso()
        op = {**headers, "version": _VERSION, "manifest_digest": manifest["manifest_digest"],
              "row_count": len(rows), "created_at": now}
        conn.execute(f"INSERT INTO {_OP_TABLE} ({','.join(op)}) VALUES ({','.join(p for _ in op)})", tuple(op.values()))
        for row in rows:
            receipt = {**row, **{key: headers[key] for key in _HEADERS if key != "expected_lifecycle_epoch"},
                       "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
                       "manifest_digest": manifest["manifest_digest"], "created_at": now}
            conn.execute(f"INSERT INTO {_ROW_TABLE} ({','.join(receipt)}) VALUES ({','.join(p for _ in receipt)})", tuple(receipt.values()))
        for row in rows:
            updated = conn.execute(
                f"UPDATE canonical_turns SET origin_channel_id={p} "
                f"WHERE canonical_turn_id={p} AND conversation_id={p} AND origin_channel_id={p} "
                f"AND turn_hash={p} AND audience_conversation_id={p} AND audience_attribution_version={p} "
                f"AND EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id=canonical_turns.conversation_id "
                f"AND c.lifecycle_epoch={p} AND c.phase='active' AND c.deleted_at IS NULL)",
                (row["to_channel_id"], row["assistant_canonical_turn_id"], headers["owner_conversation_id"],
                 row["from_channel_id"], row["assistant_turn_hash"], headers["audience_conversation_id"],
                 AUDIENCE_ATTRIBUTION_VERSION, headers["expected_lifecycle_epoch"]),
            )
            if updated.rowcount != 1:
                raise RuntimeError("assistant channel enrichment lost its canonical compare-and-set")
            report["updated"] += 1
        _, receipts = _operation_receipts(conn, _operation(conn, headers["operation_id"], dialect), dialect)
        for receipt in receipts:
            _verify_receipt(conn, receipt, headers, dialect)
        _verify_prior_operations(conn, rows, dialect)
        return report
