"""Append-only occurrence times bound to immutable, exact message sources.

The source ledger is unchanged. Occurrence is supplied by a trusted adapter,
never inferred from ingestion dates, ordinals, or platform-specific identifiers.
"""
from __future__ import annotations

import hashlib

from ..core.canonical_turns import compute_turn_hash_from_raw, utcnow_iso
from ..core.exceptions import CanonicalSourceConflict, SourceEventTimeUnavailable
from ..types import AUDIENCE_ATTRIBUTION_VERSION, normalize_source_occurred_at
from .audience_proof import effective_attested_audience, verify_source_replay_audience
from .audience_reassignment import _memberships, _placeholder, _record, source_fingerprint


TABLE = "canonical_source_event_times"
_SQLITE_GUARDS = {
    "trg_source_event_time_update": f"""CREATE TRIGGER trg_source_event_time_update
        BEFORE UPDATE ON {TABLE}
        BEGIN SELECT RAISE(ABORT, 'source event time is immutable'); END""",
    "trg_source_event_time_delete": f"""CREATE TRIGGER trg_source_event_time_delete
        BEFORE DELETE ON {TABLE}
        WHEN EXISTS (SELECT 1 FROM canonical_turns
                     WHERE canonical_turn_id=OLD.canonical_turn_id)
        BEGIN SELECT RAISE(ABORT, 'live source event time is immutable'); END""",
}
_PG_GUARD_BODY = """BEGIN
    IF TG_OP = 'UPDATE' OR EXISTS (SELECT 1 FROM canonical_turns
        WHERE canonical_turn_id=OLD.canonical_turn_id) THEN
        RAISE EXCEPTION 'source event time is immutable';
    END IF;
    RETURN OLD;
END;"""


def _sql(value):
    return " ".join(str(value).split()).rstrip(";").lower()


def ensure_source_event_time_schema(conn, dialect):
    """Add independently immutable event metadata in the bootstrap transaction."""
    _placeholder(dialect)
    canonical_type = "UUID" if dialect == "postgres" else "TEXT"
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        canonical_turn_id {canonical_type} NOT NULL PRIMARY KEY
            REFERENCES canonical_turns(canonical_turn_id) ON DELETE CASCADE,
        tenant_id TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
        occurred_at TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    if dialect == "sqlite":
        for definition in _SQLITE_GUARDS.values():
            conn.execute(definition.replace("CREATE TRIGGER", "CREATE TRIGGER IF NOT EXISTS", 1))
    else:
        conn.execute(f"""CREATE OR REPLACE FUNCTION guard_source_event_time()
            RETURNS trigger AS $$ {_PG_GUARD_BODY} $$ LANGUAGE plpgsql""")
        conn.execute(f"DROP TRIGGER IF EXISTS trg_source_event_time ON {TABLE}")
        conn.execute(f"""CREATE TRIGGER trg_source_event_time
            BEFORE UPDATE OR DELETE ON {TABLE}
            FOR EACH ROW EXECUTE FUNCTION guard_source_event_time()""")


def assert_source_event_time_schema(conn, dialect):
    """Refuse an incomplete event-time store without mutating a read-only DB."""
    required = {"canonical_turn_id", "tenant_id", "source_fingerprint", "occurred_at", "created_at"}
    if dialect == "sqlite":
        columns = conn.execute(f"PRAGMA table_info({TABLE})").fetchall()
        names = {row["name"] for row in columns}
        primary = {row["name"] for row in columns if row["pk"]}
        guards = {row["name"]: row["sql"] for row in conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (TABLE,),
        )}
        safe_guards = all(_sql(guards.get(name, "")) == _sql(definition)
                          for name, definition in _SQLITE_GUARDS.items())
        foreign = conn.execute(f"PRAGMA foreign_key_list({TABLE})").fetchall()
        safe_fk = any(row["from"] == "canonical_turn_id" and row["table"] == "canonical_turns"
                      and row["to"] == "canonical_turn_id" and row["on_delete"].upper() == "CASCADE"
                      for row in foreign)
        nonnull = {row["name"] for row in columns if row["notnull"]}
    else:
        columns = conn.execute(
            "SELECT column_name,is_nullable FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s", (TABLE,),
        ).fetchall()
        names = {row["column_name"] for row in columns}
        nonnull = {row["column_name"] for row in columns if row["is_nullable"] == "NO"}
        if not names:
            raise RuntimeError("source event time schema is missing")
        primary = {row["attname"] for row in conn.execute(
            "SELECT a.attname FROM pg_constraint c JOIN pg_attribute a "
            "ON a.attrelid=c.conrelid AND a.attnum=ANY(c.conkey) "
            "WHERE c.conrelid=%s::regclass AND c.contype='p'", (TABLE,),
        )}
        guard = conn.execute(
            "SELECT p.prosrc FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid "
            "JOIN pg_class c ON c.oid=t.tgrelid WHERE t.tgrelid=%s::regclass "
            "AND t.tgname='trg_source_event_time' AND t.tgenabled IN ('O','A') "
            "AND t.tgtype=27 AND NOT t.tgisinternal AND t.tgqual IS NULL AND t.tgnargs=0 "
            "AND t.tgattr::text='' AND p.proname='guard_source_event_time' "
            "AND p.pronamespace=c.relnamespace", (TABLE,),
        ).fetchone()
        safe_guards = guard is not None and _sql(guard["prosrc"]) == _sql(_PG_GUARD_BODY)
        foreign = conn.execute(
            "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid=%s::regclass AND contype='f' AND convalidated", (TABLE,),
        ).fetchall()
        safe_fk = any(row["definition"] == "FOREIGN KEY (canonical_turn_id) REFERENCES canonical_turns(canonical_turn_id) ON DELETE CASCADE" for row in foreign)
    if (not required.issubset(names) or not required.issubset(nonnull)
            or primary != {"canonical_turn_id"} or not safe_guards or not safe_fk):
        raise RuntimeError("source event time schema is incomplete or unguarded")


def _exact_source(conn, canonical_id, owner, tenant_id, dialect, *, lock=False):
    """Verify present physical roles, immutable membership, and effective audience."""
    p = _placeholder(dialect)
    source_rows = _memberships(conn, {canonical_id}, dialect)
    if len(source_rows) != 1 or source_rows[0]["canonical_turn_id"] != canonical_id:
        raise CanonicalSourceConflict("source occurrence requires one exact user membership")
    source = source_rows[0]
    assistant_id = source["assistant_canonical_turn_id"]
    if len(_memberships(conn, {canonical_id, assistant_id}, dialect)) != 1:
        raise CanonicalSourceConflict("source occurrence pair membership overlaps")
    suffix = " FOR UPDATE" if lock and dialect == "postgres" else ""
    rows = {
        str(row["canonical_turn_id"]): _record(row)
        for row in conn.execute(
            f"SELECT * FROM canonical_turns WHERE canonical_turn_id IN ({p},{p}) "
            f"ORDER BY CASE WHEN canonical_turn_id={p} THEN 0 ELSE 1 END{suffix}",
            (canonical_id, assistant_id, canonical_id),
        ).fetchall()
    }
    user, assistant = rows.get(canonical_id), rows.get(assistant_id)
    if user is None or assistant is None or user is assistant:
        raise CanonicalSourceConflict("source occurrence pair is missing")
    audience = effective_attested_audience(conn, source, dialect=dialect)
    scopes = conn.execute(
        f"SELECT * FROM conversations WHERE conversation_id IN ({p},{p})",
        (owner, audience),
    ).fetchall()
    scopes = {row["conversation_id"]: dict(row) for row in scopes}
    if any(key not in scopes or scopes[key]["tenant_id"] != tenant_id
           or scopes[key]["phase"] == "deleted" or scopes[key]["deleted_at"] is not None
           for key in (owner, audience)) or scopes[owner]["phase"] == "merged":
        raise CanonicalSourceConflict("source occurrence lifecycle or tenant is ineligible")
    tombstones = conn.execute(
        f"SELECT deleted FROM conversation_lifecycle WHERE conversation_id IN ({p},{p})",
        (owner, audience),
    ).fetchall()
    if any(row["deleted"] for row in tombstones):
        raise CanonicalSourceConflict("source occurrence lifecycle is deleted")
    if (source["tenant_id"] != tenant_id or not audience or int(source["pair_version"]) != 1
            or user["conversation_id"] != owner or assistant["conversation_id"] != owner
            or not (user["user_content"] or "").strip()
            or (user["assistant_content"] or "").strip()
            or (assistant["user_content"] or "").strip()
            or not (assistant["assistant_content"] or "").strip()
            or int(user["turn_group_number"]) != int(assistant["turn_group_number"])
            or user["turn_hash"] != source["canonical_turn_hash"]
            or assistant["turn_hash"] != source["assistant_turn_hash"]
            or hashlib.sha256(user["user_content"].encode()).hexdigest() != source["canonical_body_sha256"]
            or user["sender_actor_id"] != source["source_actor_id"]
            or user["source_message_id"] != source["message_id"]
            or user["origin_channel_id"] != source["channel_id"]
            or user["reply_target_message_id"] != source["reply_target_message_id"]
            or user["audience_conversation_id"] != audience
            or int(user["audience_attribution_version"] or 0) != AUDIENCE_ATTRIBUTION_VERSION):
        raise CanonicalSourceConflict("source occurrence body, identity, or audience disagrees")
    for row in (user, assistant):
        actual, normalized_user, normalized_assistant = compute_turn_hash_from_raw(
            row["user_content"] or "", row["assistant_content"] or "",
        )
        if (actual != row["turn_hash"] or normalized_user != row["normalized_user_text"]
                or normalized_assistant != row["normalized_assistant_text"]):
            raise CanonicalSourceConflict("source occurrence canonical body hash is stale")
    legacy_audience = (not assistant["audience_conversation_id"]
                       and int(assistant["audience_attribution_version"] or 0) == 0)
    if not legacy_audience and (
        assistant["audience_conversation_id"] != audience
        or int(assistant["audience_attribution_version"] or 0) != AUDIENCE_ATTRIBUTION_VERSION
    ):
        raise CanonicalSourceConflict("source occurrence assistant audience disagrees")
    if legacy_audience or (assistant["sender_actor_id"] or ""):
        raise SourceEventTimeUnavailable("legacy assistant metadata cannot attest occurrence")
    return source, user


def record_source_event_time(conn, row, claim, *, dialect):
    """Persist one optional timestamp within successful exact source admission."""
    if "occurred_at" not in claim:
        return False
    try:
        occurred_at = normalize_source_occurred_at(claim["occurred_at"])
    except (TypeError, ValueError) as exc:
        raise CanonicalSourceConflict("invalid source occurrence timestamp") from exc
    p = _placeholder(dialect)
    tenant = conn.execute(f"SELECT tenant_id FROM conversations WHERE conversation_id={p}",
                          (row.conversation_id,)).fetchone()
    if tenant is None:
        raise CanonicalSourceConflict("source occurrence owner is missing")
    existing = conn.execute(f"SELECT * FROM {TABLE} WHERE canonical_turn_id={p}",
                            (row.canonical_turn_id,)).fetchone()
    if existing is not None and existing["occurred_at"] != occurred_at:
        raise CanonicalSourceConflict("source occurrence timestamp conflicts with prior proof")
    source, current = _exact_source(conn, row.canonical_turn_id, row.conversation_id,
                                    tenant["tenant_id"], dialect, lock=True)
    original = verify_source_replay_audience(
        conn, canonical_turn_id=row.canonical_turn_id,
        current_audience=current["audience_conversation_id"],
        incoming_audience=row.audience_conversation_id, dialect=dialect,
    )
    if (source["audience_conversation_id"] != original
            or current["user_content"] != row.user_content
            or current["turn_hash"] != row.turn_hash
            or current["sender_actor_id"] != row.sender_actor_id
            or any(str(source.get(key) or "") != str(value)
                   for key, value in claim.items() if key != "occurred_at")):
        raise CanonicalSourceConflict("source occurrence claim disagrees with exact source")
    expected = (tenant["tenant_id"], source_fingerprint(source), occurred_at)
    # The canonical pair lock also serializes concurrent metadata replays.
    existing = conn.execute(f"SELECT * FROM {TABLE} WHERE canonical_turn_id={p}",
                            (row.canonical_turn_id,)).fetchone()
    if existing is not None:
        if tuple(existing[key] for key in ("tenant_id", "source_fingerprint", "occurred_at")) != expected:
            raise CanonicalSourceConflict("source occurrence timestamp conflicts with prior proof")
        return False
    conn.execute(f"INSERT INTO {TABLE} (canonical_turn_id,tenant_id,source_fingerprint,occurred_at,created_at) "
                 f"VALUES ({','.join([p] * 5)})", (row.canonical_turn_id, *expected, utcnow_iso()))
    conn.execute(f"UPDATE actor_profiles SET card_dirty=1,card_build_marker='' WHERE tenant_id={p} AND actor_id={p}",
                 (tenant["tenant_id"], row.sender_actor_id))
    return True


def read_source_event_times(conn, keys, *, tenant_id, dialect):
    """Read source-bound timestamps in the caller's consistent read snapshot."""
    p = _placeholder(dialect)
    keys = sorted({(str(owner), str(canonical_id)) for owner, canonical_id in keys})
    if len(keys) > 2000:
        raise ValueError("source occurrence lookup exceeds its bounded key budget")
    found = {}
    for offset in range(0, len(keys), 200):
        batch = keys[offset:offset + 200]
        by_id = {canonical_id: owner for owner, canonical_id in batch}
        if len(by_id) != len(batch):
            raise ValueError("source occurrence keys disagree about owner")
        rows = conn.execute(f"SELECT * FROM {TABLE} WHERE tenant_id={p} "
                            f"AND canonical_turn_id IN ({','.join([p] * len(batch))})",
                            [tenant_id, *by_id]).fetchall()
        for row in rows:
            canonical_id = str(row["canonical_turn_id"])
            try:
                source, _ = _exact_source(conn, canonical_id, by_id[canonical_id], tenant_id, dialect)
                if source_fingerprint(source) != row["source_fingerprint"]:
                    continue
                value = normalize_source_occurred_at(row["occurred_at"])
            except (CanonicalSourceConflict, SourceEventTimeUnavailable, TypeError, ValueError):
                continue
            found[(by_id[canonical_id], canonical_id)] = value
    return found
