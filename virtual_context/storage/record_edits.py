"""Admin edits to the conversation record, shared by the SQL backends.

The conversation is the record and everything else is derived from it. An
edit changes canonical turns directly and writes one immutable ``turn_edits``
row per changed turn, holding the text and tags it had before. Segments whose
evidence changed are listed on a ``record_edit_operations`` row together with
the turn ids they held, so the derived rebuild can run in the background and
resume after an interruption.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid

GENERAL_TAG = "_general"
_ACTIVE = ("pending", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tags(raw) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [tag for tag in value if isinstance(tag, str) and tag] if isinstance(value, list) else []


def _topical(tags) -> list[str]:
    return [tag for tag in tags if tag != GENERAL_TAG]


def _turn_ids(metadata_json) -> list[str]:
    try:
        ids = json.loads(metadata_json or "{}").get("canonical_turn_ids")
    except (TypeError, ValueError, AttributeError):
        return []
    return [str(i) for i in ids if i] if isinstance(ids, list) else []


class RecordEditMixin:
    def _ensure_record_edit_schema(self, conn):
        conn.execute("""CREATE TABLE IF NOT EXISTS turn_edits (
            edit_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
            canonical_turn_id TEXT NOT NULL, action TEXT NOT NULL,
            actor TEXT NOT NULL, reason TEXT NOT NULL, operation_id TEXT NOT NULL,
            created_at TEXT NOT NULL, before_user_content TEXT NOT NULL,
            before_assistant_content TEXT NOT NULL, before_tags_json TEXT NOT NULL,
            before_turn_hash TEXT NOT NULL, after_user_content TEXT,
            after_assistant_content TEXT, after_tags_json TEXT, after_turn_hash TEXT)""")
        columns = ("edit_id", "canonical_turn_id", "action", "actor", "reason",
                   "operation_id", "created_at", "before_user_content",
                   "before_assistant_content", "before_tags_json", "before_turn_hash",
                   "after_user_content", "after_assistant_content", "after_tags_json",
                   "after_turn_hash")
        if self._relational_dialect == "postgres":
            conn.execute("""CREATE OR REPLACE FUNCTION guard_turn_edit_content()
                RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
                IF (to_jsonb(NEW) - 'conversation_id') IS DISTINCT FROM
                   (to_jsonb(OLD) - 'conversation_id') THEN
                    RAISE EXCEPTION 'turn edit records are immutable';
                END IF;
                RETURN NEW;
                END $$""")
            if not conn.execute("""SELECT 1 FROM pg_trigger WHERE
                    tgrelid='turn_edits'::regclass AND tgname='guard_turn_edit_content'""").fetchone():
                conn.execute("""CREATE TRIGGER guard_turn_edit_content
                    BEFORE UPDATE ON turn_edits FOR EACH ROW
                    EXECUTE FUNCTION guard_turn_edit_content()""")
        elif not conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='trigger' AND name='guard_turn_edit_content'"
        ).fetchone():
            changed = " OR ".join(f"NEW.{key} IS NOT OLD.{key}" for key in columns)
            conn.execute(f"""CREATE TRIGGER guard_turn_edit_content
                BEFORE UPDATE ON turn_edits WHEN {changed}
                BEGIN SELECT RAISE(ABORT, 'turn edit records are immutable'); END""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_turn_edits_turn
            ON turn_edits (conversation_id, canonical_turn_id, created_at)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS record_edit_operations (
            operation_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
            action TEXT NOT NULL, target TEXT NOT NULL, actor TEXT NOT NULL,
            status TEXT NOT NULL, segments_json TEXT NOT NULL,
            done_json TEXT NOT NULL DEFAULT '[]', failed_json TEXT NOT NULL DEFAULT '[]',
            retag_json TEXT NOT NULL DEFAULT '[]', tags_json TEXT NOT NULL DEFAULT '[]',
            forgotten_json TEXT NOT NULL DEFAULT '[]', error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_record_edit_operations_owner
            ON record_edit_operations (conversation_id, status)""")

    # -- shared helpers --------------------------------------------------

    def _record_edit_lock(self, conn, conversation_id):
        p = self._placeholder
        lock = " FOR SHARE" if self._relational_dialect == "postgres" else ""
        state = conn.execute(
            f"SELECT deleted FROM conversation_lifecycle WHERE conversation_id={p}{lock}",
            (conversation_id,),
        ).fetchone()
        if state is not None and state["deleted"]:
            raise ValueError("Record edit refused for a deleted conversation")
        active = conn.execute(
            f"""SELECT operation_id FROM record_edit_operations
            WHERE conversation_id={p} AND status IN ('pending','running')""",
            (conversation_id,),
        ).fetchone()
        if active is not None:
            raise RuntimeError(
                f"record edit {active['operation_id']} is still being processed"
            )

    def _segments_holding(self, conn, conversation_id, turn_ids, tags=()):
        """Segments whose evidence includes a turn in ``turn_ids`` or carries a tag."""
        p = self._placeholder
        turn_ids, tags = set(turn_ids), set(tags)
        rows = conn.execute(
            f"SELECT ref, primary_tag, metadata_json FROM segments WHERE conversation_id={p}",
            (conversation_id,),
        ).fetchall()
        tag_rows = conn.execute(
            f"""SELECT st.segment_ref, st.tag FROM segment_tags st
            JOIN segments s ON s.ref=st.segment_ref WHERE s.conversation_id={p}""",
            (conversation_id,),
        ).fetchall()
        seg_tags: dict[str, set[str]] = {}
        for row in tag_rows:
            seg_tags.setdefault(row["segment_ref"], set()).add(row["tag"])
        held = []
        for row in rows:
            ids = _turn_ids(row["metadata_json"])
            all_tags = seg_tags.get(row["ref"], set()) | {row["primary_tag"]}
            if turn_ids.intersection(ids) or tags & all_tags:
                held.append({"ref": row["ref"], "turn_ids": ids,
                             "primary_tag": row["primary_tag"], "tags": sorted(all_tags)})
        return held

    def _delete_segment_rows(self, conn, refs):
        if not refs:
            return
        p = self._placeholder
        marks = ",".join([p] * len(refs))
        refs = list(refs)
        conn.execute(
            f"""DELETE FROM fact_embeddings WHERE fact_id IN
            (SELECT id FROM facts WHERE segment_ref IN ({marks}))""", refs,
        )
        for table, column in (("segment_tags", "segment_ref"), ("segment_chunks", "segment_ref"),
                              ("segment_tool_outputs", "segment_ref"), ("facts", "segment_ref"),
                              ("segments", "ref")):
            conn.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", refs)

    def _delete_turn_rows(self, conn, conversation_id, turn_ids):
        if not turn_ids:
            return
        p = self._placeholder
        marks = ",".join([p] * len(turn_ids))
        for table in ("canonical_turn_chunks", "canonical_turns"):
            conn.execute(
                f"DELETE FROM {table} WHERE conversation_id={p} AND canonical_turn_id IN ({marks})",
                [conversation_id, *turn_ids],
            )

    def _write_turn_edit(self, conn, row, *, action, actor, reason, operation_id,
                         after=None, after_tags=None, after_hash=None):
        """Audit one turn; ``after`` is ``(user, assistant)`` text, None for a removal."""
        p = self._placeholder
        after_user, after_assistant = after if after is not None else (None, None)
        conn.execute(
            f"""INSERT INTO turn_edits (edit_id,conversation_id,canonical_turn_id,action,actor,
            reason,operation_id,created_at,before_user_content,before_assistant_content,
            before_tags_json,before_turn_hash,after_user_content,after_assistant_content,
            after_tags_json,after_turn_hash)
            VALUES ({','.join([p] * 16)})""",
            (str(uuid.uuid4()), row["conversation_id"], str(row["canonical_turn_id"]), action,
             actor, reason, operation_id, _now(), row["user_content"] or "",
             row["assistant_content"] or "", row["tags_json"] or "[]", row["turn_hash"],
             after_user, after_assistant,
             None if after_tags is None else json.dumps(after_tags), after_hash),
        )

    def _open_record_edit(self, conn, *, operation_id, conversation_id, action, target,
                          actor, segments, retag=(), tags=(), forgotten=()):
        p = self._placeholder
        now = _now()
        # The record no longer supports what was extracted from these
        # segments; their facts are withheld until the rebuild replaces them.
        refs = sorted({segment["ref"] for segment in segments})
        if refs:
            conn.execute(
                f"""UPDATE facts SET trust_state='retracted' WHERE conversation_id={p}
                AND segment_ref IN ({','.join([p] * len(refs))})""",
                [conversation_id, *refs],
            )
        conn.execute(
            f"""INSERT INTO record_edit_operations (operation_id,conversation_id,action,target,
            actor,status,segments_json,retag_json,tags_json,forgotten_json,created_at,updated_at)
            VALUES ({','.join([p] * 12)})""",
            (operation_id, conversation_id, action, target, actor,
             "pending" if segments or retag or tags else "completed",
             json.dumps(segments), json.dumps(sorted(set(retag))),
             json.dumps(sorted(set(tags) - set(forgotten))), json.dumps(sorted(set(forgotten))),
             now, now),
        )

    def _turn_rows(self, conn, conversation_id, *, turn_ids=None, group=None, columns="*"):
        p = self._placeholder
        if group is not None:
            return conn.execute(
                f"""SELECT {columns} FROM canonical_turns WHERE conversation_id={p}
                AND turn_group_number={p} ORDER BY sort_key""",
                (conversation_id, group),
            ).fetchall()
        if turn_ids is not None:
            if not turn_ids:
                return []
            return conn.execute(
                f"""SELECT {columns} FROM canonical_turns WHERE conversation_id={p}
                AND canonical_turn_id IN ({','.join([p] * len(turn_ids))}) ORDER BY sort_key""",
                [conversation_id, *turn_ids],
            ).fetchall()
        return conn.execute(
            f"SELECT {columns} FROM canonical_turns WHERE conversation_id={p} ORDER BY sort_key",
            (conversation_id,),
        ).fetchall()

    # -- operations ------------------------------------------------------

    def forget_topic_records(self, conversation_id, tags, *, actor, reason="", operation_id=None):
        """Remove ``tags`` from every turn; remove turns left with no topic."""
        operation_id = operation_id or str(uuid.uuid4())
        forgotten = {tag for tag in tags if tag and tag != GENERAL_TAG}
        if not forgotten:
            raise ValueError("forget_topic needs a topic tag")
        p = self._placeholder
        with self._relational_connection(write=True, scope=f"vc-record-edit:{conversation_id}") as conn:
            self._record_edit_lock(conn, conversation_id)
            index = self._turn_rows(
                conn, conversation_id,
                columns="canonical_turn_id,turn_group_number,tags_json",
            )
            new_tags = {}
            for row in index:
                before = _tags(row["tags_json"])
                if forgotten.intersection(before):
                    new_tags[str(row["canonical_turn_id"])] = [t for t in before if t not in forgotten]
            groups: dict[object, list] = {}
            for row in index:
                key = row["turn_group_number"] if row["turn_group_number"] >= 0 else str(row["canonical_turn_id"])
                groups.setdefault(key, []).append(row)
            removed, untagged = [], []
            for rows in groups.values():
                ids = [str(row["canonical_turn_id"]) for row in rows]
                if not any(i in new_tags for i in ids):
                    continue
                remaining = [
                    _topical(new_tags[i] if i in new_tags else _tags(row["tags_json"]))
                    for i, row in zip(ids, rows)
                ]
                if any(remaining):
                    untagged.extend(i for i in ids if i in new_tags)
                else:
                    removed.extend(ids)
            held = self._segments_holding(conn, conversation_id, set(removed) | set(untagged), forgotten)
            full = {str(r["canonical_turn_id"]): r for r in self._turn_rows(
                conn, conversation_id, turn_ids=sorted(set(removed) | set(untagged)))}
            for turn_id in untagged:
                after = new_tags[turn_id]
                self._write_turn_edit(conn, full[turn_id], action="forget_untag", actor=actor,
                                      reason=reason, operation_id=operation_id, after_tags=after,
                                      after_hash=full[turn_id]["turn_hash"],
                                      after=(full[turn_id]["user_content"] or "",
                                             full[turn_id]["assistant_content"] or ""))
                primary = _topical(after)[0] if _topical(after) else GENERAL_TAG
                conn.execute(
                    f"""UPDATE canonical_turns SET tags_json={p}, primary_tag={p}, updated_at={p}
                    WHERE conversation_id={p} AND canonical_turn_id={p}""",
                    (json.dumps(after), primary, _now(), conversation_id, turn_id),
                )
            for turn_id in removed:
                self._write_turn_edit(conn, full[turn_id], action="remove", actor=actor,
                                      reason=reason, operation_id=operation_id)
            self._delete_turn_rows(conn, conversation_id, removed)
            # The topic's own segments go now, with their summaries and facts;
            # their surviving turns are rebuilt into the same refs.
            topic_refs = [seg["ref"] for seg in held if seg["primary_tag"] in forgotten]
            self._delete_segment_rows(conn, topic_refs)
            marks = ",".join([p] * len(forgotten))
            conn.execute(
                f"""DELETE FROM segment_tags WHERE tag IN ({marks}) AND segment_ref IN
                (SELECT ref FROM segments WHERE conversation_id={p})""",
                [*sorted(forgotten), conversation_id],
            )
            for table in ("tag_summaries", "tag_summary_embeddings"):
                conn.execute(
                    f"DELETE FROM {table} WHERE conversation_id={p} AND tag IN ({marks})",
                    [conversation_id, *sorted(forgotten)],
                )
            conn.execute(
                f"""DELETE FROM tag_aliases WHERE conversation_id={p}
                AND (alias IN ({marks}) OR canonical IN ({marks}))""",
                [conversation_id, *sorted(forgotten), *sorted(forgotten)],
            )
            # A segment without turn provenance cannot be rebuilt; dropping the
            # tag (or the whole topic segment, above) is all that applies to it.
            segments = [{"ref": seg["ref"], "turn_ids": seg["turn_ids"]} for seg in held
                        if seg["turn_ids"]]
            topics = {tag for seg in held for tag in seg["tags"]}
            self._open_record_edit(
                conn, operation_id=operation_id, conversation_id=conversation_id,
                action="forget_topic", target=",".join(sorted(forgotten)), actor=actor,
                segments=segments, tags=topics, forgotten=forgotten,
            )
        return {"operation_id": operation_id, "turns_untagged": len(untagged),
                "turns_removed": len(removed), "segments_deleted": len(topic_refs),
                "segments_to_rebuild": len(segments)}

    def edit_canonical_turn(self, conversation_id, canonical_turn_id, *, user_content=None,
                            assistant_content=None, actor, reason="", operation_id=None):
        """Replace a turn's text, keeping the original in ``turn_edits``.

        ``None`` leaves that side as it is.
        """
        from ..core.canonical_turns import HASH_VERSION, compute_turn_hash_from_raw

        operation_id = operation_id or str(uuid.uuid4())
        if user_content is None and assistant_content is None:
            raise ValueError("edit_turn needs user_content or assistant_content")
        p = self._placeholder
        with self._relational_connection(write=True, scope=f"vc-record-edit:{conversation_id}") as conn:
            self._record_edit_lock(conn, conversation_id)
            rows = self._turn_rows(conn, conversation_id, turn_ids=[canonical_turn_id])
            if not rows:
                raise KeyError(f"turn {canonical_turn_id} not found")
            row = rows[0]
            user_text = row["user_content"] if user_content is None else user_content
            assistant_text = row["assistant_content"] if assistant_content is None else assistant_content
            if not (user_text or "").strip() and not (assistant_text or "").strip():
                raise ValueError("an edit cannot leave a turn empty; use remove_turn instead")
            turn_hash, norm_user, norm_assistant = compute_turn_hash_from_raw(
                user_text or "", assistant_text or "", version=HASH_VERSION,
            )
            self._write_turn_edit(conn, row, action="edit", actor=actor, reason=reason,
                                  operation_id=operation_id, after=(user_text, assistant_text),
                                  after_tags=_tags(row["tags_json"]), after_hash=turn_hash)
            # An edited turn no longer matches the message its source attested.
            conn.execute(
                f"""DELETE FROM canonical_message_sources
                WHERE canonical_turn_id={p} OR assistant_canonical_turn_id={p}""",
                (canonical_turn_id, canonical_turn_id),
            )
            conn.execute(
                f"""UPDATE canonical_turns SET user_content={p}, assistant_content={p},
                user_raw_content={p}, assistant_raw_content={p}, normalized_user_text={p},
                normalized_assistant_text={p}, turn_hash={p}, hash_version={p}, updated_at={p}
                WHERE conversation_id={p} AND canonical_turn_id={p}""",
                (user_text, assistant_text,
                 row["user_raw_content"] if user_content is None else None,
                 row["assistant_raw_content"] if assistant_content is None else None,
                 norm_user, norm_assistant, turn_hash,
                 HASH_VERSION, _now(), conversation_id, canonical_turn_id),
            )
            conn.execute(
                f"DELETE FROM canonical_turn_chunks WHERE conversation_id={p} AND canonical_turn_id={p}",
                (conversation_id, canonical_turn_id),
            )
            held = self._segments_holding(conn, conversation_id, {str(canonical_turn_id)})
            self._open_record_edit(
                conn, operation_id=operation_id, conversation_id=conversation_id,
                action="edit_turn", target=str(canonical_turn_id), actor=actor,
                segments=[{"ref": s["ref"], "turn_ids": s["turn_ids"]} for s in held],
                retag=[str(canonical_turn_id)], tags={t for s in held for t in s["tags"]},
            )
        return {"operation_id": operation_id, "turns_edited": 1, "segments_to_rebuild": len(held)}

    def remove_canonical_turn(self, conversation_id, canonical_turn_id, *, actor, reason="",
                              operation_id=None):
        """Remove a turn (both halves of its exchange), keeping the text in ``turn_edits``."""
        operation_id = operation_id or str(uuid.uuid4())
        with self._relational_connection(write=True, scope=f"vc-record-edit:{conversation_id}") as conn:
            self._record_edit_lock(conn, conversation_id)
            rows = self._turn_rows(conn, conversation_id, turn_ids=[canonical_turn_id])
            if not rows:
                raise KeyError(f"turn {canonical_turn_id} not found")
            if rows[0]["turn_group_number"] >= 0:
                rows = self._turn_rows(conn, conversation_id, group=rows[0]["turn_group_number"])
            ids = [str(row["canonical_turn_id"]) for row in rows]
            for row in rows:
                self._write_turn_edit(conn, row, action="remove", actor=actor, reason=reason,
                                      operation_id=operation_id)
            self._delete_turn_rows(conn, conversation_id, ids)
            held = self._segments_holding(conn, conversation_id, set(ids))
            self._open_record_edit(
                conn, operation_id=operation_id, conversation_id=conversation_id,
                action="remove_turn", target=str(canonical_turn_id), actor=actor,
                segments=[{"ref": s["ref"], "turn_ids": s["turn_ids"]} for s in held],
                tags={t for s in held for t in s["tags"]},
            )
        return {"operation_id": operation_id, "turns_removed": len(ids),
                "segments_to_rebuild": len(held)}

    # -- reads and progress ---------------------------------------------

    def get_turn_edits(self, conversation_id, *, canonical_turn_id=None, operation_id=None,
                       limit=100):
        p = self._placeholder
        where, params = f"conversation_id={p}", [conversation_id]
        if canonical_turn_id is not None:
            where += f" AND canonical_turn_id={p}"
            params.append(str(canonical_turn_id))
        if operation_id is not None:
            where += f" AND operation_id={p}"
            params.append(operation_id)
        with self._relational_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM turn_edits WHERE {where} ORDER BY created_at DESC, edit_id LIMIT {int(limit)}",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_tag_summaries(self, conversation_id, tags):
        """Drop the summaries of topics that no longer have any segment."""
        tags = sorted(set(tags))
        if not tags:
            return 0
        p = self._placeholder
        marks = ",".join([p] * len(tags))
        with self._relational_connection(write=True, scope=f"vc-record-edit:{conversation_id}") as conn:
            conn.execute(
                f"DELETE FROM tag_summary_embeddings WHERE conversation_id={p} AND tag IN ({marks})",
                [conversation_id, *tags],
            )
            cur = conn.execute(
                f"DELETE FROM tag_summaries WHERE conversation_id={p} AND tag IN ({marks})",
                [conversation_id, *tags],
            )
            return cur.rowcount or 0

    def get_retired_turns(self, conversation_id):
        """Turns an admin edit retired, keyed by the hash they had.

        The value is the replacement (``turn_hash``, ``user_content``,
        ``assistant_content``) for an edited turn and ``None`` for a removed one;
        a chain of edits resolves to its latest version.
        """
        p = self._placeholder
        with self._relational_connection() as conn:
            rows = conn.execute(
                f"""SELECT action, before_turn_hash, after_turn_hash, after_user_content,
                after_assistant_content FROM turn_edits
                WHERE conversation_id={p} AND action IN ('edit','remove')
                ORDER BY created_at, edit_id""",
                (conversation_id,),
            ).fetchall()
        retired = {}
        for row in rows:
            retired[row["before_turn_hash"]] = None if row["action"] == "remove" else {
                "turn_hash": row["after_turn_hash"],
                "user_content": row["after_user_content"] or "",
                "assistant_content": row["after_assistant_content"] or "",
            }
        resolved = {}
        for old, new in retired.items():
            seen = {old}
            while new is not None and new["turn_hash"] in retired and new["turn_hash"] not in seen:
                seen.add(new["turn_hash"])
                new = retired[new["turn_hash"]]
            if new is None or new["turn_hash"] != old:
                resolved[old] = new
        return resolved

    @staticmethod
    def _operation_dict(row):
        if row is None:
            return None
        op = dict(row)
        for key in ("segments", "done", "failed", "retag", "tags", "forgotten"):
            op[key] = json.loads(op.pop(f"{key}_json") or "[]")
        return op

    def get_record_edit_operation(self, operation_id):
        p = self._placeholder
        with self._relational_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM record_edit_operations WHERE operation_id={p}", (operation_id,),
            ).fetchone()
        return self._operation_dict(row)

    def list_record_edit_operations(self, conversation_id, *, active_only=False, limit=20):
        p = self._placeholder
        status = " AND status IN ('pending','running')" if active_only else ""
        with self._relational_connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM record_edit_operations WHERE conversation_id={p}{status}
                ORDER BY created_at DESC LIMIT {int(limit)}""",
                (conversation_id,),
            ).fetchall()
        return [self._operation_dict(row) for row in rows]

    def update_record_edit_operation(self, operation_id, *, done=(), failed=(), retagged=False,
                                     status=None, error=None):
        """Record progress; ``done``/``failed`` refs accumulate, ``retagged`` clears the retag list."""
        p = self._placeholder
        with self._relational_connection(write=True, scope=f"vc-record-edit-op:{operation_id}") as conn:
            lock = " FOR UPDATE" if self._relational_dialect == "postgres" else ""
            row = conn.execute(
                f"SELECT * FROM record_edit_operations WHERE operation_id={p}{lock}", (operation_id,),
            ).fetchone()
            if row is None:
                return None
            op = self._operation_dict(row)
            op["done"] = sorted(set(op["done"]) | set(done))
            op["failed"] = sorted((set(op["failed"]) | set(failed)) - set(done))
            if retagged:
                op["retag"] = []
            op["status"] = status or op["status"]
            if error is not None:
                op["error"] = error
            conn.execute(
                f"""UPDATE record_edit_operations SET done_json={p}, failed_json={p}, retag_json={p},
                status={p}, error={p}, updated_at={p} WHERE operation_id={p}""",
                (json.dumps(op["done"]), json.dumps(op["failed"]), json.dumps(op["retag"]),
                 op["status"], op["error"], _now(), operation_id),
            )
        return op
