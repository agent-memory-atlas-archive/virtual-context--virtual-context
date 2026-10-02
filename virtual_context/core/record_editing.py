"""Admin edits to a conversation's record, and the rebuild of what derives from it.

An operation changes the record synchronously: turns are edited, untagged or
removed, and every change is audited in ``turn_edits``. The derived state
(segment summaries, facts, chunk embeddings, topic summaries) is rebuilt
afterwards by :meth:`RecordEditor.process`, which a host runs in the
background. ``process`` is idempotent: rerunning an interrupted operation
finishes the segments it had not reached.

Who may call these operations is the host's decision; the engine records the
``actor`` it is given.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

logger = logging.getLogger(__name__)


class RecordEditor:
    def __init__(self, engine: "VirtualContextEngine") -> None:
        self._engine = engine

    @property
    def _store(self):
        return self._engine._store

    @property
    def _conversation_id(self) -> str:
        return self._engine.config.conversation_id

    # -- record changes ----------------------------------------------------

    def resolve_topic(self, tag: str) -> list[str]:
        """The stored tag ``tag`` names (case-insensitive, aliases followed), plus its aliases."""
        wanted = (tag or "").strip()
        if not wanted:
            return []
        aliases = dict(self._store.get_tag_aliases(conversation_id=self._conversation_id) or {})
        known = {
            ts.tag for ts in self._store.get_all_tags(conversation_id=self._conversation_id)
        }
        known.update(aliases.values())
        canonical = aliases.get(wanted) or next(
            (t for t in sorted(known) if t == wanted),
            next((t for t in sorted(known) if t.lower() == wanted.lower()), None),
        )
        if canonical is None:
            alias = next((a for a in aliases if a.lower() == wanted.lower()), None)
            canonical = aliases[alias] if alias else None
        if canonical is None:
            return []
        return [canonical, *sorted(a for a, c in aliases.items() if c == canonical and a != canonical)]

    def forget_topic(self, tag: str, *, actor: str, reason: str = "") -> dict:
        tags = self.resolve_topic(tag)
        if not tags:
            return {"operation_id": "", "topic": tag, "found": False}
        report = self._store.forget_topic_records(
            self._conversation_id, tags, actor=actor, reason=reason,
            operation_id=str(uuid.uuid4()),
        )
        self._refresh_index(self._store.get_record_edit_operation(report["operation_id"]))
        return {**report, "topic": tags[0], "found": True}

    def edit_turn(
        self, canonical_turn_id: str, *, user_content: str | None = None,
        assistant_content: str | None = None, actor: str, reason: str = "",
    ) -> dict:
        report = self._store.edit_canonical_turn(
            self._conversation_id, canonical_turn_id, user_content=user_content,
            assistant_content=assistant_content, actor=actor, reason=reason,
            operation_id=str(uuid.uuid4()),
        )
        self._refresh_index(self._store.get_record_edit_operation(report["operation_id"]))
        return report

    def remove_turn(self, canonical_turn_id: str, *, actor: str, reason: str = "") -> dict:
        report = self._store.remove_canonical_turn(
            self._conversation_id, canonical_turn_id, actor=actor, reason=reason,
            operation_id=str(uuid.uuid4()),
        )
        self._refresh_index(self._store.get_record_edit_operation(report["operation_id"]))
        return report

    def _refresh_index(self, op: dict | None) -> None:
        """Bring the live turn index in line with the rows this operation changed."""
        if not op:
            return
        index = self._engine._turn_tag_index
        edits = self._store.get_turn_edits(
            self._conversation_id, operation_id=op["operation_id"], limit=100_000,
        )
        index.remove_canonical_turns(e["canonical_turn_id"] for e in edits if e["action"] == "remove")
        for edit in edits:
            if edit["action"] == "forget_untag":
                tags = json.loads(edit["after_tags_json"] or "[]")
                topical = [t for t in tags if t != "_general"]
                index.set_turn_tags(edit["canonical_turn_id"], tags, topical[0] if topical else "_general")
        paging = getattr(self._engine, "_paging", None)
        working_set = getattr(paging, "working_set", None)
        if isinstance(working_set, dict):
            for tag in op.get("forgotten", []):
                working_set.pop(tag, None)

    # -- review ------------------------------------------------------------

    def facts(self, topic: str | None = None, *, limit: int = 200) -> list[dict]:
        """Current facts with their trust state and the source turns behind them.

        This is what an admin reviews before deciding to edit or remove the
        turn a fact came from.
        """
        tags = self.resolve_topic(topic) if topic else None
        if topic and not tags:
            return []
        facts = self._store.query_facts(
            conversation_id=self._conversation_id, tags=tags or None, limit=limit,
        )
        refs = {f.segment_ref for f in facts if f.segment_ref}
        turn_ids: dict[str, list[str]] = {}
        for ref in refs:
            segment = self._store.get_segment(ref, conversation_id=self._conversation_id)
            ids = getattr(getattr(segment, "metadata", None), "canonical_turn_ids", None) or []
            turn_ids[ref] = [str(i) for i in ids]
        rows = self._turn_rows(sorted({i for ids in turn_ids.values() for i in ids}))
        by_id = {str(row.canonical_turn_id): row for row in rows}
        return [
            {
                "fact_id": f.id,
                "fact": f.what or f"{f.subject} {f.verb} {f.object}",
                "trust_state": f.trust_state,
                "topic": (f.tags or [""])[0],
                "segment_ref": f.segment_ref,
                "source_turns": [
                    {
                        "turn_id": turn_id,
                        "user": by_id[turn_id].user_content,
                        "assistant": by_id[turn_id].assistant_content,
                    }
                    for turn_id in turn_ids.get(f.segment_ref, []) if turn_id in by_id
                ],
            }
            for f in facts
        ]

    def reject_fact(self, fact_id: str, *, actor: str, reason: str = "") -> dict:
        """Reject a fact: it, and the same statement extracted again from the same turns, is never served."""
        return self._store.record_fact_verdict(
            self._conversation_id, fact_id, "rejected", actor=actor, reason=reason,
        )

    def restore_fact(self, fact_id: str, *, actor: str, reason: str = "") -> dict:
        """Lift a rejection, so the fact is served again."""
        return self._store.record_fact_verdict(
            self._conversation_id, fact_id, "restored", actor=actor, reason=reason,
        )

    def verify_facts(self) -> dict:
        """Re-derive every current fact's trust state from its source turns."""
        refs = {
            f.segment_ref
            for f in self._store.query_facts(conversation_id=self._conversation_id, limit=1_000_000)
            if f.segment_ref and f.trust_state != "retracted"
        }
        return self._store.refresh_fact_trust(self._conversation_id, refs) if refs else {}

    # -- derived rebuild ---------------------------------------------------

    def status(self, operation_id: str | None = None) -> list[dict]:
        if operation_id:
            op = self._store.get_record_edit_operation(operation_id)
            return [op] if op else []
        return self._store.list_record_edit_operations(self._conversation_id, active_only=True)

    def process(self, operation_id: str, *, lease_operation_id: str | None = None) -> dict | None:
        """Rebuild everything derived from the turns this operation changed.

        ``lease_operation_id`` is the compaction operation the caller holds for
        this conversation, so rebuild writes are fenced like compaction's.
        """
        from ..types import CompactionLeaseLost

        store = self._store
        op = store.get_record_edit_operation(operation_id)
        if op is None or op["status"] == "completed":
            return op
        store.update_record_edit_operation(operation_id, status="running")
        try:
            if op["retag"]:
                self._engine.retag_canonical_turns(
                    canonical_turn_ids=set(op["retag"]), only_general=False,
                )
                for row in self._turn_rows(op["retag"]):
                    self._engine._turn_tag_index.set_turn_tags(
                        row.canonical_turn_id, list(row.tags or []), row.primary_tag,
                    )
                store.update_record_edit_operation(operation_id, retagged=True)
            pipeline = self._engine._compaction
            for segment in op["segments"]:
                ref = segment["ref"]
                if ref in op["done"]:
                    continue
                try:
                    outcome = pipeline.rebuild_segment(
                        ref, segment["turn_ids"], forgotten=op["forgotten"],
                        operation_id=lease_operation_id,
                    )
                    logger.info("RECORD_EDIT op=%s segment=%s %s", operation_id[:8], ref[:12], outcome)
                    store.update_record_edit_operation(operation_id, done=[ref])
                except CompactionLeaseLost:
                    raise
                except Exception as exc:  # noqa: BLE001 - recorded, retried on rerun
                    logger.warning("RECORD_EDIT op=%s segment=%s failed: %s", operation_id[:8], ref[:12], exc)
                    store.update_record_edit_operation(operation_id, failed=[ref])
            self._rebuild_topics(op)
            pipeline._refresh_shared_retrieval_snapshots()
            final = store.get_record_edit_operation(operation_id)
            status = "failed" if final["failed"] else "completed"
            return store.update_record_edit_operation(operation_id, status=status)
        except Exception as exc:
            logger.exception("RECORD_EDIT op=%s failed", operation_id[:8])
            store.update_record_edit_operation(operation_id, status="failed", error=str(exc))
            raise

    def _turn_rows(self, turn_ids):
        rows = self._store.get_canonical_turn_rows_by_id(
            [(self._conversation_id, t) for t in turn_ids], internal_validation=True,
        )
        return list(rows.values())

    def _rebuild_topics(self, op: dict) -> None:
        present = {ts.tag for ts in self._store.get_all_tags(conversation_id=self._conversation_id)}
        topics = [t for t in op["tags"] if t != "_general"]
        gone = [t for t in topics if t not in present]
        if gone:
            self._store.delete_tag_summaries(self._conversation_id, gone)
        rebuild = [t for t in topics if t in present]
        if rebuild:
            self._engine.backfill_tag_summaries(tags=rebuild)

    def resume(self) -> list[dict]:
        """Finish operations interrupted before their rebuild completed."""
        return [
            self.process(op["operation_id"])
            for op in self._store.list_record_edit_operations(self._conversation_id, active_only=True)
        ]
