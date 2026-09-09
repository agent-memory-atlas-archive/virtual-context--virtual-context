"""Backend-equivalent guards for one authorized assistant channel transition."""

from __future__ import annotations


def authorized_channel_transition_sql(dialect: str) -> str:
    """Match an installed receipt, immutable source pair, and channel-only CAS."""
    if dialect not in {"sqlite", "postgres"}:
        raise ValueError("unsupported channel enrichment dialect")
    same = "IS" if dialect == "sqlite" else "IS NOT DISTINCT FROM"
    unchanged = (
        "canonical_turn_id",
        "conversation_id",
        "turn_group_number",
        "sort_key",
        "turn_hash",
        "hash_version",
        "normalized_user_text",
        "normalized_assistant_text",
        "user_content",
        "assistant_content",
        "user_raw_content",
        "assistant_raw_content",
        "sender",
        "origin_channel_label",
        "sender_actor_id",
        "source_message_id",
        "reply_target_message_id",
        "reply_subject_actor_id",
        "reply_subject_label",
        "reply_target_body",
        "reply_attribution_version",
        "origin_conversation_id",
        "audience_conversation_id",
        "audience_attribution_version",
        "source_batch_id",
        "created_at",
        "first_seen_at",
        "updated_at",
    )
    stable = "\n AND ".join(f"OLD.{field} {same} NEW.{field}" for field in unchanged)
    return f"""EXISTS (
        SELECT 1 FROM canonical_assistant_channel_enrichments ce
        JOIN assistant_channel_enrichment_operations op
          ON op.operation_id = ce.operation_id
         AND op.manifest_digest = ce.manifest_digest
         AND op.tenant_id = ce.tenant_id
         AND op.owner_conversation_id = ce.owner_conversation_id
         AND op.audience_conversation_id = ce.audience_conversation_id
         AND op.version = 1
        JOIN conversations owner
          ON owner.conversation_id = OLD.conversation_id
         AND owner.tenant_id = ce.tenant_id
         AND owner.lifecycle_epoch = op.expected_lifecycle_epoch
         AND owner.phase = 'active'
        JOIN canonical_message_sources src
          ON src.assistant_canonical_turn_id = ce.assistant_canonical_turn_id
         AND src.canonical_turn_id = ce.user_canonical_turn_id
         AND src.tenant_id = ce.tenant_id
         AND src.canonical_turn_hash = ce.user_turn_hash
         AND src.assistant_turn_hash = ce.assistant_turn_hash
         AND src.channel_id = ce.to_channel_id
         AND src.pair_version = 1
        WHERE ce.assistant_canonical_turn_id = OLD.canonical_turn_id
          AND ce.owner_conversation_id = OLD.conversation_id
          AND ce.assistant_turn_hash = OLD.turn_hash
          AND ce.audience_conversation_id = OLD.audience_conversation_id
          AND ce.audience_attribution_version = OLD.audience_attribution_version
          AND ce.audience_attribution_version = 1
          AND ce.from_channel_id = '' AND OLD.origin_channel_id = ''
          AND ce.to_channel_id <> '' AND ce.to_channel_id = NEW.origin_channel_id
          AND OLD.user_content = '' AND OLD.assistant_content <> ''
          AND OLD.sender = '' AND OLD.sender_actor_id = ''
          AND OLD.source_message_id = '' AND OLD.reply_subject_actor_id = ''
          AND {stable}
    )"""


def receipted_channel_guard_predicate(dialect: str) -> str:
    """Exact refusal predicate used by both backend guards and schema checks."""
    allowed = authorized_channel_transition_sql(dialect)
    different = "IS NOT" if dialect == "sqlite" else "IS DISTINCT FROM"
    return f"""(
        OLD.origin_channel_id {different} NEW.origin_channel_id OR
        OLD.origin_channel_label {different} NEW.origin_channel_label
    ) AND (
        EXISTS (SELECT 1 FROM canonical_audience_reassignments
                 WHERE canonical_turn_id = OLD.canonical_turn_id)
        OR EXISTS (SELECT 1 FROM canonical_assistant_channel_enrichments
                    WHERE assistant_canonical_turn_id = OLD.canonical_turn_id)
    ) AND NOT ({allowed})"""


def receipted_channel_guard_function_body(dialect: str = "postgres") -> str:
    """Return the exact installed PostgreSQL body for read-only capability checks."""
    if dialect != "postgres":
        raise ValueError("channel guard function body requires postgres")
    guarded = receipted_channel_guard_predicate(dialect)
    return f""" BEGIN
            IF {guarded} THEN
                RAISE EXCEPTION 'receipted channel enrichment requires authorization'
                    USING ERRCODE = 'integrity_constraint_violation';
            END IF;
            RETURN NEW;
        END; """


def ensure_receipted_channel_guard(conn, dialect: str) -> None:
    """Preserve previous audience/enrichment receipts on every channel writer."""
    guarded = receipted_channel_guard_predicate(dialect)
    name = "trg_guard_receipted_assistant_channel_update"
    if dialect == "sqlite":
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute(f"""CREATE TRIGGER {name}
            BEFORE UPDATE OF origin_channel_id, origin_channel_label ON canonical_turns
            FOR EACH ROW WHEN {guarded}
            BEGIN SELECT RAISE(ABORT, 'receipted channel enrichment requires authorization'); END""")
        return
    # Caller holds a transaction for replacement; no interval without a guard.
    conn.execute(f"""CREATE OR REPLACE FUNCTION vc_guard_receipted_assistant_channel_update()
        RETURNS trigger AS $${receipted_channel_guard_function_body(dialect)}$$ LANGUAGE plpgsql""")
    conn.execute(f"DROP TRIGGER IF EXISTS {name} ON canonical_turns")
    conn.execute(f"""CREATE TRIGGER {name}
        BEFORE UPDATE OF origin_channel_id, origin_channel_label ON canonical_turns
        FOR EACH ROW EXECUTE FUNCTION vc_guard_receipted_assistant_channel_update()""")
