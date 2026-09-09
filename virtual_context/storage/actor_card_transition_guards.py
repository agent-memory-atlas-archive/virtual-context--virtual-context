"""Canonical card invalidation SQL shared with read-only capability checks."""

from __future__ import annotations

from .channel_enrichment_guards import authorized_channel_transition_sql


def sqlite_actor_card_turn_source_update_sql() -> str:
    """Exact update guard: invalidate cited cards and retain authorized claims."""
    return f"""            CREATE TRIGGER trg_invalidate_actor_card_turn_source_update
            BEFORE UPDATE OF
                conversation_id, user_content, sender_actor_id,
                audience_conversation_id, audience_attribution_version,
                origin_channel_id, created_at, first_seen_at
            ON canonical_turns
            FOR EACH ROW
            WHEN OLD.conversation_id IS NOT NEW.conversation_id
              OR OLD.user_content IS NOT NEW.user_content
              OR OLD.sender_actor_id IS NOT NEW.sender_actor_id
              OR OLD.audience_conversation_id
                    IS NOT NEW.audience_conversation_id
              OR OLD.audience_attribution_version
                    IS NOT NEW.audience_attribution_version
              OR OLD.origin_channel_id IS NOT NEW.origin_channel_id
              OR OLD.created_at IS NOT NEW.created_at
              OR OLD.first_seen_at IS NOT NEW.first_seen_at
            BEGIN
                -- Authorship arms: a change to ANY of the author's rows —
                -- including provenance enrichment one-way fills on rows no
                -- card cites — queues re-curation only. Invalidation is the
                -- citation arm's job: only a change to a row a card CITES
                -- can falsify a rendered claim.
                UPDATE actor_profiles
                   SET card_dirty = 1,
                       card_build_marker = ''
                 WHERE actor_id = OLD.sender_actor_id
                   AND OLD.sender_actor_id <> ''
                   AND tenant_id = (
                       SELECT tenant_id
                         FROM conversations
                        WHERE conversation_id = OLD.conversation_id
                   );
                UPDATE actor_profiles
                   SET card_dirty = 1,
                       card_build_marker = ''
                 WHERE actor_id = NEW.sender_actor_id
                   AND NEW.sender_actor_id <> ''
                   AND tenant_id = (
                       SELECT tenant_id
                         FROM conversations
                        WHERE conversation_id = NEW.conversation_id
                   );
                UPDATE actor_profiles
                   SET card_dirty = 1, card_invalid = 1,
                       card_build_marker = ''
                 WHERE (tenant_id, actor_id) IN (
                       SELECT e.tenant_id, e.actor_id
                         FROM actor_card_entries e
                         JOIN actor_card_turn_sources s
                           ON s.entry_id = e.id
                          AND s.tenant_id = e.tenant_id
                        WHERE s.canonical_turn_id =
                              OLD.canonical_turn_id
                 );
                DELETE FROM actor_card_entries
                 WHERE id IN (
                       SELECT entry_id
                         FROM actor_card_turn_sources
                        WHERE canonical_turn_id =
                              OLD.canonical_turn_id
                 ) AND NOT EXISTS (
                       SELECT 1 FROM canonical_audience_reassignments ar
                        WHERE ar.canonical_turn_id = OLD.canonical_turn_id
                          AND ar.owner_conversation_id = OLD.conversation_id
                          AND ar.from_audience = OLD.audience_conversation_id
                          AND ar.to_audience = NEW.audience_conversation_id
                          AND ar.from_attribution_version = OLD.audience_attribution_version
                          AND ar.to_attribution_version = NEW.audience_attribution_version
                          AND ar.turn_hash = OLD.turn_hash
                          AND OLD.turn_hash IS NEW.turn_hash
                          AND OLD.conversation_id IS NEW.conversation_id
                          AND OLD.user_content IS NEW.user_content
                          AND OLD.sender_actor_id IS NEW.sender_actor_id
                          AND OLD.origin_channel_id IS NEW.origin_channel_id
                          AND OLD.created_at IS NEW.created_at
                          AND OLD.first_seen_at IS NEW.first_seen_at
                 ) AND NOT ({authorized_channel_transition_sql("sqlite")});
            END;"""


def postgres_actor_card_turn_source_function_body() -> str:
    """Exact trigger body for card invalidation and authorized claim retention."""
    return f"""
                BEGIN
                    -- PostgreSQL fires an UPDATE OF trigger when a column is
                    -- named in SET even if its value did not change. Canonical
                    -- tagging re-upserts the request-owned row and names these
                    -- identity/evidence columns, so treat an exact no-op as
                    -- the additive reconciliation it is. Otherwise every
                    -- normal live turn invalidates (and can delete) the card
                    -- entry that just cited it.
                    IF TG_OP = 'UPDATE'
                       AND OLD.conversation_id
                           IS NOT DISTINCT FROM NEW.conversation_id
                       AND OLD.user_content
                           IS NOT DISTINCT FROM NEW.user_content
                       AND OLD.sender_actor_id
                           IS NOT DISTINCT FROM NEW.sender_actor_id
                       AND OLD.audience_conversation_id
                           IS NOT DISTINCT FROM NEW.audience_conversation_id
                       AND OLD.audience_attribution_version
                           IS NOT DISTINCT FROM
                               NEW.audience_attribution_version
                       AND OLD.origin_channel_id
                           IS NOT DISTINCT FROM NEW.origin_channel_id
                       AND OLD.created_at
                           IS NOT DISTINCT FROM NEW.created_at
                       AND OLD.first_seen_at
                           IS NOT DISTINCT FROM NEW.first_seen_at THEN
                        RETURN NEW;
                    END IF;

                    -- Authorship arms: changes to ANY of the author's
                    -- rows (including provenance enrichment on rows no card
                    -- cites) queue a re-curation only. Invalidation is the
                    -- citation arm's job below.
                    UPDATE actor_profiles p
                       SET card_dirty = 1,
                           card_build_marker = ''
                      FROM conversations c
                     WHERE c.conversation_id = OLD.conversation_id
                       AND p.tenant_id = c.tenant_id
                       AND p.actor_id = OLD.sender_actor_id
                       AND OLD.sender_actor_id <> '';

                    IF TG_OP = 'UPDATE' THEN
                        UPDATE actor_profiles p
                           SET card_dirty = 1,
                               card_build_marker = ''
                          FROM conversations c
                         WHERE c.conversation_id = NEW.conversation_id
                           AND p.tenant_id = c.tenant_id
                           AND p.actor_id = NEW.sender_actor_id
                           AND NEW.sender_actor_id <> '';
                    END IF;

                    UPDATE actor_profiles p
                       SET card_dirty = 1, card_invalid = 1,
                           card_build_marker = ''
                      FROM actor_card_entries e,
                           actor_card_turn_sources s
                     WHERE s.canonical_turn_id = OLD.canonical_turn_id
                       AND e.id = s.entry_id
                       AND e.tenant_id = s.tenant_id
                       AND p.tenant_id = e.tenant_id
                       AND p.actor_id = e.actor_id;

                    -- An audited scope-only transition retains immutable
                    -- claims for source reprojection and re-admission. The
                    -- invalid flag above still blocks serving stale cards.
                    IF TG_OP = 'UPDATE' THEN
                        IF {authorized_channel_transition_sql("postgres")} THEN
                            RETURN NEW;
                        END IF;
                        IF EXISTS (
                            SELECT 1 FROM canonical_audience_reassignments ar
                             WHERE ar.canonical_turn_id = OLD.canonical_turn_id
                               AND ar.owner_conversation_id = OLD.conversation_id
                               AND ar.from_audience = OLD.audience_conversation_id
                               AND ar.to_audience = NEW.audience_conversation_id
                               AND ar.from_attribution_version = OLD.audience_attribution_version
                               AND ar.to_attribution_version = NEW.audience_attribution_version
                               AND ar.turn_hash = OLD.turn_hash
                               AND OLD.turn_hash IS NOT DISTINCT FROM NEW.turn_hash
                               AND OLD.conversation_id IS NOT DISTINCT FROM NEW.conversation_id
                               AND OLD.user_content IS NOT DISTINCT FROM NEW.user_content
                               AND OLD.sender_actor_id IS NOT DISTINCT FROM NEW.sender_actor_id
                               AND OLD.origin_channel_id IS NOT DISTINCT FROM NEW.origin_channel_id
                               AND OLD.created_at IS NOT DISTINCT FROM NEW.created_at
                               AND OLD.first_seen_at IS NOT DISTINCT FROM NEW.first_seen_at
                        ) THEN
                            RETURN NEW;
                        END IF;
                    END IF;

                    DELETE FROM actor_card_entries e
                     USING actor_card_turn_sources s
                     WHERE s.canonical_turn_id = OLD.canonical_turn_id
                       AND e.id = s.entry_id
                       AND e.tenant_id = s.tenant_id;

                    IF TG_OP = 'DELETE' THEN
                        RETURN OLD;
                    END IF;
                    RETURN NEW;
                END;
"""
