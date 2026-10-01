"""Facts record the turns they were extracted from.

Every fact pointed back to its segment, but ``turn_numbers`` stayed empty: the
compaction pipeline knew each segment's exact turns and never passed them on,
so no fact could be traced to the turns it came from. A fact now carries its
segment's turns unless extraction already gave it narrower ones.
"""

from __future__ import annotations

import pytest

from virtual_context.core.compaction_pipeline import stamp_fact_provenance
from virtual_context.types import Fact


@pytest.mark.regression("BUG-107")
def test_facts_carry_their_segments_turns():
    facts = [Fact(subject="a", verb="likes", object="b"), Fact(subject="c", verb="ran", object="d")]
    stamp_fact_provenance(facts, segment_ref="seg-1", conversation_id="conv", turn_numbers=[41, 42, 43])
    for fact in facts:
        assert fact.segment_ref == "seg-1"
        assert fact.conversation_id == "conv"
        assert fact.turn_numbers == [41, 42, 43]
    facts[0].turn_numbers.append(99)
    assert facts[1].turn_numbers == [41, 42, 43]


@pytest.mark.regression("BUG-107")
def test_narrower_turns_from_extraction_are_kept():
    fact = Fact(subject="a", verb="likes", object="b", turn_numbers=[42])
    stamp_fact_provenance([fact], segment_ref="seg-1", conversation_id="conv", turn_numbers=[41, 42, 43])
    assert fact.turn_numbers == [42]
