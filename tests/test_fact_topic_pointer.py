"""A fact shown to the model names the topic it can open for the source.

Facts are cached copies of what was said; the conversation is the authority.
Each fact stores the segment it came from, but its rendered line carried no
reference, so the model saw the value without knowing where to page in the
source. The line now names the fact's topic, which ``vc_expand_topic`` opens.
"""

from __future__ import annotations

import pytest

from virtual_context.types import Fact


@pytest.mark.regression("BUG-108")
def test_a_fact_line_names_its_topic():
    fact = Fact(subject="Kuw", verb="has", object="a 365 lb deadlift training max",
                tags=["deadlift-programming", "training-max"])
    line = fact.format_for_prompt()
    assert "[topic: deadlift-programming]" in line
    assert "training-max" not in line


@pytest.mark.regression("BUG-108")
def test_a_fact_without_tags_renders_as_before():
    fact = Fact(subject="Kuw", verb="has", object="a 365 lb deadlift training max")
    assert "[topic:" not in fact.format_for_prompt()
