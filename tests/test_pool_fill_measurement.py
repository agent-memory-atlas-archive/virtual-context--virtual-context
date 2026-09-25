"""Pool fill measures the facts block without re-tokenizing it per candidate.

The fill charged each candidate fact by tokenizing the whole prospective
block, so a turn with a few hundred candidate facts tokenized the growing
block hundreds of times. With a tokenizer whose count is additive across
lines the block count is the sum of its parts; the fill uses that sum and
confirms it once, falling back to whole-block measurement when it differs.
"""

from __future__ import annotations

import random

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.token_counter import create_token_counter


def _lines(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    words = ["deadlift", "235", "lb", "x5", "BigTex", "finasteride", "1 mg", "daily",
             "ApoB", "105", "mg/dL", "retatrutide", "24.2%", "at", "48 weeks", "—", "|"]
    return [
        "- " + " ".join(rng.choice(words) for _ in range(rng.randint(4, 40)))
        for _ in range(n)
    ]


def _inputs(counter, n: int, seed: int):
    lines = _lines(n, seed)
    fact_lines = dict(enumerate(lines))
    fact_tokens = {i: counter(line + "\n") for i, line in fact_lines.items()}
    rng = random.Random(seed + 1)
    scored = [(rng.random(), "fact", str(i), fact_tokens[i]) for i in fact_lines]
    scored.append((0.99, "tag", "t1", 40))
    scored.sort(key=lambda item: item[0], reverse=True)
    dense_order = list(range(0, n, 3))
    return scored, fact_lines, fact_tokens, {"t1": "section"}, dense_order


def _kwargs(counter, n, seed, *, facts_cap, dense):
    scored, fact_lines, fact_tokens, sections, dense_order = _inputs(counter, n, seed)
    return scored, dict(
        tag_cap=1000, retrieved_cap=1000, pool=facts_cap + 2000,
        facts_cap=facts_cap, working_set=None, retrieval_scores={"t1": 1.0},
        built_sections=sections, fact_lines=fact_lines, fact_tokens=fact_tokens,
        dense_order=dense_order if dense else None,
    )


@pytest.mark.regression("BUG-095")
@pytest.mark.parametrize("counter_mode", ["tiktoken", "estimate"])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("n,facts_cap", [(1, 50), (40, 400), (300, 3000), (300, 100000)])
def test_the_fill_matches_whole_block_measurement(counter_mode, dense, n, facts_cap):
    counter = create_token_counter(counter_mode)
    assembler = ContextAssembler.__new__(ContextAssembler)
    assembler.token_counter = counter
    scored, kwargs = _kwargs(counter, n, 7, facts_cap=facts_cap, dense=dense)
    exact = assembler._fill_pool(scored, exact=True, **kwargs)
    assert assembler._fill_pool_measured(scored, **kwargs) == exact


@pytest.mark.regression("BUG-095")
def test_an_additive_counter_is_called_far_less():
    base = create_token_counter("tiktoken")
    calls = {"n": 0}

    def counting(text):
        calls["n"] += 1
        return base(text)

    assembler = ContextAssembler.__new__(ContextAssembler)
    assembler.token_counter = counting
    scored, fact_lines, fact_tokens, sections, _dense = _inputs(base, 300, 3)
    kwargs = dict(
        tag_cap=1000, retrieved_cap=1000, pool=100000, facts_cap=100000,
        working_set=None, retrieval_scores={"t1": 1.0}, built_sections=sections,
        fact_lines=fact_lines, fact_tokens=fact_tokens, dense_order=None,
    )
    calls["n"] = 0
    assembler._fill_pool(scored, exact=True, **kwargs)
    exact_calls = calls["n"]
    calls["n"] = 0
    assembler._fill_pool_measured(scored, **kwargs)
    assert exact_calls >= 300
    assert calls["n"] <= 5
