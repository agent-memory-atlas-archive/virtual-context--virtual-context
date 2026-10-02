"""Each LLM component uses its own configured model.

The tagger, supersession checker and fact curator each have a ``model``
setting, but providers were built with the summarization model unless the
provider block named one, so ``tag_generator.model`` had no effect.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest


@pytest.mark.regression("BUG-113")
def test_the_tagger_runs_on_tag_generator_model(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "providers": {"local": {"type": "generic_openai", "base_url": "http://127.0.0.1:1/v1"}},
        "tag_generator": {"type": "llm", "provider": "local", "model": "tagger-model"},
        "summarization": {"provider": "local", "model": "summary-model"},
        "supersession": {"enabled": True, "model": "supersession-model"},
        "curation": {"enabled": True, "model": "curation-model"},
        "conversation_id": "conv-models",
    }))
    assert engine._tag_generator.llm.model == "tagger-model"
    assert engine._llm_provider.model == "summary-model"
    assert engine._supersession_checker.llm.model == "supersession-model"
    assert engine._fact_curator.llm.model == "curation-model"
