"""The context hint asks for a search only when the answer needs stored detail."""
from __future__ import annotations

import pytest

from virtual_context.core.hint_builder import build_autonomous_hint, build_supervised_hint
from virtual_context.types import TagSummary


def _summaries():
    return [TagSummary(tag="leg-press", summary="Leg press progression at 640 lb.")]


@pytest.mark.regression("BUG-121")
def test_hints_do_not_require_a_search_before_every_answer():
    count = lambda text: len(text) // 4
    supervised = build_supervised_hint(_summaries(), {}, 5000, count)
    compact = build_autonomous_hint(_summaries(), {}, 1000, 60, count)
    for hint in (supervised, compact, build_autonomous_hint(_summaries(), {}, 1000, 5000, count)):
        assert "Never answer without searching first" not in hint
    assert "greetings" in supervised


@pytest.mark.regression("BUG-121")
def test_a_change_to_the_hint_text_invalidates_stored_hints(tmp_path, monkeypatch):
    from virtual_context.core import hint_builder
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import StorageConfig, VirtualContextConfig

    engine = VirtualContextEngine(config=VirtualContextConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "t.db")),
    ))
    before = engine._retrieval._build_context_hint_cache_key("supervised")
    monkeypatch.setattr(hint_builder, "TEMPLATE_FINGERPRINT", "changed-template")
    assert engine._retrieval._build_context_hint_cache_key("supervised") != before
