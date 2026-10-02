"""Paging tools reach native Gemini requests and current model families.

A native Gemini request names its model in the URL path, not the body, so the
proxy resolved an empty model name and never offered the paging tools. The
default autonomous-model list also predated the current Claude and GPT
families.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.proxy.server import model_from_path


@pytest.mark.regression("BUG-114")
@pytest.mark.parametrize("path, model", [
    ("v1beta/models/gemini-2.5-pro:generateContent", "gemini-2.5-pro"),
    ("v1beta/models/gemini-3-flash:streamGenerateContent", "gemini-3-flash"),
    ("v1/messages", ""),
    ("v1/chat/completions", ""),
])
def test_model_from_path(path, model):
    assert model_from_path(path) == model


@pytest.mark.regression("BUG-114")
@pytest.mark.parametrize("model", [
    "gemini-2.5-pro", "claude-opus-5-5", "claude-sonnet-5", "claude-fable-5-1",
    "claude-haiku-4-5-20251001", "gpt-6-sol",
])
def test_current_models_page_autonomously(tmp_path, model):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    engine = VirtualContextEngine(config=load_config(config_dict={
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tag_generator": {"type": "keyword"},
        "paging": {"enabled": True},
        "conversation_id": "conv-paging",
    }))
    assert engine._retrieval._resolve_paging_mode(model) == "autonomous"
