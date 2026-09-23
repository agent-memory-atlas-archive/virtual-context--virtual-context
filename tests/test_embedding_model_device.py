"""The shared embedding model runs on the CPU.

Encoding is not serialized: request, tagging and compaction threads share
one model. On Apple silicon sentence-transformers otherwise picks the MPS
device, where concurrent encodes crash the process with a segfault inside
torch's MPS backend.
"""

from __future__ import annotations

import sys
import types

import pytest

from virtual_context.core import embedding_provider


@pytest.mark.regression("BUG-087")
def test_the_model_is_loaded_on_the_cpu(monkeypatch):
    seen = {}

    class _Model:
        def __init__(self, name, **kwargs):
            seen["name"], seen["kwargs"] = name, kwargs

        def encode(self, texts, **kwargs):
            import numpy as np
            return np.zeros((len(texts), 2))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_Model))
    embed = embedding_provider._load_sentence_transformer("all-MiniLM-L6-v2")
    assert embed(["a"]) == [[0.0, 0.0]]
    assert seen["kwargs"].get("device") == "cpu"
