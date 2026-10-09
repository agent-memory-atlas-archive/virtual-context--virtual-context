"""A shadow seam answers from legacy without waiting for Jev."""
from __future__ import annotations

import logging
import threading
import time

import pytest

from virtual_context.core import judgment


@pytest.mark.regression("BUG-120")
def test_shadow_returns_legacy_before_jev_answers(monkeypatch, caplog):
    monkeypatch.setattr(judgment, "_submit_shadow", judgment._submit_shadow_background)
    release = threading.Event()
    called = threading.Event()

    def slow_jev(_client):
        called.set()
        release.wait(5)
        return None

    runtime = type("Runtime", (), {
        "enabled_for": lambda self, seam: True,
        "mode_for": lambda self, seam: judgment.JudgmentMode.SHADOW,
        "client": object(),
    })()
    with caplog.at_level(logging.INFO, logger=judgment.logger.name):
        started = time.monotonic()
        value = judgment.decide("temporal_intent", lambda: "legacy", slow_jev, runtime=runtime)
        elapsed = time.monotonic() - started
        assert value == "legacy"
        assert elapsed < 1.0
        assert called.wait(2)
        release.set()
        deadline = time.monotonic() + 3
        while "JUDGMENT_SHADOW seam=temporal_intent" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.02)
    assert "JUDGMENT_SHADOW seam=temporal_intent agree=None legacy='legacy'" in caplog.text
