"""Each engine owns its judgment runtime; nothing is installed process-wide."""
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import JudgmentMode
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import JudgmentConfig, VirtualContextConfig


@pytest.fixture(autouse=True)
def _reset_runtime():
    judgment.reset()
    yield
    judgment.reset()


def _engine(tmp_path, monkeypatch, mode="legacy", seams=None, name="a"):
    cfg = VirtualContextConfig(storage_root=str(tmp_path / f"vc-{name}"))
    cfg.judgment = JudgmentConfig(mode=mode, seams=dict(seams or {}))
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    return VirtualContextEngine(config=cfg)


def test_engine_default_runtime_is_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("VC_JUDGMENT_MODE", raising=False)
    eng = _engine(tmp_path, monkeypatch)
    assert eng.judgment_runtime.mode is JudgmentMode.LEGACY and eng.judgment_runtime.client is None


def test_engine_runtime_follows_its_config(tmp_path, monkeypatch):
    monkeypatch.delenv("VC_JUDGMENT_MODE", raising=False)
    eng = _engine(tmp_path, monkeypatch, mode="shadow", seams={"admission": "legacy"})
    assert eng.judgment_runtime.mode is JudgmentMode.SHADOW
    assert eng.judgment_runtime.mode_for("admission") is JudgmentMode.LEGACY
    assert eng.judgment_runtime.mode_for("rerank") is JudgmentMode.SHADOW


def test_env_set_before_construction_wins_and_is_pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("VC_JUDGMENT_MODE", "jev")
    eng = _engine(tmp_path, monkeypatch, mode="legacy")
    assert eng.judgment_runtime.mode is JudgmentMode.JEV
    monkeypatch.setenv("VC_JUDGMENT_MODE", "legacy")
    assert eng.judgment_runtime.mode is JudgmentMode.JEV


def test_invalid_env_fails_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("VC_JUDGMENT_MODE", "maybe")
    with pytest.raises(ValueError):
        _engine(tmp_path, monkeypatch)


def test_two_engines_keep_separate_runtimes_and_never_touch_the_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("VC_JUDGMENT_MODE", raising=False)
    shadow = _engine(tmp_path, monkeypatch, mode="shadow", name="shadow")
    legacy = _engine(tmp_path, monkeypatch, mode="legacy", name="legacy")
    assert shadow.judgment_runtime.mode is JudgmentMode.SHADOW
    assert legacy.judgment_runtime.mode is JudgmentMode.LEGACY
    assert judgment.current().mode is JudgmentMode.LEGACY  # constructing engines never installs globally
    assert shadow.judgment_runtime is not legacy.judgment_runtime


def test_engine_hands_its_runtime_to_every_seam_host(tmp_path, monkeypatch):
    monkeypatch.delenv("VC_JUDGMENT_MODE", raising=False)
    eng = _engine(tmp_path, monkeypatch, mode="shadow")
    rt = eng.judgment_runtime
    assert eng._retriever.judgment_runtime is rt
    assert eng._assembler.judgment_runtime is rt
    assert eng._search._judgment_runtime is rt
    assert eng._temporal._judgment_runtime is rt
    assert eng._compaction._judgment_runtime is rt
    assert eng._compaction._actor_card_admission_service()._judgment_runtime is rt
    if eng._compactor is not None:
        assert eng._compactor.judgment_runtime is rt
