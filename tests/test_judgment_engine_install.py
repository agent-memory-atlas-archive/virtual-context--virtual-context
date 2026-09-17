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


def _engine(tmp_path, monkeypatch, mode="legacy"):
    cfg = VirtualContextConfig(storage_root=str(tmp_path / "vc"))
    cfg.judgment = JudgmentConfig(mode=mode)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    return VirtualContextEngine(config=cfg)


def test_engine_installs_legacy_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VC_JUDGMENT_MODE", raising=False)
    _engine(tmp_path, monkeypatch)
    assert judgment.current().mode is JudgmentMode.LEGACY


def test_engine_installs_configured_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("VC_JUDGMENT_MODE", raising=False)
    _engine(tmp_path, monkeypatch, mode="shadow")
    assert judgment.current().mode is JudgmentMode.SHADOW


def test_env_set_before_construction_wins_and_is_pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("VC_JUDGMENT_MODE", "jev")
    _engine(tmp_path, monkeypatch, mode="legacy")
    assert judgment.current().mode is JudgmentMode.JEV
    monkeypatch.setenv("VC_JUDGMENT_MODE", "legacy")
    assert judgment.current().mode is JudgmentMode.JEV  # pinned: later env changes do nothing


def test_invalid_env_fails_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("VC_JUDGMENT_MODE", "maybe")
    with pytest.raises(ValueError):
        _engine(tmp_path, monkeypatch)
