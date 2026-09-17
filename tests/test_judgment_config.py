import pytest

from virtual_context.config import _build_config
from virtual_context.types import JUDGMENT_MODES, JudgmentConfig, VirtualContextConfig


def test_judgment_defaults_are_dormant():
    cfg = JudgmentConfig()
    assert cfg.mode == "legacy"
    assert cfg.model == "jev-latest"
    assert cfg.api_key_env == "TYPESAFE_API_KEY"
    assert cfg.base_url == "https://api.typesafe.ai/v1/systemone"
    assert cfg.timeout_s == 3.0
    assert cfg.noul_threshold == 0.5
    assert cfg.rerank_min_probability == 0.0
    assert cfg.rerank_max_state_bytes == 200_000
    assert VirtualContextConfig().judgment == JudgmentConfig()
    assert JUDGMENT_MODES == ("legacy", "shadow", "jev")


def test_judgment_yaml_block_is_parsed():
    cfg = _build_config({
        "judgment": {
            "mode": "shadow",
            "model": "jev-2",
            "timeout_s": 1.5,
            "noul_threshold": 0.6,
            "rerank_min_probability": 0.2,
            "rerank_max_state_bytes": 1000,
        }
    }, validate=False)
    assert cfg.judgment.mode == "shadow"
    assert cfg.judgment.model == "jev-2"
    assert cfg.judgment.timeout_s == 1.5
    assert cfg.judgment.noul_threshold == 0.6
    assert cfg.judgment.rerank_min_probability == 0.2
    assert cfg.judgment.rerank_max_state_bytes == 1000


def test_judgment_missing_block_uses_defaults():
    cfg = _build_config({}, validate=False)
    assert cfg.judgment == JudgmentConfig()


@pytest.mark.parametrize("bad", ["", "on", "JEV ", "hybrid"])
def test_judgment_invalid_mode_raises(bad):
    with pytest.raises(ValueError, match="judgment.mode"):
        _build_config({"judgment": {"mode": bad}}, validate=False)
