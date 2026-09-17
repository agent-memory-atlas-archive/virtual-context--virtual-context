from pathlib import Path

from benchmarks.jev import cli, labeled
from benchmarks.jev.runtime import FakeJevClient
from virtual_context.core.judgment import JudgmentMode, JudgmentRuntime
from virtual_context.types import JudgmentConfig


def _fake_runtime():
    cfg = JudgmentConfig(mode="jev")
    return JudgmentRuntime(JudgmentMode.JEV, FakeJevClient(cfg, environ={"TYPESAFE_API_KEY": "k"}), cfg)


def test_labeled_areas_run_with_fake_client(tmp_path):
    for area in ("intent", "temporal", "safety"):
        result = labeled.run_area(area, _fake_runtime(), limit=3, dataset_path=None)
        assert result["area"] == area and result["n"] == 3
        for col in ("legacy", "jev_raw", "jev_deployed"):
            assert 0.0 <= result[col]["accuracy_all"] <= 1.0
        assert len(result["rows"]) == 3 and {"id", "label", "legacy", "jev_raw", "jev_deployed"} <= set(result["rows"][0])


def test_cli_writes_a_result_file(tmp_path):
    rc = cli.main(["intent", "--limit", "2", "--fake", "--out", str(tmp_path)])
    assert rc == 0
    files = list(Path(tmp_path).glob("intent-*.json"))
    assert len(files) == 1
