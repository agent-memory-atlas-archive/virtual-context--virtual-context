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


from benchmarks.jev.admission import load_sets, run_admission


def test_admission_sets_load_and_offline_run_scores_jev_only():
    sets = load_sets()
    assert len(sets) >= 6
    for s in sets:
        ids = {c["candidate_id"] for c in s["candidates"]}
        assert set(s["expected"]["decisions"]) == ids
        assert s["expected"]["coverage_reason"] in ("substantive", "greeting_only", "one_off_trivia", "bot_meta_or_test", "no_durable_context", "insufficient_evidence")
    result = run_admission(_fake_runtime(), limit=2, offline=True)
    assert result["n_sets"] == 2 and result["legacy"]["reason_accuracy"] is None
    assert 0.0 <= result["jev"]["reason_accuracy"] <= 1.0


import sqlite3

from benchmarks.jev.rerank import first_gold_rank, gold_segment_refs, normalize_snippet


def test_first_gold_rank():
    assert first_gold_rank(["a", "b", "c"], {"c"}) == 3
    assert first_gold_rank(["a", "b"], {"z"}) is None


def test_gold_segment_refs_matches_on_normalized_turn_text(tmp_path):
    db = tmp_path / "store.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE segments (ref TEXT, full_text TEXT)")
    conn.execute("INSERT INTO segments VALUES ('seg1', 'user:   I ran the   Chicago marathon in 2024 and loved it. assistant: great')")
    conn.execute("INSERT INTO segments VALUES ('seg2', 'user: unrelated text')")
    conn.commit(); conn.close()
    gold = [[{"role": "user", "content": "I ran the Chicago marathon in 2024 and loved it."}]]
    assert gold_segment_refs(db, gold) == {"seg1"}
    assert normalize_snippet("  a   b\nc ") == "a b c"
