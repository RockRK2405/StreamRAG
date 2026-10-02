import json

import pytest

from streamrag.bench import load_eval_items, mrr_at_k, ndcg_at_k, recall_at_k, success_at_k
from streamrag.bench.harness import run_benchmark
from streamrag.models import BenchmarkCase

from conftest import FIX


def test_metric_math():
    keys = ["a", "x", "b", "y"]
    gold = {"a", "b", "c"}
    assert recall_at_k(keys, gold, 1) == pytest.approx(1 / 3)
    assert recall_at_k(keys, gold, 3) == pytest.approx(2 / 3)
    assert success_at_k(keys, gold, 1) == 1.0 and success_at_k(["x"], gold, 1) == 0.0
    assert mrr_at_k(["x", "b"], gold, 10) == 0.5 and mrr_at_k(["x"], gold, 10) == 0.0
    ideal = 1 + 1 / 1.5849625007211562 + 0.5
    assert ndcg_at_k(keys, gold, 4) == pytest.approx((1 + 0.5) / ideal)
    assert ndcg_at_k(["a", "a"], {"a"}, 2) == pytest.approx(1.0)        # gold credited once


def test_harness_on_fixture_is_not_reportable(fixture_bundle, tmp_path):
    cfg, b = fixture_bundle
    m = run_benchmark(cfg, FIX / "eval" / "fixture_retrieval.jsonl", ["bm25", "hybrid"], tmp_path / "run",
                      index_path=b.path)
    assert m["REPORTABLE"] is False and "TEST FIXTURE" in m["banner"]
    for f in ("metrics.json", "per_query.jsonl", "per_query.csv", "run_manifest.json"):
        assert (tmp_path / "run" / f).exists()
    man = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert man["REPORTABLE"] is False and man["corpus_is_test_fixture"] and man["eval_data"]["n_items"] == 6
    assert set(m["modes"]["hybrid"]["metrics"]) >= {"recall@1", "recall@5", "recall@10", "mrr@10"}
    assert "p95" in m["modes"]["hybrid"]["latency_ms"]["total"]


def test_benchmark_case_directory_flattening(tmp_path):
    case = {"case_id": "mi-1", "category": "multi_intent", "split": "test", "is_fixture": True,
            "session": {"session_id": "s", "turns": [{"utterance_id": "u1", "utterance_text": "two needs",
                        "utterance_end_s": 2.0, "expected": {"turn_type": "query", "retrieval_required": True, "intents": [
                            {"gold_intent_id": "g1", "description": "first need", "gold_evidence": ["D §1"]},
                            {"gold_intent_id": "g2", "description": "second need", "gold_evidence": ["D §2"]},
                            {"gold_intent_id": "g3", "description": "unanswerable", "answerable": False}]}}]}}
    (tmp_path / "c.json").write_text(json.dumps(case))
    BenchmarkCase.model_validate(case)
    items = load_eval_items(tmp_path)
    assert [(i.intent_id, i.gold) for i in items] == [("g1", ["D §1"]), ("g2", ["D §2"])]


def test_duplicate_item_ids_rejected(tmp_path):
    line = json.dumps({"item_id": "x", "query": "q", "gold": ["D §1"]})
    (tmp_path / "e.jsonl").write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError):
        load_eval_items(tmp_path / "e.jsonl")
