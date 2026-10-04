"""Statistics, error categorisation and regression checks on known inputs."""

import pytest

from streamrag.evaluation import errors, regression, stats


def test_describe_known_values():
    d = stats.describe([1, 2, 3, 4])
    assert d["mean"] == 2.5 and d["median"] == 2.5 and d["std"] == pytest.approx(1.291, abs=1e-3)
    assert d["ci95"][0] <= 2.5 <= d["ci95"][1]
    assert stats.describe([])["n"] == 0


def test_paired_identical_and_mcnemar():
    a = {f"s{i}": 1.0 for i in range(30)}
    same = stats.paired(a, dict(a))
    assert same["mean_diff"] == 0.0 and same["test"] == "too_few_nonzero_differences"
    b = dict(a)
    for i in range(10):
        b[f"s{i}"] = 0.0
    r = stats.paired(a, b)
    assert r["test"] == "mcnemar_exact" and r["discordant"] == [10, 0] and r["p_value"] < 0.01
    assert r["mean_diff"] == pytest.approx(-1 / 3, abs=1e-4)


def test_wilcoxon_matches_scipy():
    from scipy.stats import wilcoxon
    a = {f"s{i}": float(i) for i in range(25)}
    b = {f"s{i}": float(i) + (1.5 if i % 3 else -0.5) for i in range(25)}
    r = stats.paired(a, b)
    d = [b[k] - a[k] for k in a]
    assert r["test"] == "wilcoxon_signed_rank" and r["p_value"] == pytest.approx(wilcoxon(d).pvalue, abs=1e-6)
    assert 0 < r["rank_biserial"] <= 1


def test_too_few_pairs():
    assert stats.paired({"a": 1.0}, {"a": 0.0})["test"] == "too_few_pairs"


def _row(**m):
    base = {"retrieval": {}, "evidence": {}, "generation": {}, "hallucination": {}, "citation": {}, "latency": {}}
    for k, v in m.items():
        g, key = k.split("__")
        base[g][key] = v
    return {"status": "ok", "error": None, "metrics": base, "adaptive": {}}


def test_error_categories():
    from types import SimpleNamespace as NS
    s = NS(query_type="SIMPLE", ground_truth_evidence=["D §1"], required_constraints=[])
    assert errors.categorize(s, _row(retrieval__recall_at10=None)) == []
    r = _row(**{"retrieval__recall@10": 0.0})
    assert errors.categorize(s, r) == ["RETRIEVAL_FAILURE"]
    fu = NS(query_type="CONTEXTUAL_FOLLOWUP", ground_truth_evidence=["D §1"], required_constraints=[])
    assert errors.categorize(fu, r) == ["MEMORY_FAILURE"]
    gen = _row(**{"retrieval__recall@10": 1.0, "evidence__evidence_coverage": 1.0, "generation__completeness": 0.5})
    assert "GENERATION_FAILURE" in errors.categorize(s, gen)
    slow = _row(**{"latency__ttva_after_end": 9000})
    assert errors.categorize(s, slow, latency_gate_ms=5000) == ["LATENCY_FAILURE"]
    assert errors.categorize(s, {"status": "failed", "error": "x", "metrics": {}}) == ["ORCHESTRATION_FAILURE"]


def test_regression_compare_and_gates():
    rules = {"q": {"direction": "higher", "tolerance": 0.02}, "lat": {"direction": "lower", "tolerance": 10}}
    rows = {r["metric"]: r["status"] for r in regression.compare({"q": 0.80, "lat": 130}, {"q": 0.85, "lat": 100}, rules)}
    assert rows == {"q": "regression", "lat": "regression"}
    rows = {r["metric"]: r["status"] for r in regression.compare({"q": 0.86, "lat": 95}, {"q": 0.85, "lat": 100}, rules)}
    assert rows == {"q": "ok", "lat": "ok"}
    g = regression.gates({"q": 0.9}, {"recall": {"metric": "q", "min": 0.8}, "lat": {"metric": "x", "max": 5}})
    assert [x["status"] for x in g] == ["PASS", "NOT MEASURED"]
