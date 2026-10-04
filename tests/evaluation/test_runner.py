"""ExperimentRunner on a tiny slice (extractive, no LLM): files, schema, FAILED marking, run reuse."""

import json
import sys

import pytest
import yaml

from conftest import REPO, requires_bge
from grounding_helpers import requires_nli

sys.path.insert(0, str(REPO / "experiments" / "runners"))


@requires_bge
@requires_nli
def test_runner_end_to_end(tmp_path):
    from corpora import CORPORA
    from run_experiments import instrument_factory
    from streamrag.evaluation.runner import ExperimentRunner
    from streamrag.evaluation.systems import Stacks
    systems = yaml.safe_load((REPO / "experiments" / "configs" / "systems.yaml").read_text())
    systems["broken"] = {"family": "nonexistent", "generation": "extractive"}
    stacks = Stacks(REPO, CORPORA, tmp_path / "idx")
    r = ExperimentRunner(REPO, systems, stacks, instrument_factory(stacks), results_dir=tmp_path / "res")
    orig = r.samples
    r.samples = lambda split, types=None, ids=None: orig(split, types, ["T01", "S12"])
    res = r.run_experiment({"experiment_id": "T", "splits": ["test"], "systems": ["topk_5", "adaptive_x"],
                            "comparisons": [["topk_5", "adaptive_x"]], "metrics": ["retrieval.recall@5"]})
    assert res["status"] == "COMPLETED"
    d = tmp_path / "res" / "T"
    for f in ("config_record.json", "results.json", "samples.jsonl", "samples.csv", "log.jsonl"):
        assert (d / f).exists()
    rows = [json.loads(x) for x in (d / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and {r["system_variant"] for r in rows} == {"topk_5", "adaptive_x"}
    for row in rows:
        assert {"experiment_id", "sample_id", "system_variant", "query_category", "difficulty", "metrics"} <= set(row)
        assert set(row["metrics"]) >= {"retrieval", "evidence", "claims", "generation", "citation", "latency",
                                       "efficiency"}
    log = [json.loads(x)["event"] for x in (d / "log.jsonl").read_text().splitlines()]
    assert log[0] == "start" and log[-1] == "end" and "configuration" in log
    rec = json.loads((d / "config_record.json").read_text())
    assert rec["git"]["commit"] and rec["dataset_version"]["test"] and rec["embedding_model"]
    again = r.run_experiment({"experiment_id": "T2", "splits": ["test"], "systems": ["topk_5"]})
    assert again["status"] == "COMPLETED"
    assert "run_reused" in (tmp_path / "res" / "T2" / "log.jsonl").read_text()
    bad = r.run_experiment({"experiment_id": "B", "splits": ["test"], "systems": ["broken"]})
    rows = [json.loads(x) for x in (tmp_path / "res" / "B" / "samples.jsonl").read_text().splitlines()]
    assert bad["status"] == "COMPLETED" and all(x["status"] == "failed" for x in rows)   # failed, never dropped
    assert "ORCHESTRATION_FAILURE" in bad["splits"]["test"]["systems"]["broken"]["error_categories"]
