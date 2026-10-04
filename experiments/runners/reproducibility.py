"""Reproducibility check (brief §57-58): re-runs selected variants with the same configuration, seeds and dataset into
a separate directory and compares them with the stored Phase 10 runs (experiments/results/runs/).

Measured per variant (test split): share of turns with an identical answer text, identical evidence citation list and
identical answer-correctness label, and the run-to-run latency difference (median / max absolute difference of
ttva_after_end). Nothing is tuned here; the second run only measures how deterministic the first one was.

Usage: .venv/bin/python experiments/runners/reproducibility.py --index-root /tmp/idx10 [--systems a,b]
       (LLM variants need `ollama serve`)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpora import CORPORA  # noqa: E402
from run_experiments import instrument_factory, llm_info  # noqa: E402

from streamrag.evaluation.runner import ExperimentRunner, Logger  # noqa: E402
from streamrag.evaluation.systems import Stacks  # noqa: E402

DEFAULT = "adaptive_x,naive_rag,adaptive_rag,full_system"


def _scored(path: Path) -> dict[str, dict]:
    return {r["sample_id"]: r for r in (json.loads(x) for x in path.read_text().splitlines())}


def compare(a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = sorted(set(a) & set(b))
    same = lambda f: sum(1 for i in ids if f(a[i]) == f(b[i])) / len(ids) if ids else None  # noqa: E731
    lat = [(a[i]["metrics"]["latency"].get("ttva_after_end"), b[i]["metrics"]["latency"].get("ttva_after_end"))
           for i in ids]
    diffs = [abs(x - y) for x, y in lat if x is not None and y is not None]
    rel = [abs(x - y) / x for x, y in lat if x and y is not None]
    return {"turns": len(ids),
            "identical_answer_text": same(lambda r: r["answer"]),
            "identical_evidence": same(lambda r: r["evidence_citations"]),
            "identical_answer_correct": same(lambda r: r["metrics"]["generation"].get("answer_correct")),
            "identical_status": same(lambda r: r["status"]),
            "answer_correct_mean": {"run1": _mean(a, ids), "run2": _mean(b, ids)},
            "ttva_after_end_abs_diff_ms": {"median": round(statistics.median(diffs), 1) if diffs else None,
                                           "max": round(max(diffs), 1) if diffs else None, "n": len(diffs)},
            "ttva_after_end_rel_diff": {"median": round(statistics.median(rel), 4) if rel else None,
                                        "max": round(max(rel), 4) if rel else None},
            "differing_answers": [i for i in ids if a[i]["answer"] != b[i]["answer"]]}


def _mean(rows, ids):
    v = [rows[i]["metrics"]["generation"].get("answer_correct") for i in ids]
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--systems", default=DEFAULT)
    ap.add_argument("--model", default="qwen3:4b")
    a = ap.parse_args()
    systems = yaml.safe_load((REPO / "experiments" / "configs" / "systems.yaml").read_text())
    from streamrag.generation.llm import OllamaBackend
    url = "http://127.0.0.1:11434"
    llm = info = None
    if OllamaBackend.reachable(url, a.model):
        llm = OllamaBackend(url, a.model, temperature=0.0, seed=7)
        info = llm_info(url, a.model)
    stacks = Stacks(REPO, CORPORA, a.index_root, llm=llm)
    out_dir = REPO / "experiments" / "results" / "REPRODUCIBILITY"
    runner = ExperimentRunner(REPO, systems, stacks, instrument_factory(stacks), llm=llm, results_dir=out_dir,
                              llm_info=info)
    log = Logger(out_dir / "log.jsonl")
    log("start", experiment_id="REPRODUCIBILITY")
    res = {"meta": {"purpose": "second run of stored variants, identical configuration", "reportable": False,
                    "llm": info}, "systems": {}}
    for s in a.systems.split(","):
        if systems[s].get("generation") == "llm" and llm is None:
            res["systems"][s] = {"status": "NOT RUN", "reason": "needs `ollama serve`"}
            continue
        first = REPO / "experiments" / "results" / "runs" / f"{s}__test.scored.jsonl"
        if not first.exists():
            res["systems"][s] = {"status": "NOT RUN", "reason": "no stored first run"}
            continue
        second = runner.ensure_run(s, "test", log)
        res["systems"][s] = {"status": "COMPLETED", **compare(_scored(first), second)}
        print(s, json.dumps({k: v for k, v in res["systems"][s].items() if k != "differing_answers"}), flush=True)
    log("end", status="COMPLETED")
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
