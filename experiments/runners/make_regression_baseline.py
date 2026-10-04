"""Refreshes experiments/results/regression_baseline.json from stored experiment results (after an accepted change).

The reference holds the per-system aggregate means (and p95 for latencies) of the listed experiments' test split.

Usage: .venv/bin/python experiments/runners/make_regression_baseline.py [--experiments EXP01_baselines,ABLATION_runtime]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from streamrag.evaluation import regression as REG  # noqa: E402
from streamrag.evaluation.runner import git_commit  # noqa: E402

RES = REPO / "experiments" / "results"


def build(experiments: list[str], split: str = "test") -> dict | None:
    snap: dict = {}
    used = []
    for exp in experiments:
        p = RES / exp / "results.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        if res.get("status") != "COMPLETED":
            continue
        used.append(exp)
        for system, vals in REG.snapshot(res, split).items():
            snap.setdefault(system, vals)          # a variant shared by several experiments is one stored run
    if not snap:
        return None
    out = {"source": used, "split": split, "git": git_commit(REPO), "systems": snap}
    (RES / "regression_baseline.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="EXP01_baselines,ABLATION_runtime,ABLATION_answer_stage,EXP02_adaptive_topk")
    a = ap.parse_args()
    out = build(a.experiments.split(","))
    print("no completed experiment" if out is None else f"{len(out['systems'])} systems from {out['source']}")


if __name__ == "__main__":
    main()
