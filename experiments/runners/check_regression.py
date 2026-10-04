"""Regression check + quality gates for a new evaluation run (docs/evaluation/regression.md).

Compares the per-system aggregates of a results.json (from run_experiments.py, usually written to a separate
results directory) with experiments/results/regression_baseline.json using experiments/configs/regression_rules.yaml,
then evaluates experiments/configs/quality_gates.yaml on the gated system. Reports ok / improvement / regression /
missing and PASS / FAIL / NOT MEASURED; exit code 1 only with --strict and a FAIL gate or a regression.

Usage: .venv/bin/python experiments/runners/check_regression.py --results <dir>/EXP01_baselines/results.json [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from streamrag.evaluation import regression as REG  # noqa: E402

CONF = REPO / "experiments" / "configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=REPO / "experiments" / "results" / "regression_baseline.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    current = REG.snapshot(json.loads(a.results.read_text()), a.split)
    base = json.loads(a.baseline.read_text())["systems"]
    rules = yaml.safe_load((CONF / "regression_rules.yaml").read_text())
    gcfg = yaml.safe_load((CONF / "quality_gates.yaml").read_text())
    report = {"results": str(a.results), "baseline": str(a.baseline), "regression": {}, "gates": {}}
    bad = False
    for system, vals in current.items():
        if system not in base:
            report["regression"][system] = "no baseline for this variant"
            continue
        rows = REG.compare(vals, base[system], rules["metrics"])
        report["regression"][system] = rows
        bad |= any(r["status"] == "regression" for r in rows)
        for r in rows:
            if r["status"] != "ok":
                print(f"{system:28s} {r['metric']:36s} {r['status']:12s} baseline={r['baseline']} "
                      f"current={r['current']}")
    target = gcfg.get("system", "full_system")
    if target in current:
        g = REG.gates(current[target], gcfg["gates"])
        report["gates"] = {"system": target, "results": g}
        bad |= any(x["status"] == "FAIL" for x in g)
        for x in g:
            print(f"gate {x['gate']:32s} {x['status']:13s} value={x['value']} threshold={x.get('threshold')}")
    else:
        report["gates"] = {"system": target, "status": "NOT MEASURED (variant not in these results)"}
        print(f"gates: {target} not in results - NOT MEASURED")
    out = a.out or a.results.parent / "regression_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("report:", out)
    if a.strict and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
