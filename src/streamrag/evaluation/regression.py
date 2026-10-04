"""Evaluation regression checks and quality gates (Phase 10; docs/evaluation/regression.md).

* ``snapshot(results)``: flattens an experiment's per-system aggregate means into ``{system: {metric: value}}`` -
  the reference a later run is compared with (experiments/results/regression_baseline.json).
* ``compare(current, baseline, rules)``: per metric, direction-aware (``higher`` / ``lower`` is better) with an
  absolute tolerance and / or a relative one (fraction of the baseline value; the larger applies); returns ok /
  improvement / regression / missing rows. It never rejects a change by itself - it reports, and the gates decide
  what blocks.
* ``gates(values, config)``: project-specific thresholds (experiments/configs/quality_gates.yaml). The thresholds are
  derived from the Phase 10 fixture measurements (stated in the file) - they are not universal targets and must be
  re-derived on the official corpus.
"""

from __future__ import annotations


def snapshot(results: dict, split: str = "test") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system, agg in results["splits"].get(split, {}).get("systems", {}).items():
        flat: dict[str, float] = {}
        for g, metrics in agg.items():
            if not isinstance(metrics, dict) or g in ("by_category", "by_difficulty", "adaptivity",
                                                      "error_categories"):
                continue
            for k, v in metrics.items():
                if isinstance(v, dict):
                    val = v.get("mean") if v.get("mean") is not None else v.get("p50")
                    if val is not None:
                        flat[f"{g}.{k}"] = val
                    if v.get("p95") is not None:
                        flat[f"{g}.{k}.p95"] = v["p95"]
        if agg.get("failure_rate") is not None:
            flat["failure_rate"] = agg["failure_rate"]
        out[system] = flat
    return out


def compare(current: dict[str, float], baseline: dict[str, float], rules: dict[str, dict]) -> list[dict]:
    rows = []
    for metric, rule in rules.items():
        b, c = baseline.get(metric), current.get(metric)
        if b is None or c is None:
            rows.append({"metric": metric, "baseline": b, "current": c, "status": "missing"})
            continue
        delta = c - b
        tol = max(float(rule.get("tolerance", 0.0)), float(rule.get("relative_tolerance", 0.0)) * abs(b))
        better = delta > tol if rule["direction"] == "higher" else delta < -tol
        worse = delta < -tol if rule["direction"] == "higher" else delta > tol
        rows.append({"metric": metric, "baseline": b, "current": c, "delta": round(delta, 4), "tolerance": tol,
                     "status": "regression" if worse else "improvement" if better else "ok"})
    return rows


def gates(values: dict[str, float], config: dict) -> list[dict]:
    out = []
    for name, g in config.items():
        v = values.get(g["metric"])
        if v is None:
            out.append({"gate": name, "metric": g["metric"], "value": None, "status": "NOT MEASURED"})
            continue
        ok = v >= g["min"] if "min" in g else v <= g["max"]
        out.append({"gate": name, "metric": g["metric"], "value": v, "threshold": g.get("min", g.get("max")),
                    "kind": "min" if "min" in g else "max", "status": "PASS" if ok else "FAIL",
                    "basis": g.get("basis", "")})
    return out
