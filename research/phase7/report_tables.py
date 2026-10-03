"""Markdown tables for PHASE_7_GROUNDED_GENERATION_REPORT.md, generated from the result files (nothing typed by hand).

Usage: .venv/bin/python research/phase7/report_tables.py  ->  research/phase7/results/report_tables.md
Reads research/phase7/results_first_run/ (first run, kept unchanged) and research/phase7/results/ (final run). The
intermediate post-fix run (research/phase7/results_postfix_run/, where the hand labels were made) is not tabulated.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {"first run": HERE / "results_first_run", "final run": HERE / "results"}
ARMS = ["A", "B", "C", "D", "E", "F", "X"]


def load(run: Path, name: str):
    p = run / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, pct=False, nd=3):
    if x is None:
        return "–"
    if isinstance(x, dict):                       # percentile dict
        return f"{x['p50']:.0f} / {x['p95']:.0f}" if x.get("n") else "–"
    if isinstance(x, float):
        if pct:
            return f"{x * 100:.1f} %"
        return str(int(x)) if x.is_integer() else f"{x:.{nd}f}"
    return str(x)


def arm_table(s: dict) -> list[str]:
    cols = [("gold_required_fact_recall", "required-fact recall", True),
            ("raw_claims", "raw claims", False),
            ("raw_claim_support_rate_verifier", "raw support (verifier)", True),
            ("raw_claim_unsupported_rate_verifier", "raw unsupported (verifier)", True),
            ("final_claims", "final claims", False),
            ("final_unsupported_rate_verifier", "final unsupported (verifier)", True),
            ("citation_coverage", "citation coverage", True),
            ("citation_precision_verifier", "citation precision (verifier)", True),
            ("intent_coverage", "intent coverage", True),
            ("forbidden_assertions", "forbidden assertions", False),
            ("uncertainty_expressed_when_expected", "uncertainty when expected", True),
            ("conflict_presented_rate", "conflict presented", True),
            ("llm_calls", "LLM calls", False),
            ("generation_ms", "generation ms p50 / p95", False),
            ("validation_ms", "validation ms p50 / p95", False),
            ("latency_ms", "total ms p50 / p95", False)]
    out = ["| metric | " + " | ".join(a for a in ARMS if a in s) + " |",
           "|---|" + "---|" * len([a for a in ARMS if a in s])]
    for key, label, pct in cols:
        out.append(f"| {label} | " + " | ".join(f(s[a].get(key), pct) for a in ARMS if a in s) + " |")
    return out


def category_table(cat: dict, keys=("gold_required_fact_recall", "raw_claim_support_rate_verifier",
                                    "final_unsupported_rate_verifier", "uncertainty_expressed_when_expected",
                                    "conflict_presented_rate", "forbidden_assertions")) -> list[str]:
    out = ["| category | turns | " + " | ".join(k.replace("_verifier", "").replace("_", " ") for k in keys) + " |",
           "|---|---|" + "---|" * len(keys)]
    for c, arms in cat.items():
        F = arms.get("F", {})
        out.append(f"| {c} | {F.get('turns', '–')} | " + " | ".join(
            f(F.get(k), k != "forbidden_assertions") for k in keys) + " |")
    return out


def main() -> None:
    md = ["# Phase 7 report tables (generated from research/phase7/results*/; NOT REPORTABLE, fixture dev suite)", ""]
    for name, run in RUNS.items():
        s = load(run, "summary.json")
        if not s:
            continue
        md += [f"## Ablation by arm: {name}", ""] + arm_table(s["by_arm"]) + [""]
        md += [f"## Full system (F) by category: {name}", ""] + category_table(s["by_category"]) + [""]
        h = load(run, "hallucination.json")
        if h:
            md += [f"## Hallucination tags: {name}", "",
                   "| tag | turns | arm A forbidden / raw unsupported | arm C forbidden / final unsupported | "
                   "arm F forbidden / final unsupported | F uncertainty when expected |", "|---|---|---|---|---|---|"]
            for tag, arms in h["by_tag"].items():
                A, C, F = arms.get("A", {}), arms.get("C", {}), arms.get("F", {})
                md.append(f"| {tag} | {F.get('turns', '–')} | {A.get('forbidden_assertions', '–')} / "
                          f"{f(A.get('raw_claim_unsupported_rate_verifier'), True)} | "
                          f"{C.get('forbidden_assertions', '–')} / {f(C.get('final_unsupported_rate_verifier'), True)}"
                          f" | {F.get('forbidden_assertions', '–')} / "
                          f"{f(F.get('final_unsupported_rate_verifier'), True)} | "
                          f"{f(F.get('uncertainty_expressed_when_expected'), True)} |")
            md.append("")
        r = load(run, "revision.json")
        if r:
            md += [f"## Revision: {name}", "", "| mode | later turns | LLM calls | claims kept | required recall | "
                   "forbidden | wall ms p50 / p95 |", "|---|---|---|---|---|---|---|"]
            for mode, v in r["summary"].items():
                md.append(f"| {mode} | {v['later_turns']} | {v['llm_calls']} | "
                          f"{f(v['claims_kept_from_previous_version'], True)} | "
                          f"{f(v['revision_accuracy_required_recall'], True)} | {v['forbidden_after_revision']} | "
                          f"{f(v['wall_ms'])} |")
            md.append("")
        lat = load(run, "latency.json")
        if lat:
            md += [f"## F stage latency (ms): {name}", "", "| stage | n | p50 | p95 | max |", "|---|---|---|---|---|"]
            for st, v in lat["F_stages_ms"].items():
                md.append(f"| {st} | {v['n']} | {v['p50']:.1f} | {v['p95']:.1f} | {v['max']:.1f} |")
            t = lat["F_ttft_raw_ms"]
            md += [f"| raw first token | {t['n']} | {t['p50']:.1f} | {t['p95']:.1f} | {t['max']:.1f} |", "",
                   f"Measured tokens (F, all new versions): prompt {lat['F_tokens']['prompt']}, "
                   f"output {lat['F_tokens']['output']}.", ""]
        v = load(run, "verifier_eval.json")
        if v:
            md += [f"## Verifier on perturbations (labels by construction): {name}", "",
                   "| verifier | items | accuracy | support precision | support recall | unsupported detected | "
                   "ms / claim |", "|---|---|---|---|---|---|---|"]
            for k in ("nli", "rules"):
                x = v[k]
                md.append(f"| {k} | {x['items']} | {f(x['accuracy'], True)} | {f(x['support_precision'], True)} | "
                          f"{f(x['support_recall'], True)} | {f(x['unsupported_detection_recall'], True)} | "
                          f"{x['ms_per_claim']:.1f} |")
            md += ["", "| perturbation | n | nli accepted | rules accepted |", "|---|---|---|---|"]
            for kind, x in v["nli"]["by_kind"].items():
                md.append(f"| {kind} | {x['n']} | {x['accepted_as_supported']} | "
                          f"{v['rules']['by_kind'][kind]['accepted_as_supported']} |")
            md.append("")
    la = load(HERE / "results", "label_agreement.json")
    if la:
        md += ["## Verifier vs hand labels (final run claims; labels made blind on the post-fix run sheet)", "",
               "| source | n | accuracy | support precision | support recall | unsupported detected |",
               "|---|---|---|---|---|---|"]
        for src, x in [("pooled", la["pooled"])] + list(la["by_source"].items()):
            md.append(f"| {src} | {x['n']} | {f(x['accuracy'], True)} | {f(x['support_precision'], True)} | "
                      f"{f(x['support_recall'], True)} | {f(x['unsupported_detection_recall'], True)} |")
        md += ["", f"Released-claim precision against the labels: {f(la['released_claim_precision_vs_labels'], True)}"
               f" ({la['labelled']} of {la['items']} items labelled; {len(la['unclear'])} unclear).", ""]
    out = HERE / "results" / "report_tables.md"
    out.write_text("\n".join(md) + "\n")
    print(out)


if __name__ == "__main__":
    main()
