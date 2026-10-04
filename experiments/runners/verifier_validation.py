"""Claim verification accuracy (brief §14) - the instrument / verifier on claims with labels known by construction.

From every expected claim of the TEST split that cites a section, build:
  original      the reference claim                                   -> label SUPPORTED   (by its cited section)
  number_swap   first number replaced by a different number           -> label NOT_SUPPORTED
  negation      a negation inserted / removed after the first verb    -> label NOT_SUPPORTED
  entity_swap   the claim's key string replaced by another claim's key -> label NOT_SUPPORTED
and judge each against a pool = the cited section + 4 distractor sections of the same corpus (deterministic choice).
Reports accuracy, precision / recall of SUPPORTED, false-acceptance rate per perturbation type. Perturbations are
mechanical and can occasionally produce a sentence the corpus *does* support (e.g. a number that also appears
elsewhere) - such cases are counted as they are (no relabelling), so the numbers are a lower-bound-style check.

Usage: .venv/bin/python experiments/runners/verifier_validation.py --index-root /tmp/idx10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpora import CORPORA  # noqa: E402
from run_experiments import instrument_factory  # noqa: E402

from streamrag.evaluation.dataset import load  # noqa: E402
from streamrag.evaluation.systems import Stacks  # noqa: E402

NEG = re.compile(r"\b(is|are|must|can|need|needs|costs?|receive|travel|apply|run)\b", re.I)


def perturb(text: str, keys: list[str], other_keys: list[str]) -> dict[str, str]:
    out = {}
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    if m:
        v = m.group(0)
        new = str(int(float(v.replace(",", "."))) + 7) if "." not in v else f"{float(v) + 1.5:.2f}"
        out["number_swap"] = text[:m.start()] + new + text[m.end():]
    n = NEG.search(text)
    if n:
        w = n.group(0)
        if re.search(r"\bnot\b|\bno\b", text):
            out["negation"] = re.sub(r"\bnot\s+|\bno\s+", "", text, count=1)
        else:
            out["negation"] = text[:n.end()] + " not" + text[n.end():] if w.lower() in ("is", "are", "must", "can") \
                else text[:n.start()] + "do not " + w + text[n.end():]
    cand = [k for k in other_keys if k and all(k.lower() != x.lower() for x in keys)]
    if keys and cand and keys[0] in text:
        out["entity_swap"] = text.replace(keys[0], cand[0], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    a = ap.parse_args()
    stacks = Stacks(REPO, CORPORA, a.index_root)
    inst = instrument_factory(stacks)("transit")
    st = stacks.get("transit", "extractive")
    by_cit = {c.citation: c for c in st.bundle.chunks}
    cits = sorted(by_cit)
    claims = [c for s in load(REPO / "experiments" / "datasets" / "streamrag_eval_v1" / "test.jsonl")
              for c in s.expected_claims if c.citation in by_cit]
    seen, uniq = set(), []
    for c in claims:
        if c.text not in seen:
            seen.add(c.text)
            uniq.append(c)
    all_keys = [k for c in uniq for k in c.key]
    rows = []
    for i, c in enumerate(uniq):
        pool_cits = [c.citation] + [x for x in cits if x != c.citation][i % 7::7][:4]
        pool = {by_cit[x].chunk_id: by_cit[x].text for x in pool_cits}
        variants = {"original": c.text, **perturb(c.text, c.key, all_keys[i + 1:] + all_keys[:i])}
        for kind, text in variants.items():
            v = inst.verifier.verify(f"v{i}{kind}", text, [], pool)
            rows.append({"claim": c.text, "kind": kind, "text": text, "label": "SUPPORTED" if kind == "original"
                         else "NOT_SUPPORTED", "verdict": "SUPPORTED" if v.supported else v.status})
    tp = sum(1 for r in rows if r["label"] == "SUPPORTED" and r["verdict"] == "SUPPORTED")
    fp = sum(1 for r in rows if r["label"] != "SUPPORTED" and r["verdict"] == "SUPPORTED")
    fn = sum(1 for r in rows if r["label"] == "SUPPORTED" and r["verdict"] != "SUPPORTED")
    tn = sum(1 for r in rows if r["label"] != "SUPPORTED" and r["verdict"] != "SUPPORTED")
    by_kind = {}
    for k in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == k]
        by_kind[k] = {"n": len(rs), "judged_supported": sum(r["verdict"] == "SUPPORTED" for r in rs)}
    res = {"meta": {"instrument": "Phase 7 ClaimVerifier (nli-deberta-v3-xsmall + rules)",
                    "labels": "by construction (perturbations of reference claims)", "reportable": False},
           "n": len(rows), "accuracy": round((tp + tn) / len(rows), 4),
           "precision_supported": round(tp / (tp + fp), 4) if tp + fp else None,
           "recall_supported": round(tp / (tp + fn), 4) if tp + fn else None,
           "false_acceptance_rate": round(fp / (fp + tn), 4) if fp + tn else None,
           "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "by_kind": by_kind, "rows": rows}
    out = REPO / "experiments" / "results" / "VERIFIER_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
