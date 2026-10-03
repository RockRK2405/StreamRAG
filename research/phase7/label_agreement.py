"""Agreement between the claim verifier and hand labels on real LLM claims (Phase 7 report §6, §16).

Inputs (``--results`` selects the run directory, default research/phase7/results):
  <results>/claims_for_labeling.jsonl        blind sheet: claim + the exact evidence pool, no verdicts
  research/phase7/labels/claim_labels.jsonl  {"case_id", "turn", "claim", "label": SUPPORTED | NOT_SUPPORTED | UNCLEAR,
                                             "note"} - keyed by claim text, so labels carry over between runs whose
                                             LLM output is identical; sheet items without a label are listed
  <results>/arms.json                        verifier verdicts (joined only here, after labelling)

Label meaning (fixed before labelling): SUPPORTED = some text of the pool states the whole claim (every part, numbers
and modality included), even if another pool text disagrees (conflict sides are presented with both sources);
NOT_SUPPORTED = some part is stated by no pool text (including a number written as a different value: ".4" for 4);
UNCLEAR = not a proposition (truncated or garbled words, meta-statements about the evidence) - excluded and listed.
Labelled blind: the sheet shows the claim and the pool only, not the arm it came from or the verifier's verdict.

Verifier "supported" = status SUPPORTED, or CONTRADICTED with supporting evidence (one side of a presented conflict).
Released claims (F_released) are supported by construction; rejected ones (F_rejected) are not.

Caveat (stated in the report): the annotator is the implementer (Claude), labelling blind to the verdicts but not
independent of the system's design; one annotator, no agreement statistic between annotators. Fixture domain, NOT
REPORTABLE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def verdicts() -> dict[tuple, bool]:
    rows = json.loads((RES / "arms.json").read_text())["rows"]
    out = {}
    for r in rows:
        key = (r["case_id"], r["turn"])
        for arm in ("B", "C"):
            for c in r["arms"].get(arm, {}).get("raw", []):
                ok = c["status"] == "SUPPORTED" or (c["status"] == "CONTRADICTED" and bool(c.get("supporting")))
                out.setdefault((*key, arm, c["text"]), ok)
        fa = r["arms"].get("F", {})
        if fa.get("new_version"):
            for c in fa["claims"]:
                if c["kind"] == "fact":
                    out.setdefault((*key, "F_released", c["text"]), True)
            for x in fa["rejected"]:
                out.setdefault((*key, "F_rejected", x["text"]), False)
    return out


def rates(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs: (verifier_supported, label_supported)."""
    tp = sum(v and g for v, g in pairs)
    fp = sum(v and not g for v, g in pairs)
    fn = sum(g and not v for v, g in pairs)
    tn = sum(not v and not g for v, g in pairs)
    n = len(pairs)
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": div(tp + tn, n),
            "support_precision": div(tp, tp + fp), "support_recall": div(tp, tp + fn),
            "unsupported_detection_recall": div(tn, tn + fp)}


def main() -> None:
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RES)
    RES = ap.parse_args().results
    sheet = [json.loads(x) for x in (RES / "claims_for_labeling.jsonl").read_text().splitlines() if x.strip()]
    by_text = {(d["case_id"], d["turn"], d["claim"]): d for d in (
        json.loads(x) for x in (HERE / "labels" / "claim_labels.jsonl").read_text().splitlines() if x.strip())}
    labels = {it["item"]: by_text[(it["case_id"], it["turn"], it["claim"])] for it in sheet
              if (it["case_id"], it["turn"], it["claim"]) in by_text}
    v = verdicts()
    by_source: dict[str, list] = {}
    unclear, missing = [], []
    for it in sheet:
        lab = labels.get(it["item"])
        if lab is None:
            missing.append(it["item"])
            continue
        if lab["label"] == "UNCLEAR":
            unclear.append(it["item"])
            continue
        gold = lab["label"] == "SUPPORTED"
        for src in it["sources"]:
            ver = v.get((it["case_id"], it["turn"], src, it["claim"]))
            if ver is not None:
                by_source.setdefault(src, []).append((ver, gold, it["item"]))
    allp = {}
    for src, lst in by_source.items():                # one pair per item for the pooled numbers
        for ver, gold, item in lst:
            allp.setdefault(item, (ver, gold))
    out = {"REPORTABLE": False, "annotator": "Claude (implementer), blind to verifier verdicts, single annotator",
           "items": len(sheet), "labelled": len(sheet) - len(missing), "unclear": unclear, "missing": missing,
           "pooled": rates(list(allp.values())),
           "by_source": {s: rates([(a, b) for a, b, _ in lst]) for s, lst in sorted(by_source.items())},
           "released_claim_precision_vs_labels": None, "disagreements": []}
    rel = [(a, b) for a, b, _ in by_source.get("F_released", [])]
    if rel:
        out["released_claim_precision_vs_labels"] = round(sum(b for _, b in rel) / len(rel), 4)
    text = {it["item"]: it for it in sheet}
    for item, (ver, gold) in sorted(allp.items()):
        if ver != gold:
            out["disagreements"].append({"item": item, "claim": text[item]["claim"], "sources": text[item]["sources"],
                                         "verifier_supported": ver, "label_supported": gold,
                                         "note": labels[item].get("note", "")})
    (RES / "label_agreement.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("items", "labelled", "pooled", "released_claim_precision_vs_labels")},
                     indent=1))


if __name__ == "__main__":
    main()
