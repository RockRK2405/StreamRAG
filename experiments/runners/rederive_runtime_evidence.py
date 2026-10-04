"""Re-derives the per-turn evidence of stored runtime runs from their event traces (evaluation-framework fix).

Bug (found during Phase 10 analysis, disclosed in the report §29): the runtime harness read each turn's evidence
from the session ledger *after the whole session*, so an earlier turn of a conversation got the evidence of the later
turn that superseded it (or none). Retrieval, evidence and citation-validity metrics of earlier turns were wrong; the
system's behaviour and answers were not affected. The fix (`systems.turn_evidence`) reads the turn's EVIDENCE_FUSED
events. This script applies it to runs made before the fix, without re-running the systems:

  for every runs/<variant>__<split>.jsonl of family runtime (not memoryless - its turns run in separate runtimes, so
  the ledger was already per turn): evidence, keys and ops.documents_retrieved are recomputed from the stored trace
  of the turn's session; the original file is kept as <variant>__<split>.pre_evidence_fix.jsonl, the scored cache is
  removed (re-scored by the next run_experiments.py), and the run's meta records the fix.

Usage: .venv/bin/python experiments/runners/rederive_runtime_evidence.py --index-root /tmp/idx10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpora import CORPORA  # noqa: E402

from streamrag.evaluation.dataset import load, sessions  # noqa: E402
from streamrag.evaluation.systems import Stacks, _ev, _keys, turn_evidence  # noqa: E402

RUNS = REPO / "experiments" / "results" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    a = ap.parse_args()
    systems = yaml.safe_load((REPO / "experiments" / "configs" / "systems.yaml").read_text())
    stacks = Stacks(REPO, CORPORA, a.index_root, rerank=False)
    summary = {}
    for path in sorted(RUNS.glob("*__*.jsonl")):
        if path.name.endswith((".scored.jsonl", ".pre_evidence_fix.jsonl")):
            continue
        variant, split = path.stem.split("__")
        spec = systems.get(variant, {})
        if spec.get("family") != "runtime" or spec.get("memoryless"):
            continue
        meta_p = path.with_suffix(".meta.json")
        meta = json.loads(meta_p.read_text())
        if meta.get("evidence_rederived"):
            summary[variant] = "already re-derived"
            continue
        data = load(REPO / "experiments" / "datasets" / "streamrag_eval_v1" / f"{split}.jsonl")
        if spec.get("query_types"):
            keep = {x.session_id for x in data if x.query_type in spec["query_types"]}
            data = [x for x in data if x.session_id in keep]
        uid_of, trace_of, corpus_of = {}, {}, {}
        for sess in sessions(data):
            for n, s in enumerate(sess, start=1):
                uid_of[s.sample_id] = f"u{n}"
                trace_of[s.sample_id] = RUNS / "traces" / f"{variant}__{sess[0].session_id.replace('.', '-')}.jsonl"
                corpus_of[s.sample_id] = s.corpus
        recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        cache: dict[Path, list] = {}
        changed = missing = 0
        out = []
        for r in recs:
            sid = r["sample_id"]
            t = trace_of.get(sid)
            if r["status"] != "ok" or t is None or not t.exists():
                missing += r["status"] == "ok"
                out.append(r)
                continue
            if t not in cache:
                cache[t] = [json.loads(x) for x in t.read_text().splitlines() if x.strip()]
            st = stacks.get(corpus_of[sid], "extractive")
            ev = _ev(st, turn_evidence(cache[t], uid_of[sid]) or [])
            keys = _keys(ev)
            if keys != r["keys"]:
                changed += 1
            r = dict(r, evidence=ev, keys=keys)
            r["ops"] = dict(r["ops"], documents_retrieved=len({e["document_id"] for e in ev}))
            out.append(r)
        path.with_suffix(".pre_evidence_fix.jsonl").write_text(path.read_text())
        path.write_text("".join(json.dumps(r, default=str) + "\n" for r in out))
        meta["evidence_rederived"] = {"from": "event traces (systems.turn_evidence)", "turns_changed": changed,
                                      "turns_without_trace": missing,
                                      "original": path.with_suffix(".pre_evidence_fix.jsonl").name}
        meta_p.write_text(json.dumps(meta, indent=2, default=str))
        sc = path.with_suffix(".scored.jsonl")
        if sc.exists():
            sc.unlink()
        summary[variant] = meta["evidence_rederived"]
        print(variant, split, meta["evidence_rederived"], flush=True)
    (RUNS / "evidence_rederivation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
