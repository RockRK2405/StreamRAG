"""Phase 7 final end-to-end test (brief §64): streaming transcript -> intents -> decomposition -> queries -> parallel
retrieval -> fusion -> session state -> delta detection / retrieval -> claim planning -> grounded generation (local
LLM) -> claim extraction -> verification -> citations -> validation -> final grounded answer.

Ten scenarios run through the *streaming* session (virtual clock, chunk by chunk) with the real local LLM. For each,
a readable trace is printed from the run's own telemetry (nothing hand-written) and the trace is replayed exactly
(LLM outputs replayed from the trace). TEST FIXTURE corpora: behaviour demonstration only, NOT a benchmark result.

Usage: .venv/bin/python research/phase7/e2e_streaming.py --index-root /tmp/idx7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "e2e"

from streamrag.config import load_config  # noqa: E402
from streamrag.models.events import SessionEnd, SessionStart  # noqa: E402
from streamrag.replay import ReplayEngine, dump_trace  # noqa: E402
from streamrag.retrieval import build_index  # noqa: E402
from streamrag.streaming import run_virtual  # noqa: E402
from streamrag.streaming import simulator as sim  # noqa: E402
from streamrag.streaming.factory import build_stack  # noqa: E402

CORPORA = {"fixture": "tests/fixtures/corpus", "grounding": "tests/fixtures/corpus_grounding",
           "injection": "tests/fixtures/corpus_injection"}
SCENARIOS = [
    ("01_simple_question", "fixture", [["How high should", "the wicks be trimmed?"]]),
    ("02_multi_intent", "grounding", [["Tell me the eligibility requirements", "and the application process",
                                       "for the permit."]]),
    ("03_incomplete_streaming_question", "fixture", [["What are the rules", "for ladders", "in the orchard?"]]),
    ("04_late_constraint", "fixture", [["What are the rules for ladders", "in the orchard?"],
                                       ["Specifically", "overnight."]]),
    ("05_entity_correction", "fixture", [["What are the rules for ladders", "in the orchard?"],
                                         ["Sorry, I meant crates", "instead of ladders."]]),
    ("06_unsupported_fact", "grounding", [["How much does it cost", "to renew a permit?"]]),
    ("07_contradictory_evidence", "grounding", [["What is the application fee", "for a new permit?"]]),
    ("08_missing_evidence", "grounding", [["What is the parking policy", "at the permit office?"]]),
    ("09_incremental_revision", "fixture", [["How are crates handled", "at harvest?"], ["Only for the", "night shift."],
                                            ["Actually, ignore", "the night shift restriction."]]),
    ("10_citation_integrity", "injection", [["Is the permit office open", "on public holidays?"]]),
]
SHOW = {"CONTEXT_CHANGE_DETECTED", "DELTA_PLAN_CREATED", "QUERY_GENERATED", "QUERY_REUSED", "CLAIM_PLAN_CREATED",
        "LLM_CALL", "CLAIM_VERIFIED", "CLAIM_REJECTED", "CLAIM_REPAIRED", "VALIDATION_RETRIEVAL", "ANSWER_VALIDATED",
        "ANSWER_FINALIZED", "ANSWER_COMPLETED", "UTTERANCE_FINALIZED"}


def line(e) -> str | None:
    t, p = e.type.value, e.payload
    if t == "UTTERANCE_FINALIZED":
        return f"utterance finalized: {p['transcript']!r}"
    if t == "CONTEXT_CHANGE_DETECTED":
        return f"change {p['change_id']} {p['change_type']} affected={p['affected_intents']} new={p['new_intents']}"
    if t == "DELTA_PLAN_CREATED":
        return (f"delta plan {p['plan_id']}: create={[q['query']['text'] for q in p['queries_to_create']]} "
                f"reuse={[q['action'] for q in p['queries_to_reuse']]}")
    if t == "QUERY_GENERATED":
        return f"query {p['query_id']} ({p['trigger']}): {p['query_text']!r}"
    if t == "QUERY_REUSED":
        return f"query reused ({p['action']}) {p['reused_query_id']}"
    if t == "CLAIM_PLAN_CREATED":
        return (f"claim plan for {p['answer_id']} ({'draft' if p['draft'] else 'final'}): "
                + "; ".join(f"{s['section_id']}: {len(s['facts'])} facts" + (f", gaps {[g['kind'] for g in s['gaps']]}"
                                                                           if s["gaps"] else "")
                            for s in p["sections"]))
    if t == "LLM_CALL":
        return (f"LLM call ({p['model']}, {p['purpose']}): ok={p['ok']} tokens {p['prompt_tokens']}->{p['output_tokens']}"
                f" ttft={p['wall'].get('ttft_ms') and round(p['wall']['ttft_ms'])} ms "
                f"total={round(p['wall']['total_ms'])} ms")
    if t in ("CLAIM_VERIFIED", "CLAIM_REJECTED"):
        return (f"{'verified' if t == 'CLAIM_VERIFIED' else 'REJECTED'} {p['status']} -> {p['action']}: {p['text']!r} "
                f"support={p['supporting_evidence']} contra={p['contradicting_evidence']}")
    if t == "CLAIM_REPAIRED":
        return f"repair {p['action']}: -> {p['repair']['to_texts']}"
    if t == "VALIDATION_RETRIEVAL":
        return f"validation retrieval for {p['claim_id']}: new {p['new_evidence']} -> {p['status_after']}"
    if t == "ANSWER_VALIDATED":
        m = p["metrics"]
        return (f"answer {p['answer_id']} {p['status']} partial={p['partial']} raw claims={m['raw_claims']} "
                f"raw support={m['raw_support_rate']} citation precision={m['citation_precision']} "
                f"intent coverage={m['intent_coverage']}")
    if t == "ANSWER_COMPLETED":
        return f"ANSWER {p['answer_id']} ({p['status']}):\n      {p['text']}"
    if t == "ANSWER_FINALIZED":
        return f"finalized {p['answer_id']} citations={p['citations']}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stacks = {}
    for k, path in CORPORA.items():
        cfg = load_config(REPO / "configs" / "default.yaml",
                          {"paths.corpus": str(REPO / path), "paths.index_root": str(a.index_root / k),
                           "telemetry.log_level": "ERROR", "multi_intent.enabled": True, "session.enabled": True,
                           "generation.enabled": True, "generation.backend": "ollama"}, base_dir=REPO)
        stacks[k] = build_stack(cfg, build_index(cfg).path)
    summary = []
    md = ["# Phase 7 end-to-end streaming scenarios (generated)", "",
          "> TEST FIXTURE corpora, local LLM; generated by `research/phase7/e2e_streaming.py` from the runs' telemetry "
          "(nothing hand-written). Behaviour demonstration only, NOT a benchmark result.", ""]
    for name, corpus, turns in SCENARIOS:
        st = stacks[corpus]
        evs = [SessionStart(session_id=name)]
        off = 0.0
        for n, chunks in enumerate(turns, start=1):
            evs += sim.stream(chunks, interval_ms=400, session_id=name, utterance_id=f"u{n}", offset_ms=off,
                              wrap_session=False)
            off += 400 * len(chunks) + 2500
        evs.append(SessionEnd(session_id=name))
        t0 = time.perf_counter()
        run = run_virtual(st.cfg, st.service, st.policy, evs, index_hash=st.index_hash, intent_stack=st.intent_stack)
        wall = (time.perf_counter() - t0) * 1000.0
        dump_trace(run.events, OUT / f"{name}.jsonl")
        rep = ReplayEngine(st.cfg, st.service, st.policy, st.index_hash, st.intent_stack).replay(run.events)
        finals = [e.payload for e in run.events if e.type.value == "ANSWER_VALIDATED"
                  and e.payload["status"] != "DRAFT"]
        drafts = [e for e in run.events if e.type.value == "ANSWER_VALIDATED" and e.payload["status"] == "DRAFT"]
        cits = [e.payload for e in run.events if e.type.value == "CITATION_VALIDATED"]
        llm = [e.payload for e in run.events if e.type.value == "LLM_CALL"]
        last_text = [e.payload["text"] for e in run.events if e.type.value == "ANSWER_COMPLETED"
                     and e.payload["status"] != "DRAFT"]
        summary.append({"scenario": name, "turns": len(turns), "wall_ms": round(wall, 1),
                        "final_answers": len(finals), "drafts": len(drafts), "llm_calls": len(llm),
                        "statuses": [f["status"] for f in finals], "replay_identical": rep.identical,
                        "citations_checked": sum(c["checked"] for c in cits),
                        "citations_valid": sum(c["valid"] for c in cits),
                        "orphan_claims": sum(len(c["orphan_claims"]) for c in cits),
                        "final_text": last_text[-1] if last_text else None})
        md += [f"## {name}", "", "Utterances: " + " / ".join(" | ".join(c) for c in turns), ""]
        for e in run.events:
            if e.type.value in SHOW:
                if e.type.value in ("CLAIM_PLAN_CREATED", "ANSWER_VALIDATED", "ANSWER_COMPLETED") \
                        and e.payload.get("status", "x") == "DRAFT" and e.type.value != "CLAIM_PLAN_CREATED":
                    continue
                if e.type.value == "CLAIM_PLAN_CREATED" and e.payload["draft"]:
                    continue
                ln = line(e)
                if ln:
                    md.append(f"- `{e.t_session_ms:7.0f} ms` {e.utterance_id or '--'} {ln}")
        md += ["", f"Drafts streamed before turn ends: {len(drafts)}. Replay identical: {rep.identical}.", ""]
        print(name, summary[-1]["statuses"], "llm", len(llm), "replay", rep.identical, flush=True)
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
