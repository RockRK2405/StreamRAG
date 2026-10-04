"""Brief §73: final end-to-end test through the full pipeline with the REAL local LLM (Ollama qwen3:4b).

Live transcript (streamed chunks, realtime) -> intent detection (Phase 5/6) -> claim requirements -> query analysis
-> adaptive retrieval policy -> cache check -> retrieval -> evidence sufficiency -> expansion / multi-hop -> evidence
sufficiency -> grounded generation (LLM) -> claim validation (NLI + rules) -> citation validation -> streaming answer
(Phase 8 runtime, adaptive_retrieval.enabled). Fixture corpus tests/fixtures/corpus_adaptive (TEST FIXTURE ONLY).

The expectations of each scenario were written before running (EXPECT below) and are checked on the runtime's own
event log; results are reported as they come out (pass / fail per check). One run per scenario. NOT REPORTABLE.

Usage: .venv/bin/python research/phase9/e2e_final.py --index-root /tmp/idx9    (needs `ollama serve`)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, meta, stack, with_cfg  # noqa: E402

from streamrag.generation.llm import OllamaBackend  # noqa: E402
from streamrag.runtime import StreamingRuntime  # noqa: E402
from streamrag.runtime.events import untraceable  # noqa: E402

S = [   # name, turns (list of chunk lists; a chunk ("R", n, text) is an ASR revision replacing chunk n)
    ("01_simple_question", [["How often must", "a residence permit", "be renewed?"]]),
    ("02_multi_intent", [["What is the application fee", "and how long does processing take",
                          "for domestic applicants?"]]),
    ("03_contextual_follow_up", [["What are the eligibility requirements", "for a residence permit?"],
                                 ["What about", "international applicants?"]]),
    ("04_exact_entity", [["What is Form PX-204", "used for?"]]),
    ("05_temporal", [["What was the application fee", "in December 2025?"]]),
    ("06_insufficient", [["Can applicants pay", "the fee by credit card?"]]),
    ("07_contradictory", [["When does the Permit", "Processing Office open?"]]),
    ("08_multi_hop", [["What documents does", "an applicant from", "Zemland need?"]]),
    ("09_cached_question", [["What is the residence permit", "application fee?"],
                            ["Which public holidays", "are there in 2026?"],
                            ["What is the application fee", "for a residence permit?"]]),
    ("10_changed_entity", [["Do applicants from Zemland", "need an interview?"], ["Sorry, I meant Norvia,", "not Zemland."]]),
    ("11_changed_constraint", [["How long does processing take", "for international applicants?"],
                               ["Actually, for domestic", "applicants instead."]]),
    ("12_streaming_correction", [["How long does processing take", "for domestic", ("R", 1, "for international"),
                                  "applicants?"]]),
]
EXPECT = {   # checks decided before running: (description, predicate over the scenario record)
    "01_simple_question": [("fast path LEXICAL", lambda r: r["strategies"][0] == "LEXICAL"),
                           ("answer cites RENEW §1", lambda r: "RENEW §1" in r["turns"][-1]["citations"])],
    "02_multi_intent": [("one plan per need (2)", lambda r: len(r["turns"][0]["plans"]) >= 2),
                        ("cites ELIG-2026 §2 and DOM-2026 §2",
                         lambda r: {"ELIG-2026 §2", "DOM-2026 §2"} <= set(r["turns"][0]["citations"]))],
    "03_contextual_follow_up": [("follow-up retrieval FILTERED international",
                                 lambda r: any(p["strategy"] == "FILTERED" for p in r["turns"][1]["plans"])),
                                ("follow-up cites INTL-2026 §1", lambda r: "INTL-2026 §1" in r["turns"][1]["citations"])],
    "04_exact_entity": [("LEXICAL on the identifier", lambda r: r["strategies"][0] == "LEXICAL"),
                        ("cites FORMS §1 or INTL-2026 §1",
                         lambda r: bool({"FORMS §1", "INTL-2026 §1"} & set(r["turns"][0]["citations"])))],
    "05_temporal": [("FILTERED by the December 2025 period",
                     lambda r: any((p.get("filters") or {}).get("valid_to") == "2025-12-31" for p in r["turns"][0]["plans"])),
                    ("cites ELIG-2024 §2, not ELIG-2026 §2",
                     lambda r: "ELIG-2024 §2" in r["turns"][0]["citations"]
                     and "ELIG-2026 §2" not in r["turns"][0]["citations"])],
    "06_insufficient": [("evidence INSUFFICIENT", lambda r: "INSUFFICIENT" in r["turns"][0]["assessments"]),
                        ("answer states the gap", lambda r: "not" in r["turns"][0]["answer"].lower())],
    "07_contradictory": [("evidence CONTRADICTORY, stop CONTRADICTION",
                          lambda r: "CONTRADICTION" in r["turns"][0]["stops"]),
                         ("answer reports 9:00 and 8:30",
                          lambda r: "9:00" in r["turns"][0]["answer"] and "8:30" in r["turns"][0]["answer"])],
    "08_multi_hop": [("hop zemland -> Group B", lambda r: "zemland -> Group B" in r["turns"][0]["hops"]),
                     ("cites GROUP-RULES §2", lambda r: "GROUP-RULES §2" in r["turns"][0]["citations"])],
    "09_cached_question": [("no adaptive search for the repeat",
                            lambda r: r["turns"][2]["adaptive_searches"] == 0),
                           ("repeat cites ELIG-2026 §2", lambda r: "ELIG-2026 §2" in r["turns"][2]["citations"])],
    "10_changed_entity": [("cache invalidated: entity_changed",
                           lambda r: "entity_changed" in r["turns"][1]["invalidations"]),
                          ("hop norvia -> Group A", lambda r: "norvia -> Group A" in r["turns"][1]["hops"])],
    "11_changed_constraint": [("re-plan FILTERED domestic",
                               lambda r: any((p.get("filters") or {}).get("metadata") == {"applicant_type": ["domestic"]}
                                             for p in r["turns"][1]["plans"])),
                              ("cites DOM-2026 §2", lambda r: "DOM-2026 §2" in r["turns"][1]["citations"])],
    "12_streaming_correction": [("last plan filters international",
                                 lambda r: (r["turns"][0]["plans"][-1].get("filters") or {}).get("metadata")
                                 == {"applicant_type": ["international"]}),
                                ("cites INTL-2026 §2", lambda r: "INTL-2026 §2" in r["turns"][0]["citations"])],
}


async def scenario(st, llm, name, turns) -> dict:
    rt = await StreamingRuntime(st.cfg, st, llm=llm).start()
    sid = rt.start_session(name.replace("_", "-"))
    for n, chunks in enumerate(turns, start=1):
        for c in chunks:
            if isinstance(c, tuple):
                rt.push_transcript_delta(sid, f"u{n}", c[2], replaces=c[1])
            else:
                rt.push_transcript_delta(sid, f"u{n}", c)
            await asyncio.sleep(0.25)
        rt.end_utterance(sid, f"u{n}")
        await rt.wait_idle(120) if hasattr(rt, "wait_idle") else await asyncio.sleep(3)
    await rt.complete_session(sid, 180)
    evs = rt.events(sid)
    await rt.shutdown()
    rec = {"name": name, "turns": [], "strategies": [], "untraceable": len(untraceable(evs)), "events": len(evs)}
    for n in range(1, len(turns) + 1):
        uid = f"u{n}"
        mine = [e for e in evs if e.utterance_id == uid]
        plans = [e.payload for e in mine if e.type.value == "RETRIEVAL_POLICY_SELECTED"]
        commits = [e for e in mine if e.type.value == "ANSWER_COMMITTED"]
        ans = commits[-1].payload if commits else {}
        verified = [e.payload for e in mine if e.type.value in ("CLAIM_VERIFIED", "CLAIM_REJECTED")]
        rec["strategies"] += [p["strategy"] for p in plans]
        rec["turns"].append({
            "utterance_id": uid, "plans": [{k: p.get(k) for k in ("strategy", "strategy_reason", "complexity",
                                                                    "top_k", "retrievers", "filters", "reranking")}
                                           for p in plans],
            "cache": [e.type.value for e in mine if e.type.value in ("CACHE_HIT", "CACHE_MISS")],
            "invalidations": [e.payload.get("reason") for e in evs if e.type.value == "RETRIEVAL_INVALIDATED"
                              and e.utterance_id == uid],
            "adaptive_searches": sum(1 for e in mine if e.type.value == "RETRIEVAL_STARTED"
                                     and e.payload.get("adaptive_search")),
            "assessments": [e.payload["assessment"] for e in mine if e.type.value == "RETRIEVAL_STOPPED"],
            "stops": [e.payload["stop_reason"] for e in mine if e.type.value == "RETRIEVAL_STOPPED"],
            "hops": [e.payload.get("bridge") for e in mine if e.type.value == "HOP_CREATED"],
            "answer": ans.get("text", ""), "citations": list(ans.get("citations", [])),
            "answer_status": ans.get("status"), "backend": ans.get("backend"),
            "claims_verified": sum(1 for v in verified if v.get("status") == "SUPPORTED"),
            "claims_rejected": sum(1 for e in mine if e.type.value == "CLAIM_REJECTED"),
            "citations_validated": sum(1 for e in mine if e.type.value == "CITATION_VALIDATED"),
            "streamed_answer_events": sum(1 for e in mine if e.type.value.startswith("ANSWER_")),
            "llm_calls": sum(1 for e in mine if e.type.value == "LLM_CALL"),
            "turn_completed": any(e.type.value == "TURN_COMPLETED" for e in mine)})
    checks = [(d, bool(_safe(f, rec))) for d, f in EXPECT[name]]
    rec["checks"] = [{"check": d, "pass": ok} for d, ok in checks]
    (OUT / "e2e_final").mkdir(parents=True, exist_ok=True)
    (OUT / "e2e_final" / f"{name}.jsonl").write_text("".join(e.canonical_json() + "\n" for e in evs))
    return rec


def _safe(f, rec):
    try:
        return f(rec)
    except (KeyError, IndexError, TypeError):
        return False


async def main_async(index_root: Path, model: str) -> None:
    url = "http://127.0.0.1:11434"
    if not OllamaBackend.reachable(url, model):
        raise SystemExit("needs `ollama serve` with the model")
    llm = OllamaBackend(url, model)
    llm.complete([{"role": "user", "content": "Reply with {}"}], {"type": "object"})        # warm-up
    st = with_cfg(stack("adaptive", index_root, llm=llm), **{"adaptive_retrieval.enabled": True,
                                                              "generation.backend": "ollama"})
    out = []
    for name, turns in S:
        r = await scenario(st, llm, name, turns)
        out.append(r)
        print(name, [(c["check"], c["pass"]) for c in r["checks"]],
              [(t["plans"][0]["strategy"] if t["plans"] else None, t["stops"], t["answer"][:120]) for t in r["turns"]],
              flush=True)
    passed = sum(c["pass"] for r in out for c in r["checks"])
    total = sum(len(r["checks"]) for r in out)
    md = ["# Phase 9 final end-to-end test (generated from the runtime event logs)", "",
          f"> Real local LLM ({model}), real retrieval / NLI, Phase 8 runtime (realtime), adaptive retrieval enabled. "
          "Fixture corpus tests/fixtures/corpus_adaptive - behaviour check, NOT a benchmark result.", "",
          f"Checks passed: {passed} / {total} (expectations written before the run).", ""]
    for r in out:
        md.append(f"## {r['name']}")
        for t in r["turns"]:
            p = t["plans"][0] if t["plans"] else {}
            md.append(f"- {t['utterance_id']}: strategy **{p.get('strategy')}** ({p.get('strategy_reason')}); cache "
                      f"{t['cache']}; searches {t['adaptive_searches']}; sufficiency {t['assessments']}; stop "
                      f"{t['stops']}; hops {t['hops']}; invalidations {t['invalidations']}; LLM calls {t['llm_calls']}; "
                      f"claims verified {t['claims_verified']}, rejected {t['claims_rejected']}; citations validated "
                      f"{t['citations_validated']}")
            md.append(f"  - answer ({t['answer_status']}, {t['backend']}): {t['answer']}")
        md.append("- checks: " + "; ".join(f"{c['check']}: {'PASS' if c['pass'] else 'FAIL'}" for c in r["checks"]))
        md.append(f"- events {r['events']}, untraceable {r['untraceable']}")
        md.append("")
    (OUT / "e2e_final" / "README.md").write_text("\n".join(md) + "\n")
    (OUT / "e2e_final" / "summary.json").write_text(json.dumps(
        {"meta": meta(llm=f"{model} (REAL)", mode="realtime runtime"), "passed": passed, "total": total,
         "scenarios": out}, indent=2, default=str))
    print("checks passed", passed, "/", total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen3:4b")
    a = ap.parse_args()
    asyncio.run(main_async(a.index_root, a.model))


if __name__ == "__main__":
    main()
