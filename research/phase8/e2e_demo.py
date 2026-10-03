"""Brief §75: end-to-end demonstration through the real runtime (real retrieval, real NLI, real local LLM).

USER (streamed): "What are the eligibility | requirements | for the permit?" ... "For international | applicants."

The trace below is generated from the runtime's own event log (nothing hand-written). To make the superseded query
still be running when the refinement arrives, the dense index is given 1200 ms of injected latency (a remote vector
store) - labelled SYNTHETIC; everything else is real. Fixture corpus: behaviour demonstration, NOT a benchmark.

Usage: .venv/bin/python research/phase8/e2e_demo.py --index-root /tmp/idx8   (needs `ollama serve`)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, meta, stack_for  # noqa: E402

from streamrag.generation.llm import OllamaBackend  # noqa: E402
from streamrag.runtime import Fault, FaultInjector, StreamingRuntime  # noqa: E402
from streamrag.runtime.events import path_to_root, untraceable  # noqa: E402
from streamrag.runtime.replay import behaviour, replay_runtime  # noqa: E402
from streamrag.streaming.printer import format_event  # noqa: E402

U1 = ["What are the eligibility", "requirements", "for the permit?"]
U2 = ["For international", "applicants."]
EXTRA = {"TASK_SCHEDULED", "TASK_STARTED", "TASK_COMPLETED", "QUERY_SUPERSEDED", "CONTEXT_CHANGE_DETECTED",
         "DELTA_PLAN_CREATED", "CLAIM_REVALIDATED", "CLAIM_INVALIDATED", "ANSWER_DELTA", "ANSWER_REVISED"}


def line(e) -> str | None:
    t, p = e.type.value, e.payload
    if t in ("TASK_SCHEDULED", "TASK_STARTED", "TASK_COMPLETED"):
        if p["task_type"] in ("generation", "draft", "answer_extractive", "lexical", "dense"):
            extra = f" for {p['query_id']}" if p.get("query_id") else f" ({p.get('answer_kind')})"
            return f"TASK       {p['task_id']} {p['task_type']}{extra} {t[5:].lower()} prio={p['priority']}"
        return None
    if t == "QUERY_SUPERSEDED":
        return f"SUPERSEDED {p.get('query_id')} (delta plan {p.get('plan_id')})"
    if t == "ANSWER_DELTA":
        return (f"DELTA      {p['answer_id']} v{p['version']} {p['status']}: +{len(p['added'])} claims, "
                f"-{len(p['removed'])}, {p['unchanged']} unchanged")
    if t in ("CLAIM_REVALIDATED", "CLAIM_INVALIDATED"):
        return f"CLAIM      {p['claim_id']} {p['from_status']}->{p['to_status']} ({p['reason']})"
    return None


async def main_async(index_root: Path, model: str) -> None:
    url = "http://127.0.0.1:11434"
    if not OllamaBackend.reachable(url, model):
        raise SystemExit("needs `ollama serve` with the model")
    llm = OllamaBackend(url, model)
    st = stack_for("grounding", index_root, backend=llm, **{"generation.backend": "ollama"})
    llm.complete([{"role": "user", "content": "Reply with {}"}], {"type": "object"})        # warm-up
    faults = FaultInjector([Fault("dense", "delay", times=-1, delay_ms=1200)])
    rt = await StreamingRuntime(st.cfg, st, llm=llm, faults=faults).start()
    sid = rt.start_session("demo")
    for c in U1:                                                    # early retrieval starts mid-sentence
        rt.push_transcript_delta(sid, "u1", c)
        await asyncio.sleep(0.35)
    rt.end_utterance(sid, "u1")
    await asyncio.sleep(0.6)                                        # u1's answer is being generated
    for c in U2:                                                    # the late detail
        rt.push_transcript_delta(sid, "u2", c)
        await asyncio.sleep(0.35)
    rt.end_utterance(sid, "u2")
    await rt.complete_session(sid, 120)
    summ = rt.summary()
    await rt.shutdown()
    evs = rt.events(sid)
    out = OUT / "e2e_demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.jsonl").write_text("".join(e.canonical_json() + "\n" for e in evs))
    md = ["# Phase 8 end-to-end demo (generated from the runtime's event log)", "",
          "> Fixture corpus, real retrieval / NLI / local LLM (qwen3:4b). Dense index latency +1200 ms injected "
          "(SYNTHETIC remote vector store). Behaviour demonstration, NOT a benchmark result.", "",
          "User (streamed): `" + " | ".join(U1) + "` ... (answer being generated) ... `" + " | ".join(U2) + "`",
          "", "```"]
    for e in evs:
        txt = format_event(e) if e.type.value not in EXTRA else None
        if txt is None and e.type.value in EXTRA:
            ln = line(e)
            txt = None if ln is None else f"[{e.t_session_ms / 1000:06.3f}] {e.utterance_id or '--':4} {ln}"
        if txt:
            md.append(txt)
    md += ["```", ""]
    commit = [e for e in evs if e.type.value == "ANSWER_COMMITTED"][-1]
    chain = [f"{x.type.value} ({x.event_id})" for x in reversed(path_to_root(evs, commit.event_id))]
    rep = replay_runtime(st.cfg, st, evs)
    md += ["Trace path of the final answer (parent links): " + " -> ".join(chain), "",
           f"Events: {len(evs)}; untraceable: {len(untraceable(evs))}; virtual-clock replay with the recorded LLM "
           f"output: behaviour identical = {rep['behaviour_identical']} (exact = {rep['identical']}: realtime "
           "timings differ by design).", ""]
    (out / "README.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "meta": meta(llm=f"{model} (REAL)", injected="dense +1200 ms (SYNTHETIC)"), "runtime": summ,
        "final_answer": commit.payload["text"], "untraceable": untraceable(evs),
        "replay": {k: v for k, v in rep.items() if k in ("identical", "behaviour_identical", "n_original", "n_replayed")},
        "behaviour": behaviour(evs)}, indent=2, default=str))
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen3:4b")
    a = ap.parse_args()
    asyncio.run(main_async(a.index_root, a.model))


if __name__ == "__main__":
    main()
