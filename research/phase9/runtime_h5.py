"""H5 (brief §46): does cancellation + adaptive retrieval reduce wasted computation? Phase 8 runtime, realtime mode.

Every single-turn question of eval/dev_adaptive_retrieval is streamed at high frequency (2 words per chunk, 80 ms
apart): early retrieval fires on partial transcripts, so most queries are superseded while the user is still
speaking. Four configurations (2 x 2):

  FIXED_NOCANCEL     fixed hybrid retrieval (Phase 8), superseded work runs to completion
  FIXED_CANCEL       fixed hybrid retrieval, Phase 8 cooperative cancellation (default)
  ADAPTIVE_NOCANCEL  adaptive retrieval (one RETRIEVAL task per query runs the controller), no cancellation
  ADAPTIVE_CANCEL    adaptive retrieval + cancellation (the controller checks the task's token between searches)

Wasted computation = measured worker time (exec_ms) of retrieval tasks (lexical, dense, assemble, retrieval) whose
query ended superseded / cancelled / stale. Useful = the same for queries that stayed current. Quality = coverage of
the gold citations by the evidence of the turn's final queries. Extractive answers (no LLM). One run per question.
NOT REPORTABLE (fixture corpora).

Usage: .venv/bin/python research/phase9/runtime_h5.py --index-root /tmp/idx9
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cases, mean, meta, pct, recall, stack, with_cfg, write  # noqa: E402

from streamrag.models.events import SessionEnd, SessionStart  # noqa: E402
from streamrag.runtime import Fault, FaultInjector, StreamingRuntime  # noqa: E402
from streamrag.runtime.driver import drive_realtime  # noqa: E402
from streamrag.streaming import simulator as sim  # noqa: E402

CONFIGS = {
    "FIXED_NOCANCEL": {"adaptive_retrieval.enabled": False, "runtime.cancel_running": False,
                       "controller.cancel_superseded": "never"},
    "FIXED_CANCEL": {"adaptive_retrieval.enabled": False},
    "ADAPTIVE_NOCANCEL": {"adaptive_retrieval.enabled": True, "runtime.cancel_running": False,
                          "controller.cancel_superseded": "never"},
    "ADAPTIVE_CANCEL": {"adaptive_retrieval.enabled": True},
}
RETRIEVAL_TASKS = {"lexical", "dense", "assemble", "retrieval"}


def chunks(text: str, n: int = 2) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]


async def one(st, case, dense_delay_ms: float = 0.0) -> dict:
    faults = FaultInjector([Fault("dense", "delay", times=-1, delay_ms=dense_delay_ms)]) if dense_delay_ms else None
    rt = await StreamingRuntime(st.cfg, st, faults=faults).start()
    text = case["turns"][0]["utterance_text"]
    sid = case["case_id"].lower()
    evs = [SessionStart(session_id=sid)] + sim.stream(chunks(text), interval_ms=80, session_id=sid, utterance_id="u1",
                                                      wrap_session=False) + [SessionEnd(session_id=sid)]
    await drive_realtime(rt, evs, close=True, timeout_s=60)
    out = rt.events(sid)
    rs = rt.sessions[sid]
    led = rs.session.ledger
    await rt.shutdown()
    wasted = useful = 0.0
    n_tasks = n_wasted_tasks = cancelled = 0
    for e in out:
        if e.type.value not in ("TASK_COMPLETED", "TASK_CANCELLED", "TASK_FAILED", "TASK_TIMED_OUT"):
            continue
        p = e.payload
        if p.get("task_type") not in RETRIEVAL_TASKS or not p.get("query_id"):
            continue
        ms = float((p.get("wall") or {}).get("exec_ms") or 0.0)
        rec = led.get(p["query_id"])
        dead = rec is None or rec.status == "cancelled" or rec.stale_reason is not None or \
            e.type.value == "TASK_CANCELLED"
        n_tasks += 1
        cancelled += e.type.value == "TASK_CANCELLED"
        if dead:
            wasted += ms
            n_wasted_tasks += 1
        else:
            useful += ms
    cite = {c.chunk_id: c.citation for c in st.bundle.chunks}
    final = [r for r in led.all() if r.status in ("completed", "reused") and r.stale_reason is None]
    keys = list(dict.fromkeys(cite.get(i, i) for r in final for i in r.evidence_ids))
    gold = case["turns"][0]["gold"]["citations"]
    stale_discarded = sum(1 for e in out if e.type.value == "STALE_RESULT_DISCARDED")
    queries = len(led.all())
    adaptive_searches = sum(1 for e in out if e.type.value == "RETRIEVAL_STARTED" and e.payload.get("adaptive_search"))
    return {"case_id": case["case_id"], "wasted_ms": round(wasted, 3), "useful_ms": round(useful, 3),
            "tasks": n_tasks, "wasted_tasks": n_wasted_tasks, "cancelled_tasks": cancelled, "queries": queries,
            "stale_discarded": stale_discarded, "adaptive_searches_committed": adaptive_searches,
            "coverage": recall(keys, gold) if gold else None,
            "turn_completed": any(e.type.value == "TURN_COMPLETED" for e in out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--dense-delay-ms", type=float, nargs="*", default=[0.0, 150.0])
    a = ap.parse_args()
    singles = [c for c in cases() if len(c["turns"]) == 1]
    res = {}
    for delay in a.dense_delay_ms:
        cond = "local" if not delay else f"remote_dense_{int(delay)}ms_SYNTHETIC"
        res[cond] = {}
        for name, ov in CONFIGS.items():
            rows = []
            for c in singles:
                st = with_cfg(stack(c["corpus"], a.index_root), **ov)
                rows.append(asyncio.run(one(st, c, delay)))
            tot_w = sum(r["wasted_ms"] for r in rows)
            tot_u = sum(r["useful_ms"] for r in rows)
            res[cond][name] = {
                "wasted_ms_total": round(tot_w, 1), "useful_ms_total": round(tot_u, 1),
                "wasted_share": round(tot_w / (tot_w + tot_u), 4) if tot_w + tot_u else None,
                "wasted_ms_per_turn": pct(r["wasted_ms"] for r in rows),
                "retrieval_tasks": sum(r["tasks"] for r in rows), "wasted_tasks": sum(r["wasted_tasks"] for r in rows),
                "cancelled_tasks": sum(r["cancelled_tasks"] for r in rows), "queries": sum(r["queries"] for r in rows),
                "stale_discarded": sum(r["stale_discarded"] for r in rows), "coverage": mean(r["coverage"] for r in rows),
                "turns_completed": sum(r["turn_completed"] for r in rows), "turns": len(rows), "rows": rows}
            print(cond, name, {k: v for k, v in res[cond][name].items() if k != "rows"}, flush=True)
    write("runtime_h5.json", {"meta": meta(mode="realtime runtime", streaming="2 words / chunk, 80 ms apart",
                                           runs="one run per question",
                                           injected="remote_* conditions: +delay per dense search (SYNTHETIC remote "
                                                    "vector store); local: none"), "conditions": res})


if __name__ == "__main__":
    main()
