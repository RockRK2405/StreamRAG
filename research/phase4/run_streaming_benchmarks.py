"""Phase 4 streaming benchmarks on the DEV suite (fixture domain => NOT REPORTABLE as official results).

1. Virtual (deterministic) mode, policies A/B/C/C-prototype           -> results/virtual/
2. Realtime mode (asyncio, wall clock, speed 1.0), policies A/B/C        -> results/realtime/
3. Sensitivity: controller in virtual mode with modeled retrieval latency 10 / 200 / 1000 ms
4. Profiling: realtime controller pass collecting per-stage wall timings  -> results/profile.json
5. One complete replay of a saved trace                                 -> results/replay_check.json
6. Realtime/virtual parity: every realtime trace replayed in virtual mode -> results/realtime_replay_parity.json

Usage: .venv/bin/python research/phase4/run_streaming_benchmarks.py --index-root <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"

from streamrag.bench.streaming import POLICIES, load_cases, run_streaming_benchmark  # noqa: E402
from streamrag.config import load_config  # noqa: E402
from streamrag.controller.lexicon import Lexicon  # noqa: E402
from streamrag.controller.query_builder import QueryBuilder  # noqa: E402
from streamrag.replay import ReplayEngine, dump_trace, read_trace  # noqa: E402
from streamrag.retrieval import build_index  # noqa: E402
from streamrag.streaming import simulator as sim  # noqa: E402
from streamrag.streaming.factory import build_stack  # noqa: E402
from streamrag.streaming.runner import arun_realtime, run_virtual  # noqa: E402
from streamrag.telemetry.timing import percentile_summary  # noqa: E402

CASES = REPO / "eval" / "dev_streaming"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--skip-realtime", action="store_true")
    a = ap.parse_args()
    cfg = load_config(REPO / "configs" / "default.yaml",
                      {"paths.corpus": str(REPO / "tests/fixtures/corpus"), "paths.index_root": str(a.index_root),
                       "telemetry.log_level": "ERROR"}, base_dir=REPO)
    stack = build_stack(cfg, build_index(cfg).path)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    v = run_streaming_benchmark(stack, CASES, ["end_only", "every_chunk", "controller", "controller_prototype"],
                                OUT / "virtual", mode="virtual")
    print("virtual done", round(time.time() - t0, 1), "s", flush=True)

    sens = {}
    for lat in (10, 200, 1000):
        c2 = stack.cfg.model_copy(update={"streaming": stack.cfg.streaming.model_copy(update={"sim_retrieval_latency_ms": lat})})
        st2 = type(stack)(c2, stack.bundle, stack.service, stack.policy)
        m = run_streaming_benchmark(st2, CASES, ["controller", "end_only"], OUT / f"sensitivity_{lat}ms", mode="virtual")
        sens[lat] = {p: {k: m["policies"][p][k] for k in ("early_retrieval_rate", "evidence_ready_at_end_rate",
                                                          "post_final_retrieval_latency_ms", "retrievals_total",
                                                          "cancelled_retrievals")} for p in m["policies"]}
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2, sort_keys=True))
    print("sensitivity done", flush=True)

    if not a.skip_realtime:
        run_streaming_benchmark(stack, CASES, ["end_only", "every_chunk", "controller"], OUT / "realtime", mode="realtime")
        print("realtime done", round(time.time() - t0, 1), "s", flush=True)
        # every realtime trace must replay (virtual) with identical behaviour: decisions, queries, finalizations,
        # retrieval outcomes; exact equality is not expected because realtime timestamps jitter
        parity = {}
        for pol in ("end_only", "every_chunk", "controller"):
            st = stack.with_policy(*POLICIES[pol])
            eng = ReplayEngine(st.cfg, st.service, st.policy, st.index_hash)
            res = {"traces": 0, "behavior_identical": 0, "exact_identical": 0, "mismatched": []}
            for f in sorted((OUT / "realtime" / "traces" / pol).glob("*.jsonl")):
                rep = eng.replay(read_trace(f))
                res["traces"] += 1
                res["behavior_identical"] += rep.behavior_identical
                res["exact_identical"] += rep.identical
                if not rep.behavior_identical:
                    res["mismatched"].append({"trace": f.name, "first": rep.behavior_differences[:1]})
            parity[pol] = res
        (OUT / "realtime_replay_parity.json").write_text(json.dumps(parity, indent=2, default=str))
        print("parity:", {p: f"{r['behavior_identical']}/{r['traces']}" for p, r in parity.items()}, flush=True)

    # profiling pass (realtime, controller, speed 1.0)
    prof = {"chunk_processing_ms": [], "controller_ms": [], "event_emit_ms": [], "retrieval_queue_wait_ms": [],
            "retrieval_wall_ms": [], "query_construction_ms": []}
    lex = Lexicon.load(stack.cfg.controller.lexicon)
    qb = QueryBuilder(lex)
    for case in load_cases(CASES):
        run = asyncio.run(arun_realtime(stack.cfg, stack.service, stack.policy, sim.from_case(case), index_hash=stack.index_hash))
        prof["chunk_processing_ms"] += run.session.chunk_wall_ms
        prof["controller_ms"] += run.session.controller_wall_ms
        prof["event_emit_ms"] += run.session.bus.emit_wall_ms
        for e in run.events:
            if e.type.value == "RETRIEVAL_STARTED":
                prof["retrieval_queue_wait_ms"].append(e.payload["queue_wait_ms"])
            if e.type.value == "RETRIEVAL_COMPLETED":
                prof["retrieval_wall_ms"].append(e.payload["wall"]["measured_ms"])
            if e.type.value == "TRANSCRIPT_UPDATED":
                t = time.perf_counter()
                qb.build(e.payload["transcript"])
                prof["query_construction_ms"].append((time.perf_counter() - t) * 1000.0)
    (OUT / "profile.json").write_text(json.dumps({k: percentile_summary(v) for k, v in prof.items()}, indent=2, sort_keys=True))
    print("profile done", flush=True)

    # one complete replay check on a saved virtual trace
    case = next(c for c in load_cases(CASES) if c.case_id == "wick_height-authored")
    run = run_virtual(stack.cfg, stack.service, stack.policy, sim.from_case(case), index_hash=stack.index_hash)
    dump_trace(run.events, OUT / "example_trace_wick_height.jsonl")
    rep = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash).replay_file(OUT / "example_trace_wick_height.jsonl")
    (OUT / "replay_check.json").write_text(json.dumps(rep.summary(), indent=2, default=str))
    print("replay identical:", rep.identical, "| total", round(time.time() - t0, 1), "s", flush=True)
    print(json.dumps({p: {k: s[k] for k in ("early_retrieval_rate", "false_retrieval_rate", "retrievals_total")}
                      for p, s in v["policies"].items()}, indent=2))


if __name__ == "__main__":
    main()
