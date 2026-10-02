"""Phase 5 measurements on the DEV multi-intent suite (fixture domain; NOT REPORTABLE; NOT held-out).

Outputs (research/phase5/results/):
  offline/          decomposition quality, per-intent retrieval, fusion strategies, ablation arms A-E (final transcripts)
  streaming_virtual/ cases streamed chunk by chunk (virtual clock): delta retrieval, per-intent early retrieval
  dispatch_latency.json  sequential vs parallel vs batched (1-4 intent queries), idle and back-to-back, memory
  ablation_latency.json  A single query | B MI sequential | C MI parallel | D C + fusion | E D + rerank
  profile.json      decomposition / query generation / fusion / rerank / end-to-end p50/p95
  replay_check.json one multi-intent virtual trace replayed exactly + one realtime trace replayed by behaviour

Usage: .venv/bin/python research/phase5/run_multi_intent_benchmarks.py --index-root /tmp/idx5
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"

from streamrag.bench.multi_intent import (  # noqa: E402
    BANNER,
    IntentMatcher,
    aggregate_offline,
    aggregate_streaming,
    evaluate_offline,
    evaluate_streaming,
    load_mi_cases,
    mi_inputs,
    pct,
    write_json,
)
from streamrag.config import load_config  # noqa: E402
from streamrag.fusion import IntentEvidence, make_fusion_engine  # noqa: E402
from streamrag.intents.tracker import IntentTracker  # noqa: E402
from streamrag.models.retrieval import RetrievalOptions, RetrievalRequest  # noqa: E402
from streamrag.multi_retrieval import MultiQueryRetriever  # noqa: E402
from streamrag.replay import ReplayEngine, dump_trace  # noqa: E402
from streamrag.retrieval import build_index  # noqa: E402
from streamrag.retrieval.embedders import load_embedder  # noqa: E402
from streamrag.retrieval.rerank import load_reranker  # noqa: E402
from streamrag.streaming import run_realtime, run_virtual  # noqa: E402
from streamrag.streaming.factory import build_stack  # noqa: E402
from streamrag.streaming.printer import format_event  # noqa: E402

CASES = REPO / "eval" / "dev_multi_intent"
KS = (3, 5, 8)


def _rows_json(rows) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r.__dict__)
        for k in ("intents", "constraint", "relation"):
            d[k] = d[k].prf()
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(REPO / "configs" / "default.yaml",
                      {"paths.corpus": str(REPO / "tests/fixtures/corpus"), "paths.index_root": str(a.index_root),
                       "telemetry.log_level": "ERROR", "multi_intent.enabled": True}, base_dir=REPO)
    stack = build_stack(cfg, build_index(cfg).path)
    b = stack.bundle
    cases = load_mi_cases(CASES)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"REPORTABLE": False, "banner": BANNER, "cases": len(cases), "index_content_hash": stack.index_hash,
            "embedder": b.dense.info.name if b.dense else None}
    t_start = time.time()

    # separate evaluation embedder (spec §22.6) and the cross-encoder for rerank arms
    minilm = load_embedder("all-minilm-l6-v2", cfg.paths.model_registry, cfg.paths.models_dir, b.analyzer,
                           cfg.dense.intra_op_threads, cfg.dense.batch_size)
    matcher = IntentMatcher(stack.intent_stack.query_builder.terms_fn, minilm)
    ce = load_reranker(cfg.rerank.model, cfg.paths.model_registry, cfg.paths.models_dir, cfg.dense.intra_op_threads,
                       cfg.rerank.batch_size)
    fusion_ce = make_fusion_engine(cfg, b, stack.service, reranker=ce)
    rerankers = {"E_intent_ce": ("intent_ce", fusion_ce), "E_cross_intent_ce": ("cross_intent_ce", fusion_ce),
                 "E_cross_intent_dense": ("cross_intent_dense", fusion_ce)}

    # 1. offline evaluation
    rows = evaluate_offline(stack, cases, matcher, KS, "parallel", rerankers)
    off = aggregate_offline(rows, KS)
    write_json(OUT / "offline" / "metrics.json", {**meta, "matcher": matcher.method, "tau_eval": matcher.tau, **off})
    (OUT / "offline" / "rows.jsonl").write_text("\n".join(json.dumps(r, default=str, sort_keys=True)
                                                          for r in _rows_json(rows)) + "\n")
    print("offline done", round(time.time() - t_start, 1), "s", flush=True)

    # 2. streaming (virtual) evaluation
    srows = evaluate_streaming(stack, cases, matcher, "virtual", OUT / "streaming_virtual" / "traces")
    write_json(OUT / "streaming_virtual" / "metrics.json", {**meta, **aggregate_streaming(srows)})
    (OUT / "streaming_virtual" / "rows.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items() if k not in ("decompose_ms", "fusion_ms")}, default=str,
                             sort_keys=True) for r in srows) + "\n")
    print("streaming done", flush=True)

    # 3. dispatch latency: sequential vs parallel vs batched for 1..4 queries (texts from the dev suite)
    texts = [u.utterance_text for c in cases for u in c.utterances]
    tracker = IntentTracker("lat", stack.intent_stack.decomposer, cfg.multi_intent)
    qsets: dict[int, list] = {}
    for n, t in enumerate(texts):
        iset, _, _ = tracker.update(f"u{n}", t, 0.0, final=True)
        qs = [stack.intent_stack.query_builder.build(i, iset.global_constraints + iset.local_constraints)
              for i in iset.intents]
        if 1 <= len(qs) <= 4:
            qsets.setdefault(len(qs), qs)
    q4 = qsets.get(3, []) + qsets.get(1, [])
    if len(qsets.get(4, [])) < 4 and len(q4) >= 4:
        qsets[4] = q4[:4]
    dispatch = {}
    opts = RetrievalOptions(top_k=cfg.multi_intent.max_candidates_per_intent)
    for idle_ms in (0, 300):
        for n, qs in sorted(qsets.items()):
            for mode in ("sequential", "parallel", "batched"):
                mr = MultiQueryRetriever(stack.service, opts, max_concurrent=cfg.multi_intent.max_concurrent_retrievals)
                mr.retrieve(qs, mode)                                  # warm-up
                tot, crit, conc = [], [], []
                for _ in range(a.reps):
                    if idle_ms:
                        time.sleep(idle_ms / 1000.0)
                    r = mr.retrieve(qs, mode)
                    tot.append(r.total_ms)
                    crit.append(r.critical_path_ms)
                    conc.append(r.max_concurrency)
                gc.collect()
                tracemalloc.start()
                rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mr.retrieve(qs, mode)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                dispatch[f"idle{idle_ms}ms/n{n}/{mode}"] = {
                    "n_queries": n, "mode": mode, "idle_before_ms": idle_ms, "total_ms": pct(tot),
                    "critical_path_ms": pct(crit), "max_concurrency": max(conc),
                    "python_heap_peak_kib": round(peak / 1024, 1),
                    "rss_max_kib_growth": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0}
                mr.close()
    write_json(OUT / "dispatch_latency.json", {**meta, "reps": a.reps, "results": dispatch,
                                               "note": "python_heap_peak excludes ONNX native allocations"})
    print("dispatch done", flush=True)

    # 4. ablation latency A-E on compound utterances (idle 300 ms before each, as in live streaming)
    compound = [(c.case_id, u) for c in cases for u in c.utterances
                if len([g for g in u.expected_intents if not g.superseded]) >= 2]
    abl = {k: [] for k in ("A_single_query", "B_mi_sequential", "C_mi_parallel", "D_parallel_fusion",
                           "E_parallel_fusion_rerank")}
    from streamrag.controller.query_builder import QueryBuilder
    single = QueryBuilder(stack.intent_stack.decomposer.lx.base)
    mr = MultiQueryRetriever(stack.service, opts, max_concurrent=cfg.multi_intent.max_concurrent_retrievals)
    for _rep in range(max(1, a.reps // 10)):
        for cid, u in compound:
            tr = IntentTracker(f"abl-{cid}", stack.intent_stack.decomposer, cfg.multi_intent)
            time.sleep(0.3)
            t0 = time.perf_counter()
            stack.service.retrieve(RetrievalRequest(query=single.build(u.utterance_text).text,
                                                    options=RetrievalOptions(top_k=5)))
            abl["A_single_query"].append((time.perf_counter() - t0) * 1000.0)
            time.sleep(0.3)
            t0 = time.perf_counter()
            iset, _, _ = tr.update(u.utterance_id, u.utterance_text, 0.0, final=True)
            qs = [stack.intent_stack.query_builder.build(i, iset.global_constraints + iset.local_constraints)
                  for i in iset.intents]
            t_dec = (time.perf_counter() - t0) * 1000.0
            r_seq = mr.retrieve(qs, "sequential")
            abl["B_mi_sequential"].append(t_dec + r_seq.total_ms)
            time.sleep(0.3)
            r_par = mr.retrieve(qs, "parallel")
            abl["C_mi_parallel"].append(t_dec + r_par.total_ms)
            inputs = [IntentEvidence(i, q.text, None, r_par.by_intent()[i.intent_id].evidence)
                      for i, q in zip(iset.intents, qs)]
            t0 = time.perf_counter()
            stack.intent_stack.fusion.fuse("abl", u.utterance_id, iset.version, inputs, strategy="intent_aware")
            t_f = (time.perf_counter() - t0) * 1000.0
            abl["D_parallel_fusion"].append(t_dec + r_par.total_ms + t_f)
            t0 = time.perf_counter()
            fusion_ce.fuse("abl", u.utterance_id, iset.version, inputs, strategy="intent_aware", rerank="intent_ce")
            abl["E_parallel_fusion_rerank"].append(t_dec + r_par.total_ms + (time.perf_counter() - t0) * 1000.0)
    mr.close()
    write_json(OUT / "ablation_latency.json", {**meta, "utterances": len(compound),
                                               "end_to_end_ms": {k: pct(v) for k, v in abl.items()},
                                               "note": "decomposition + retrieval (+ fusion, + rerank); 300 ms idle "
                                                       "before A, B, C as in live streaming"})
    print("ablation latency done", flush=True)

    # 5. profile from the streaming run (all decomposition updates and fusions) + rerank cost
    dec_ms = [x for r in srows for x in r["decompose_ms"]]
    fus_ms = [x for r in srows for x in r["fusion_ms"]]
    rr = {}
    for name, (mode, eng) in rerankers.items():
        ts = []
        for cid, u in compound:
            tr = IntentTracker(f"rr-{cid}", stack.intent_stack.decomposer, cfg.multi_intent)
            iset, _, _ = tr.update(u.utterance_id, u.utterance_text, 0.0, final=True)
            qs = [stack.intent_stack.query_builder.build(i, iset.global_constraints + iset.local_constraints)
                  for i in iset.intents]
            res = MultiQueryRetriever(stack.service, opts).retrieve(qs, "sequential")
            inputs = [IntentEvidence(i, q.text, None, res.by_intent()[i.intent_id].evidence)
                      for i, q in zip(iset.intents, qs)]
            u2 = eng.fuse("rr", u.utterance_id, 1, inputs, strategy="intent_aware", rerank=mode)
            ts.append(u2.timings_ms.get("rerank", 0.0))
        rr[mode] = pct(ts)
    write_json(OUT / "profile.json", {**meta, "decomposition_update_ms": pct(dec_ms),
                                      "fusion_ms_streaming": pct(fus_ms), "rerank_ms": rr,
                                      "offline": off["latency_ms"]})
    print("profile done", flush=True)

    # 6. replay: virtual exact + realtime behaviour
    case = next(c for c in cases if c.case_id == "C1")
    run = run_virtual(stack.cfg, stack.service, stack.policy, mi_inputs(case), index_hash=stack.index_hash,
                      intent_stack=stack.intent_stack)
    dump_trace(run.events, OUT / "example_trace_C1.jsonl")
    eng = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash, stack.intent_stack)
    rep_v = eng.replay(run.events)
    rt = run_realtime(stack.cfg, stack.service, stack.policy, mi_inputs(case), index_hash=stack.index_hash,
                      intent_stack=stack.intent_stack)
    rep_r = eng.replay(rt.events)
    write_json(OUT / "replay_check.json", {"virtual": rep_v.summary(), "realtime": rep_r.summary()})
    print("replay", rep_v.identical, rep_r.behavior_identical, "| total", round(time.time() - t_start, 1), "s",
          flush=True)

    # 7. human-readable validation traces (single / two / three / incremental / constraint / supersession)
    tdir = Path(__file__).resolve().parent / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    for cid in ("S1", "A1", "B1", "C1", "D3", "L1", "F3"):
        c = next(x for x in cases if x.case_id == cid)
        run = run_virtual(stack.cfg, stack.service, stack.policy, mi_inputs(c), index_hash=stack.index_hash,
                          intent_stack=stack.intent_stack)
        dump_trace(run.events, tdir / f"{cid}.jsonl")
        (tdir / f"{cid}.txt").write_text("NOTE: TEST FIXTURE index - behavior demo only, not a benchmark result.\n"
                                         + "\n".join(x for x in (format_event(e) for e in run.events) if x) + "\n")


if __name__ == "__main__":
    main()
