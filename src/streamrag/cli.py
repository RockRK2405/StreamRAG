"""Command-line interface (argparse; no extra framework).

  streamrag corpus-status                   prints OFFICIAL_CORPUS_STATUS
  streamrag build-index [--force]           corpus -> chunks -> BM25 + dense index + manifest
  streamrag search "query" [--mode ...]     retrieval API -> EvidenceSet (JSON with --json)
  streamrag bench --eval PATH [--modes ...] retrieval benchmark -> runs/<run_id>/
  streamrag export-schemas                  docs/schemas/*.schema.json
  streamrag fetch-models NAME ...           download pinned models (build-time, network)
  streamrag stream --text "a | b | c"       Phase 4 debug stream (controller decisions, retrievals, timings)
  streamrag replay TRACE.jsonl              re-run a saved trace and verify identical events
                                            (virtual: exact; realtime: identical decisions/queries/outcomes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from streamrag.config.settings import load_config
from streamrag.errors import StreamRagError
from streamrag.telemetry.logging import configure_logging


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got '{p}'")
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def _cfg(args: argparse.Namespace):
    ov = _overrides(args.set)
    if getattr(args, "corpus", None):
        ov["paths.corpus"] = args.corpus
    if getattr(args, "embedder", None):
        ov["dense.embedder"] = args.embedder
    cfg = load_config(args.config, ov)
    configure_logging(cfg.telemetry.log_level, cfg.telemetry.log_format)
    return cfg


def cmd_corpus_status(args: argparse.Namespace) -> int:
    from streamrag.corpus.source import CorpusSource
    cfg = _cfg(args)
    src = CorpusSource(cfg.paths.corpus, cfg.corpus.include_extensions, cfg.corpus.fixture_marker)
    status = src.status()
    official = "AVAILABLE" if status == "AVAILABLE" else "NOT_AVAILABLE"
    print(f"OFFICIAL_CORPUS_STATUS = {official}")
    print(f"corpus_path = {cfg.paths.corpus}")
    print(f"detected = {status}")
    if status in ("AVAILABLE", "TEST_FIXTURE"):
        entries = src.entries()
        print(f"files = {len(entries)}  corpus_hash = {src.corpus_hash(entries)[:16]}")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from streamrag.retrieval.store import build_index
    cfg = _cfg(args)
    b = build_index(cfg, force=args.force)
    m = b.manifest
    if m.is_test_fixture:
        print("NOTE: built from a TEST FIXTURE corpus - not the official corpus.")
    skipped = [f for f in m.source_files if f.status != "loaded"]
    print(json.dumps({"index_path": str(b.path), "is_test_fixture": m.is_test_fixture, "documents": m.document_count,
                      "sections": m.section_count, "chunks": m.chunk_count, "skipped_files": len(skipped),
                      "embedding": m.embedding_model.name if m.embedding_model else None,
                      "truncated_chunks": m.embedding_model.truncated_chunks if m.embedding_model else None,
                      "content_hash": m.content_hash[:16], "timings_ms": {k: round(v, 1) for k, v in b.timings_ms.items()}},
                     indent=2))
    for f in skipped:
        print(f"skipped: {f.path} ({f.reason})", file=sys.stderr)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from streamrag.models.retrieval import RetrievalOptions
    from streamrag.retrieval.service import RetrievalService
    cfg = _cfg(args)
    svc = RetrievalService.from_config(cfg, Path(args.index) if args.index else None, load_rerank=args.rerank or None)
    es = svc.retrieve(args.query, RetrievalOptions(mode=args.mode, top_k=args.top_k, rerank=args.rerank or None))
    if args.json:
        print(es.model_dump_json(indent=2))
    else:
        print(f"[{es.trace.status}] mode={es.trace.mode} rerank={es.trace.rerank_applied} "
              f"total={es.trace.timings_ms.get('total', 0):.1f} ms warnings={es.trace.warnings}")
        for e in es.items:
            print(f"{e.rank:2d}. {e.citation:28s} score={e.score:.4f} bm25#{e.bm25_rank} dense#{e.dense_rank} "
                  f"| {e.text[:90]!r}")
    svc.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from streamrag.bench.harness import run_benchmark
    cfg = _cfg(args)
    run_id = args.run_id or time.strftime("bench-%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else cfg.paths.runs_dir / run_id
    metrics = run_benchmark(cfg, Path(args.eval), args.modes, out, Path(args.index) if args.index else None,
                            command=" ".join(sys.argv))
    if not metrics["REPORTABLE"]:
        print(f"*** {metrics['banner']} ({metrics['not_reportable_reason']}) ***")
    print(json.dumps({m: v["metrics"] for m, v in metrics["modes"].items()}, indent=2))
    print(f"outputs: {out}")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    from streamrag.models.events import SessionEnd, SessionStart
    from streamrag.streaming import run_realtime, run_virtual
    from streamrag.streaming import simulator as sim
    from streamrag.streaming.factory import build_stack
    from streamrag.streaming.printer import format_event
    ov = {}
    if args.policy:
        ov["controller.strategy"] = args.policy
    if args.mode:
        ov["streaming.mode"] = args.mode
    if args.multi_intent or args.session:
        ov["multi_intent.enabled"] = "true"
    if args.session:
        ov["session.enabled"] = "true"
    args.set = list(args.set) + [f"{k}={v}" for k, v in ov.items()]
    cfg = _cfg(args)
    stack = build_stack(cfg, Path(args.index) if args.index else None)
    if args.input:
        inputs = sim.read_inputs(Path(args.input))
    else:
        inputs, offset = [SessionStart(session_id=args.session_id)], 0.0
        for i, utt in enumerate(args.text or [], start=1):
            chunks = sim.parse_inline(utt)
            evs = (sim.stream(chunks, args.interval_ms, cfg.streaming.end_gap_ms, args.session_id, f"u{i}", offset, False)
                   if args.interval_ms else
                   sim.timed(chunks, cfg.streaming.words_per_second, cfg.streaming.end_gap_ms, 0.0, 0, args.session_id,
                             f"u{i}", offset, False))
            inputs += evs
            offset += evs[-1].payload.timestamp_s * 1000.0 + args.gap_ms
        inputs.append(SessionEnd(session_id=args.session_id))
    runner = run_realtime if cfg.streaming.mode == "realtime" else run_virtual
    run = runner(cfg, stack.service, stack.policy, inputs, Path(args.trace) if args.trace else None, stack.index_hash,
                 stack.intent_stack if cfg.multi_intent.enabled else None)
    if stack.bundle.manifest.is_test_fixture:
        print("NOTE: TEST FIXTURE index - behavior demo only, not a benchmark result.")
    for ev in run.events:
        line = format_event(ev)
        if line:
            print(line)
    if args.trace:
        print(f"trace: {args.trace} ({len(run.events)} events)")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from streamrag.replay import ReplayEngine
    from streamrag.streaming.factory import build_stack
    from streamrag.replay import read_trace
    trace = read_trace(Path(args.trace))
    mi = any(e.type.value == "SESSION_STARTED" and e.payload.get("multi_intent") for e in trace)
    adaptive = any(e.type.value == "SESSION_STARTED" and e.payload.get("session_mode") for e in trace)
    if mi:                                     # replay in the mode the trace was recorded in
        args.set = list(args.set) + ["multi_intent.enabled=true"]
    if adaptive:
        args.set = list(args.set) + ["session.enabled=true"]
    cfg = _cfg(args)
    stack = build_stack(cfg, Path(args.index) if args.index else None)
    report = ReplayEngine(cfg, stack.service, stack.policy, stack.index_hash,
                          stack.intent_stack if mi else None).replay(trace)
    print(json.dumps(report.summary(), indent=2, default=str))
    return 0 if report.ok else 1


def cmd_export_schemas(args: argparse.Namespace) -> int:
    from streamrag.models.schemas import export_schemas
    for p in export_schemas(Path(args.out)):
        print(p)
    return 0


def cmd_fetch_models(args: argparse.Namespace) -> int:
    from streamrag.tools.fetch_models import fetch
    cfg = _cfg(args)
    for p in fetch(args.names, cfg.paths.model_registry, cfg.paths.models_dir):
        print(f"fetched {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="streamrag", description="StreamRAG retrieval foundation (Phase 3)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("corpus-status"); s.add_argument("--corpus"); s.set_defaults(fn=cmd_corpus_status)
    s = sub.add_parser("build-index"); s.add_argument("--corpus"); s.add_argument("--embedder")
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_build_index)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--corpus"); s.add_argument("--index")
    s.add_argument("--mode", choices=["bm25", "dense", "hybrid"]); s.add_argument("--top-k", type=int, default=None)
    s.add_argument("--rerank", action="store_true"); s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("bench"); s.add_argument("--eval", required=True); s.add_argument("--corpus")
    s.add_argument("--index"); s.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid", "hybrid_rerank"])
    s.add_argument("--out"); s.add_argument("--run-id"); s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("export-schemas"); s.add_argument("--out", default="docs/schemas"); s.set_defaults(fn=cmd_export_schemas)
    s = sub.add_parser("stream"); s.add_argument("--text", action="append", help='utterance, chunks separated by "|"')
    s.add_argument("--input", help="JSONL of input events"); s.add_argument("--corpus"); s.add_argument("--index")
    s.add_argument("--embedder"); s.add_argument("--policy", choices=["rules", "end_only", "every_chunk"])
    s.add_argument("--mode", choices=["virtual", "realtime"]); s.add_argument("--interval-ms", type=float, default=None)
    s.add_argument("--gap-ms", type=float, default=1500.0); s.add_argument("--session-id", default="cli")
    s.add_argument("--trace"); s.add_argument("--multi-intent", action="store_true",
                                              help="Phase 5: decompose intents, retrieve per intent, fuse evidence")
    s.add_argument("--session", action="store_true",
                   help="Phase 6: adaptive session (late details, delta retrieval, claims, answer versions)")
    s.set_defaults(fn=cmd_stream)
    s = sub.add_parser("replay"); s.add_argument("trace"); s.add_argument("--corpus"); s.add_argument("--index")
    s.add_argument("--embedder"); s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("fetch-models"); s.add_argument("names", nargs="+"); s.set_defaults(fn=cmd_fetch_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except StreamRagError as exc:
        print(f"error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
