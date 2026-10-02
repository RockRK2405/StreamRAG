"""Retrieval-stack performance profile on SYNTHETIC corpora (latency/memory only; no quality claims).

Stages: corpus loading, chunking, BM25 indexing, embedding indexing, index loading, BM25 query, query embedding,
dense query, fusion, dedup, reranking, end-to-end per mode. Each scenario runs in a fresh subprocess so peak RSS
is attributable. Usage:
  .venv/bin/python research/phase3/profile_retrieval.py --scratch <dir> [--sizes 135 670]
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1e6 if sys.platform == "darwin" else r / 1e3, 1)


def cfg_for(corpus: Path, index_root: Path, threads: int = 0, batch: int = 32, rerank_model: str | None = None):
    from streamrag.config import load_config
    ov = {"paths.corpus": str(corpus), "paths.index_root": str(index_root), "telemetry.log_level": "ERROR",
          "dense.intra_op_threads": threads, "dense.batch_size": batch}
    if rerank_model:
        ov["rerank.model"] = rerank_model
    return load_config(REPO / "configs" / "default.yaml", ov, base_dir=REPO)


def do_build(args) -> dict:
    from streamrag.retrieval import build_index
    cfg = cfg_for(Path(args.corpus), Path(args.index_root), batch=args.batch)
    t = time.perf_counter()
    b = build_index(cfg, force=True)
    total = time.perf_counter() - t
    sizes = {p.relative_to(b.path).as_posix(): p.stat().st_size for p in b.path.rglob("*") if p.is_file()}
    return {"chunks": len(b.chunks), "documents": b.manifest.document_count, "build_total_s": round(total, 2),
            "timings_ms": {k: round(v, 1) for k, v in b.timings_ms.items()},
            "embedding_chunks_per_s": round(len(b.chunks) / (b.timings_ms["embedding_build"] / 1000), 1),
            "truncated_chunks": b.manifest.embedding_model.truncated_chunks,
            "index_bytes": sizes, "index_total_mb": round(sum(sizes.values()) / 1e6, 2),
            "peak_rss_mb": rss_mb(), "index_path": str(b.path), "batch_size": args.batch}


def do_query(args) -> dict:
    import synth
    from streamrag.models import RetrievalOptions
    from streamrag.retrieval import RetrievalService
    from streamrag.retrieval.store import load_index
    from streamrag.telemetry.timing import Stopwatch, percentile_summary
    cfg = cfg_for(Path(args.corpus), Path(args.index_root), threads=args.threads, rerank_model=args.rerank_model)
    with Stopwatch() as sw:
        load_index(Path(args.index))
    index_load_ms = sw.ms
    with Stopwatch() as sw:
        svc = RetrievalService.from_config(cfg, index_path=Path(args.index), load_rerank=True)
    service_init_ms = sw.ms
    queries = synth.queries(args.n_queries, seed=11)
    out = {"index_load_ms": round(index_load_ms, 1), "service_init_ms(models+index)": round(service_init_ms, 1),
           "threads": args.threads or "default", "rerank_model": cfg.rerank.model, "modes": {}}
    for name, opts in {"bm25": RetrievalOptions(mode="bm25"), "dense": RetrievalOptions(mode="dense"),
                       "hybrid": RetrievalOptions(mode="hybrid"),
                       "hybrid_rerank": RetrievalOptions(mode="hybrid", rerank=True, rerank_k=20)}.items():
        for q in queries[:5]:
            svc.retrieve(q, opts)                                   # warm-up
        stages: dict[str, list[float]] = {}
        for q in queries:
            es = svc.retrieve(q, opts)
            for k, v in es.trace.timings_ms.items():
                stages.setdefault(k, []).append(v)
        out["modes"][name] = {k: percentile_summary(v) for k, v in sorted(stages.items())}
    svc.close()
    out["peak_rss_mb"] = rss_mb()
    return out


def run(cmd: list[str]) -> dict:
    p = subprocess.run([sys.executable, __file__, *cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[135, 670])   # docs -> ~2k / ~10k chunks
    a = ap.parse_args()
    import synth
    from streamrag.provenance import environment
    results = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "environment": environment(),
               "note": "SYNTHETIC corpora (TEST_FIXTURE_ONLY); latency/memory only", "runs": []}
    for n_docs in a.sizes:
        corpus = a.scratch / f"synth_{n_docs}"
        if corpus.exists():
            shutil.rmtree(corpus)
        synth.make_corpus(corpus, n_docs=n_docs, seed=5)
        idx_root = a.scratch / f"idx_{n_docs}"
        print(f"build {n_docs} docs ...", flush=True)
        build = run(["build", "--corpus", str(corpus), "--index-root", str(idx_root), "--batch", "32"])
        print("  ", {k: build[k] for k in ("chunks", "build_total_s", "embedding_chunks_per_s", "index_total_mb", "peak_rss_mb")}, flush=True)
        entry = {"n_docs": n_docs, "build": build, "build_batch8": None, "query": []}
        if n_docs == a.sizes[0]:
            b8 = run(["build", "--corpus", str(corpus), "--index-root", str(a.scratch / f"idx_{n_docs}_b8"), "--batch", "8"])
            entry["build_batch8"] = {k: b8[k] for k in ("embedding_chunks_per_s", "peak_rss_mb", "batch_size")}
            print("   batch8:", entry["build_batch8"], flush=True)
        for threads, rmodel in [(0, "ms-marco-minilm-l6-v2"), (2, "ms-marco-minilm-l6-v2"),
                                (2, "ms-marco-minilm-l6-v2-qint8-arm64")]:
            q = run(["query", "--corpus", str(corpus), "--index-root", str(idx_root), "--index", build["index_path"],
                     "--threads", str(threads), "--rerank-model", rmodel, "--n-queries", "200"])
            entry["query"].append(q)
            print(f"   query threads={threads or 'default'} rerank={rmodel}: "
                  f"hybrid total p50={q['modes']['hybrid']['total']['p50']} ms, "
                  f"+rerank p50={q['modes']['hybrid_rerank']['total']['p50']} ms, rss={q['peak_rss_mb']} MB", flush=True)
        results["runs"].append(entry)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "retrieval_profile.json").write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("build", "query"):
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--corpus")
        ap.add_argument("--index-root")
        ap.add_argument("--index")
        ap.add_argument("--batch", type=int, default=32)
        ap.add_argument("--threads", type=int, default=0)
        ap.add_argument("--rerank-model")
        ap.add_argument("--n-queries", type=int, default=200)
        args = ap.parse_args()
        print(json.dumps(do_build(args) if args.cmd == "build" else do_query(args)))
    else:
        main()
