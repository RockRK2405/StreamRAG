"""Resource profile (brief §22, §60): memory and CPU of the in-process components, sizes on disk, and the LLM
server's footprint. Everything is measured on this machine and nothing is estimated.

  memory    peak resident set size (ru_maxrss) after each loading step; the increments approximate each model's
            resident footprint. The value is a process peak, so it never decreases.
  CPU       per-call process CPU time and wall time (median and p90 over 30 test questions) for dense, hybrid and
            hybrid + rerank retrieval, and for the claim verification of one claim against five sections
  disk      index directory and model files
  LLM       `ollama ps` (resident size, CPU / GPU split) and the server process RSS

Usage: .venv/bin/python experiments/runners/resources.py --index-root /tmp/idx10   (run alone; `ollama serve` up)
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpora import CORPORA  # noqa: E402


def rss_mb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(v / (1024 * 1024) if sys.platform == "darwin" else v / 1024, 1)      # bytes on macOS, KiB on Linux


def du_mb(p: Path) -> float | None:
    if not p.exists():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024), 2)


def timed(fn, n: int) -> dict:
    cpu, wall = [], []
    for i in range(n):
        c0, w0 = time.process_time(), time.perf_counter()
        fn(i)
        cpu.append((time.process_time() - c0) * 1000.0)
        wall.append((time.perf_counter() - w0) * 1000.0)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(round((len(v) - 1) * p)))]  # noqa: E731
    return {"n": n, "cpu_ms_median": round(statistics.median(cpu), 2), "cpu_ms_p90": round(q(cpu, 0.9), 2),
            "wall_ms_median": round(statistics.median(wall), 2), "wall_ms_p90": round(q(wall, 0.9), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    a = ap.parse_args()
    out: dict = {"meta": {"note": "measured on this machine; process = the evaluation process (not the LLM server)"},
                 "memory_peak_rss_mb": {"python + imports": None}}
    mem = out["memory_peak_rss_mb"]
    from streamrag.evaluation.dataset import load
    from streamrag.evaluation.systems import Stacks
    from streamrag.models.retrieval import RetrievalOptions
    mem["python + imports"] = rss_mb()
    stacks = Stacks(REPO, CORPORA, a.index_root, rerank=False)
    st = stacks.get("transit", "extractive")
    mem["+ index, BM25, bge-small embedder, NLI (stack)"] = rss_mb()
    stacks_r = Stacks(REPO, CORPORA, a.index_root, rerank=True)
    stacks_r._nli = stacks.nli()
    st_r = stacks_r.get("transit", "extractive")
    mem["+ cross-encoder reranker (second stack)"] = rss_mb()
    qs = [s.query for s in load(REPO / "experiments" / "datasets" / "streamrag_eval_v1" / "test.jsonl")][:30]
    n = len(qs)
    cpu = {}
    cpu["dense top-5"] = timed(lambda i: st.service.retrieve(qs[i], RetrievalOptions(mode="dense", top_k=5)), n)
    cpu["hybrid top-5"] = timed(lambda i: st.service.retrieve(qs[i], RetrievalOptions(mode="hybrid", top_k=5)), n)
    cpu["hybrid top-5 + rerank"] = timed(
        lambda i: st_r.service.retrieve(qs[i], RetrievalOptions(mode="hybrid", top_k=5, rerank=True)), n)
    from run_experiments import instrument_factory
    inst = instrument_factory(stacks)("transit")
    chunks = st.bundle.chunks
    claims = [s.expected_claims[0] for s in load(REPO / "experiments" / "datasets" / "streamrag_eval_v1" / "test.jsonl")
              if s.expected_claims][:30]
    pool = lambda i: {c.chunk_id: c.text for c in chunks[i % len(chunks):i % len(chunks) + 5]}  # noqa: E731
    cpu["verify 1 claim vs 5 sections"] = timed(lambda i: inst.verifier.verify(f"r{i}", claims[i].text, [], pool(i)),
                                               len(claims))
    mem["+ verification instrument"] = rss_mb()
    out["cpu_per_call"] = cpu
    out["disk_mb"] = {"index (transit)": du_mb(a.index_root / "transit"),
                      **{f"model {d.name}": du_mb(d) for d in sorted((REPO / "models").iterdir()) if d.is_dir()}}
    try:
        out["llm_server"] = {"ollama_ps": subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                                                         timeout=10).stdout.strip()}
        pids = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True).stdout.split()
        out["llm_server"]["process_rss_mb"] = {
            p: round(int(subprocess.run(["ps", "-o", "rss=", "-p", p], capture_output=True, text=True).stdout.strip()
                         or 0) / 1024, 1) for p in pids}
        runners = subprocess.run(["pgrep", "-f", "ollama runner"], capture_output=True, text=True).stdout.split()
        out["llm_server"]["runner_rss_mb"] = {
            p: round(int(subprocess.run(["ps", "-o", "rss=", "-p", p], capture_output=True, text=True).stdout.strip()
                         or 0) / 1024, 1) for p in runners}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        out["llm_server"] = {"status": "NOT MEASURED", "error": repr(exc)}
    out["gpu_memory"] = "NOT MEASURED (no discrete GPU; Ollama uses Apple Metal unified memory, see ollama_ps)"
    d = REPO / "experiments" / "results" / "RESOURCES"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
