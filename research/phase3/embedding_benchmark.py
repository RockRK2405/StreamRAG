"""Embedding runtime benchmark: ONNX Runtime vs PyTorch (sentence-transformers), CPU only.

Each configuration runs in a fresh subprocess (clean load time + peak RSS). Text is SYNTHETIC (synth.py), so
this measures infrastructure behavior only - NOT retrieval quality (blocked: no official corpus/labels).

Usage:  .venv/bin/python research/phase3/embedding_benchmark.py            # writes results/embedding_benchmark.json
"""

from __future__ import annotations

import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
OUT = HERE / "results"

CONFIGS = [
    {"label": "bge-small-en-v1.5 / onnx-fp32", "runtime": "onnx", "name": "bge-small-en-v1.5"},
    {"label": "all-MiniLM-L6-v2 / onnx-fp32", "runtime": "onnx", "name": "all-minilm-l6-v2"},
    {"label": "all-MiniLM-L6-v2 / onnx-qint8-arm64", "runtime": "onnx", "name": "all-minilm-l6-v2-qint8-arm64"},
    {"label": "bge-small-en-v1.5 / pytorch", "runtime": "torch", "repo": "BAAI/bge-small-en-v1.5",
     "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
     "qp": "Represent this sentence for searching relevant passages: ", "dp": ""},
    {"label": "all-MiniLM-L6-v2 / pytorch", "runtime": "torch", "repo": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "qp": "", "dp": ""},
    {"label": "e5-small-v2 / pytorch (no fp32 ONNX published)", "runtime": "torch", "repo": "intfloat/e5-small-v2",
     "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52", "qp": "query: ", "dp": "passage: "},
]
THREADS = [0, 2]   # 0 = runtime default; 2 = container-like proxy


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3   # macOS: bytes, Linux: KiB


def worker(cfg: dict) -> dict:
    import synth
    passages = synth.passages(1000, words=180, seed=0)
    queries = synth.queries(60, seed=1)
    probes = synth.passages(16, words=120, seed=7)
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    if cfg["runtime"] == "onnx":
        from streamrag.retrieval.embedders import load_embedder
        from streamrag.retrieval.text import Analyzer
        emb = load_embedder(cfg["name"], REPO / "configs" / "models.yaml", REPO / "models", Analyzer(),
                            intra_op_threads=cfg["threads"], batch_size=32)
        def enc(texts, kind):
            return emb.embed(texts, kind)
        dim = emb.dimension
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        if cfg["threads"]:
            torch.set_num_threads(cfg["threads"])
        st = SentenceTransformer(cfg["repo"], revision=cfg["revision"], device="cpu")
        def enc(texts, kind):
            pre = cfg["qp"] if kind == "query" else cfg["dp"]
            return st.encode([pre + t for t in texts], batch_size=32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
        dim = st.get_sentence_embedding_dimension()
    load_s = time.perf_counter() - t0
    enc(queries[:2], "query")
    enc(passages[:8], "document")                      # warm-up
    q_lat = []
    for q in queries[:50]:
        t = time.perf_counter(); enc([q], "query"); q_lat.append((time.perf_counter() - t) * 1000)
    b_lat = []
    for i in range(5):
        batch = passages[i * 32:(i + 1) * 32]
        t = time.perf_counter(); enc(batch, "document"); b_lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); enc(passages, "document"); idx_s = time.perf_counter() - t
    np.save(OUT / f"probe_{cfg['key']}.npy", enc(probes, "document"))
    return {"load_s": round(load_s, 3), "dim": int(dim),
            "query_ms_p50": round(statistics.median(q_lat), 2), "query_ms_p95": round(float(np.percentile(q_lat, 95)), 2),
            "batch32_ms_p50": round(statistics.median(b_lat), 1),
            "passages_per_s": round(len(passages) / idx_s, 1), "index_1000_s": round(idx_s, 2),
            "peak_rss_mb": round(_rss_mb(), 1), "rss_before_load_mb": round(rss_before, 1)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results = []
    for c in CONFIGS:
        for th in THREADS:
            cfg = {**c, "threads": th, "key": f"{c['label'].split(' /')[0]}_{c['runtime']}_{c.get('name', '')}_{th}".replace("/", "-")}
            print(f"running {c['label']} threads={th or 'default'} ...", flush=True)
            proc = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)], capture_output=True, text=True)
            if proc.returncode != 0:
                results.append({**cfg, "error": proc.stderr[-800:]})
                print("  ERROR", proc.stderr[-300:])
                continue
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            results.append({**cfg, **r})
            print("  ", r, flush=True)
    # ONNX vs PyTorch numeric agreement (validates our ONNX pooling/normalization/prefix pipeline)
    agree = {}
    for model, onnx_key_part, torch_label in [("bge-small-en-v1.5", "bge-small-en-v1.5", "bge-small-en-v1.5"),
                                              ("all-MiniLM-L6-v2", "all-minilm-l6-v2", "all-MiniLM-L6-v2")]:
        o = next(r for r in results if r["runtime"] == "onnx" and r.get("name") == onnx_key_part and r["threads"] == 0)
        t = next(r for r in results if r["runtime"] == "torch" and r["label"].startswith(torch_label) and r["threads"] == 0)
        a, b = np.load(OUT / f"probe_{o['key']}.npy"), np.load(OUT / f"probe_{t['key']}.npy")
        cos = (a * b).sum(axis=1)
        agree[model] = {"min_cosine": round(float(cos.min()), 6), "mean_cosine": round(float(cos.mean()), 6)}
    q = next(r for r in results if r.get("name") == "all-minilm-l6-v2-qint8-arm64" and r["threads"] == 0)
    f = next(r for r in results if r.get("name") == "all-minilm-l6-v2" and r["threads"] == 0)
    a, b = np.load(OUT / f"probe_{q['key']}.npy"), np.load(OUT / f"probe_{f['key']}.npy")
    agree["all-MiniLM-L6-v2 qint8 vs fp32 (ONNX)"] = {"min_cosine": round(float((a * b).sum(1).min()), 6),
                                                      "mean_cosine": round(float((a * b).sum(1).mean()), 6)}
    import platform
    from streamrag.provenance import package_versions
    payload = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": platform.platform(),
               "packages": package_versions(), "text": "synthetic (research/phase3/synth.py), 180-word passages",
               "results": results, "agreement": agree}
    (OUT / "embedding_benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    for p in OUT.glob("probe_*.npy"):
        p.unlink()
    print(json.dumps(agree, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        print(json.dumps(worker(json.loads(sys.argv[2]))))
    else:
        main()
