# Phase-2 assumption check: CPU cost of a MiniLM-L6 cross-encoder rerank pass.
# Uses the cached all-MiniLM-L6-v2 weights as a compute stand-in for cross-encoder/ms-marco-MiniLM-L-6-v2
# (same BERT shape: 6 layers, hidden 384; the cross-encoder only adds a 384->1 head). Latency only, synthetic text.
import os, statistics, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from embed_latency_bench import Enc  # minimal torch BERT forward pass (Phase 1)

threads = int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads(); torch.set_num_threads(threads)
m = Enc("sentence-transformers/all-MiniLM-L6-v2")
q = "cancellation policy workshop venue Pune"
p = ("This section describes the general procedure that applies to bookings made by employees. "
     "Requests must be submitted through the standard process and approved before the event date. ") * 4
pair = q + " [SEP] " + p          # ~ query + passage pair, as a cross-encoder sees it
ntok = len(m.tok.encode(pair).ids)
for n in (10, 20, 30):
    m.encode([pair] * n)  # warmup
    lat = []
    for _ in range(7):
        t = time.perf_counter(); m.encode([pair] * n); lat.append((time.perf_counter() - t) * 1000)
    print(f"threads={threads} pairs={n} tokens/pair={ntok} p50={statistics.median(lat):.0f}ms min={min(lat):.0f}ms")
