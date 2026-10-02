# Phase 3 — Embedding Runtime Benchmark (ONNX vs PyTorch)

- **Script:** `research/phase3/embedding_benchmark.py`
- **Raw results:** `results/embedding_benchmark.json`
- **Date:** 2026-10-02
- **Machine:** Apple M5 Pro, macOS 26.5.1 arm64, Python 3.12.0 (dev machine)
- **Libraries:** onnxruntime 1.30.0, tokenizers 0.23.2, torch 2.14.1, sentence-transformers 6.1.0, transformers 5.18.0

> **Retrieval quality is NOT AVAILABLE.** The official Theme 4 corpus and gold labels do not exist yet. The text used here is **synthetic** (`synth.py`: Zipf-sampled words from the fictional test fixtures, 180-word passages). This report measures runtime behavior only. Model choice on quality grounds remains blocked (Exp 1).

## Method

Each configuration runs in a fresh subprocess, so model load time and peak RSS are attributable to it. The steps are:

1. Load the model.
2. Warm up.
3. Time 50 single-query encodes.
4. Time 5 batches of 32 passages.
5. Encode 1,000 passages, for throughput and the 1k index time.
6. Read peak RSS (`ru_maxrss`).

Two thread settings were used: the runtime default, and **2 threads** as a container-like proxy. Both runtimes use the same pinned revisions (`configs/models.yaml`). PyTorch rows for bge and e5 were re-measured after the first run, because that run's "load" included the one-time weight download.

## MEASURED

### Default threads

| Model / runtime | Load (s) | Query p50 (ms) | Query p95 (ms) | Batch-32 p50 (ms) | Passages/s | 1k passages (s) | Peak RSS (MB) | Dim |
|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 / **ONNX fp32** | 0.32 | **2.13** | 2.31 | 300 | **108.5** | 9.2 | 1120 | 384 |
| all-MiniLM-L6-v2 / ONNX fp32 | 0.20 | 0.97 | 1.07 | 161 | 214.3 | 4.7 | 1053 | 384 |
| all-MiniLM-L6-v2 / ONNX qint8-arm64 | 0.20 | 0.84 | 1.00 | 177 | 189.7 | 5.3 | 758 | 384 |
| bge-small-en-v1.5 / PyTorch | 3.09 | 8.55 | 8.86 | 366 | 87.3 | 11.5 | 1002 | 384 |
| all-MiniLM-L6-v2 / PyTorch | 10.0 (cold import) | 4.12 | 4.61 | 190 | 170.2 | 5.9 | 958 | 384 |
| e5-small-v2 / PyTorch (no fp32 ONNX published) | 3.75 | 8.09 | 8.58 | 373 | 86.2 | 11.6 | 1016 | 384 |

### 2 intra-op threads (container proxy)

| Model / runtime | Load (s) | Query p50 (ms) | Query p95 (ms) | Batch-32 p50 (ms) | Passages/s | Peak RSS (MB) |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 / ONNX fp32 | 0.24 | 2.71 | 2.89 | 609 | 55.1 | 1113 |
| all-MiniLM-L6-v2 / ONNX fp32 | 0.20 | 1.21 | 1.34 | 289 | 113.2 | 1064 |
| all-MiniLM-L6-v2 / ONNX qint8-arm64 | 0.20 | 0.64 | 0.75 | 307 | 106.4 | 765 |
| bge-small-en-v1.5 / PyTorch | 3.11 | 8.00 | 8.37 | 505 | 65.0 | 1010 |
| all-MiniLM-L6-v2 / PyTorch | 3.12 | 3.88 | 4.28 | 250 | 124.1 | 952 |
| e5-small-v2 / PyTorch | 4.01 | 7.70 | 8.20 | 498 | 64.3 | 1010 |

### Numerical agreement (correctness of our ONNX pipeline)

Cosine similarity between our `OnnxEmbedder` output and `sentence-transformers` output for the same 16 probe passages:

| Pair | Min cosine | Mean cosine |
|---|---|---|
| bge-small-en-v1.5: ONNX vs PyTorch | 1.000000 | 1.000000 |
| all-MiniLM-L6-v2: ONNX vs PyTorch | 1.000000 | 1.000000 |
| all-MiniLM-L6-v2: ONNX qint8 vs ONNX fp32 | 0.994876 | 0.995989 |

Our tokenization, pooling (CLS for bge, mean for MiniLM, read from each model's `1_Pooling` config), normalization and prefixes reproduce the reference implementation exactly. The int8 variant shifts vectors slightly; how that affects retrieval quality is not measurable without labels.

### Build-time memory vs batch size

From `profile_retrieval.py`, bge-small ONNX, ~2,000 synthetic chunks of ~100 tokens:

| Batch size | Chunks/s | Peak RSS (MB) |
|---|---|---|
| 32 | 185.2 | 1093 |
| **8** | **187.9** | **589** |

**Decision applied:** `dense.batch_size` default changed from 32 to **8**. Throughput is the same and peak memory is roughly halved. Disabling the ONNX Runtime memory arena was also measured, and it was *worse*: query-process peak RSS rose from 1078 to 1865 MB and rerank latency increased. The arena stays on.

## ESTIMATED

- **Judge-hardware latency.** Unknown. A typical x86 judge CPU is likely slower than an M5 Pro, and the 2-thread rows are the closest proxy we have. Re-measure inside the container on the target machine.
- **bge-base-en-v1.5** was not measured, because we did not download it. By architecture (12 layers, 768 hidden, about 3× the parameters of bge-small), expect roughly 3× the bge-small cost. Download it only if Exp 1 shows bge-small is insufficient.

## NOT AVAILABLE

- Retrieval quality of any model (Recall@k, MRR) on the Theme 4 domain. **Blocked** by the missing corpus and gold labels.
- The quality impact of int8 quantization.
- The ONNX fp32 runtime for e5-small-v2. Upstream publishes only an O4 (fp16) and an x86 int8 export, and we did not export one ourselves. It is measured under PyTorch only.

## Conclusions

1. **Runtime: ONNX Runtime (CPU EP) is selected.** This confirms ADR-002 with evidence. Against PyTorch it gives:
   - 4× lower query latency for bge-small (2.1 vs 8.6 ms);
   - about 25% higher throughput;
   - load time 10× faster (0.3 vs 3.1 s);
   - no `torch` in the runtime image, since torch is only in the research-only `bench-torch` extra.
2. **Default model: bge-small-en-v1.5 is kept, provisionally.** It is retrieval-trained, has a 512-token window, and all observed chunks fit it (0 truncations, see `chunking_report.md`). Its query cost (~2–3 ms) is negligible against the streaming budget. MiniLM is about 2× cheaper but has a 256-token window and is trained for symmetric similarity. **The final choice needs Exp 1 on real labels.**
3. **Int8 MiniLM is rejected.** Its latency is the same as fp32 within noise at default threads, so there is no reason to accept an unmeasured quality risk.
4. **e5-small-v2** offers nothing over bge-small on runtime (same cost under PyTorch), and it needs our own ONNX export. It stays an optional Exp 1 arm.
