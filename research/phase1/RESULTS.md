# Phase 1 micro-benchmark results (raw)

Run on 2026-10-02 on the dev machine: Apple M5 Pro (15 cores), 24 GB RAM, macOS 26.5.1 arm64,
Python 3.12.0, torch 2.2.2 (CPU), numpy 1.26.4, scikit-learn 1.6.1.

These numbers measure **latency only**, on **synthetic text**. They say nothing about retrieval quality.
The dev machine is much faster than a typical evaluation host, so treat them as lower bounds on latency
and re-measure inside the container on target hardware.

## embed_latency_bench.py

Minimal BERT/XLM-R forward pass in plain torch, loading locally cached safetensors (the global env's
transformers 5.9 refuses torch 2.2.2, so sentence-transformers could not be used). Mean pooling + L2 norm.

```
torch 2.2.2 threads=15
sentence-transformers/all-MiniLM-L6-v2: params=23M layers=6 hidden=384 load=0.01s sanity(paraphrase=0.720 > unrelated=0.027) query_p50=4.2ms query_p95=5.5ms passages/s@122tok=207.7
intfloat/multilingual-e5-base: params=278M layers=12 hidden=768 load=0.27s sanity(paraphrase=0.847 > unrelated=0.675) query_p50=26.0ms query_p95=29.7ms passages/s@140tok=34.9
=== 2 threads ===
sentence-transformers/all-MiniLM-L6-v2: ... query_p50=3.0ms query_p95=3.4ms passages/s@122tok=179.8
intfloat/multilingual-e5-base: ... query_p50=22.2ms query_p95=23.5ms passages/s@140tok=29.6
```

## search_latency_bench.py

Synthetic Zipf-distributed random-word corpus, 150 words per chunk, 50 queries. BM25 (k1=1.5, b=0.75) via a
scipy sparse matrix; dense = exact brute-force dot product over 384-d float32 unit vectors.

```
n=   1000: bm25_build=0.04s bm25_query_p50=0.04ms dense384_bruteforce_p50=0.01ms dense_matrix=1.5MB
n=  10000: bm25_build=0.28s bm25_query_p50=0.08ms dense384_bruteforce_p50=0.10ms dense_matrix=15.4MB
n= 100000: bm25_build=2.69s bm25_query_p50=0.40ms dense384_bruteforce_p50=1.62ms dense_matrix=153.6MB
```
