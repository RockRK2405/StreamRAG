# Phase 2 assumption check: rerank latency (raw)

Run on 2026-10-02, same dev machine as Phase 1 (Apple M5 Pro, torch 2.2.2 CPU).
`rerank_latency_probe.py` runs a 6-layer / 384-hidden BERT forward pass (cached all-MiniLM-L6-v2 weights) over
query+passage pairs. This is the same compute shape as `cross-encoder/ms-marco-MiniLM-L-6-v2` (which only adds a
384→1 head). Synthetic text, latency only.

```
threads=15 pairs=10 tokens/pair=128 p50=54ms  min=52ms
threads=15 pairs=20 tokens/pair=128 p50=105ms min=99ms
threads=15 pairs=30 tokens/pair=128 p50=149ms min=137ms
threads=2  pairs=10 tokens/pair=128 p50=63ms  min=61ms
threads=2  pairs=20 tokens/pair=128 p50=124ms min=123ms
threads=2  pairs=30 tokens/pair=128 p50=185ms min=184ms
```

Implication: 3 intents × 20 candidates ≈ 0.31–0.37 s serial on this machine. That fits the 0.5 s gap between the
last chunk and the end of the utterance in the guide's Example 1, but with little margin. A slower judge CPU would
spill it onto the critical path. This motivates the rerank-on-stability rule and the 20-candidate cap in the
Phase 2 spec.
