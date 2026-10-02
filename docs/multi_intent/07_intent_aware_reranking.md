# 07: Intent-Aware Reranking

**Code:** `fusion/engine.py` (`_rerank`); cross-encoder `retrieval/rerank.py` (Phase 3, `ms-marco-MiniLM-L6-v2` ONNX)
**Config:** `fusion.rerank` (default `none`), `fusion.rerank_k`

Reranking is pluggable and scores candidates against the **intent's query** (which includes its constraints and inherited context), never against the whole utterance.

| Mode | Scores | Cost | Effect |
|---|---|---|---|
| `none` | Retrieval order | 0 | Default |
| `intent_ce` | Cross-encoder(intent query, chunk) on each intent's own top `rerank_k` | One CE call per intent | Reorders within each intent |
| `cross_intent_dense` | Cosine(intent query embedding, chunk embedding from the index matrix) for **every** candidate × **every** intent | One batched query embedding plus a matrix product | A chunk retrieved by I1 can enter I2's list when it is relevant to I2's query (`intent_relevance`) |
| `cross_intent_ce` | Cross-encoder for every candidate × every intent | candidates × intents CE pairs (most expensive) | Same as above, with a stronger model |

**Behaviour:**
- **Failure:** a missing model or an error falls back to retrieval order, with a `warnings` entry and `rerank: none` in the output (spec §18.9; tested).
- **Streaming:** reranking runs only for the **final** fusion of an utterance (rerank-on-stability, ADR-003). Provisional fusions are rerank-free. Events: `RERANK_STARTED` / `RERANK_COMPLETED`.

**Measurement:** quality and latency of each mode on the dev suite are in Phase 5 report §12 and §19.
