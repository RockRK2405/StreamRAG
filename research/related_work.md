# Related work and positioning

This is a short positioning, not a survey. Phase 1 (`../PHASE_1_RESEARCH_DOSSIER.md` §8, §12) compared the technical
options and records the measurements behind each choice.

| family | typical idea | what StreamRAG takes / does differently |
|---|---|---|
| conventional RAG | retrieve once on the finished question, generate once | used as baselines A–C (naive dense, hybrid, hybrid + cross-encoder) |
| hybrid lexical + dense retrieval with rank fusion | BM25 for exact terms, embeddings for paraphrase, reciprocal-rank fusion | adopted as the retriever layer (Phase 3); exact in-memory search, since an ANN index is unnecessary at this corpus size (measured) |
| cross-encoder reranking | re-score the top candidates with a pairwise model | implemented and measured; **off** in the final system: higher MRR but no answer-quality gain and +34–56 ms per search (Phase 10 §18) |
| query decomposition | split a compound request into sub-questions | rule-first decomposition while streaming (Phase 5); per-need retrieval and intent-aware fusion |
| adaptive / active retrieval (for example FLARE, Self-RAG, Adaptive-RAG) | decide *whether*, *when* and *how much* to retrieve, often with an LLM in the loop or a trained classifier | StreamRAG decides per *need* with cheap rules and claim requirements: strategy (fast path / filtered / iterative / multi-hop), k, and a bounded stop rule. It adds a streaming-specific dimension: *when the partial transcript is stable enough*. No LLM is on the retrieval decision path. |
| iterative / multi-hop retrieval (for example IRCoT-style interleaving) | retrieve again using what was found | bounded, requirement-targeted iteration and bridge-entity hops (Phase 9); measured to help on development data, **not** on the held-out corpus (Phase 10 §16) |
| grounding and attribution (claim verification, citation checking) | check generated statements against sources | every claim is verified by an NLI model plus rules before release; citations are rebuilt from verification; unsupported claims are repaired or removed (Phase 7) |
| streaming / incremental dialogue systems | process input incrementally and revise output | the streaming runtime (Phase 8): versioned answer states, drafts while speaking, cancellation of superseded work, deterministic replay |
| conversational / session RAG | carry conversation state into retrieval | typed context changes (refine / correct / extend), an evidence lifecycle per need, delta retrieval and validated reuse (Phase 6) |

**Positioning.** The individual parts are known techniques. The contribution is their integration under streaming
constraints. Each one is a measured, ablated mechanism:
1. per-need, claim-driven adaptive retrieval;
2. delta retrieval with an evidence lifecycle;
3. cancellation-aware asynchronous execution;
4. claim-level verification on every streamed answer.

The fixes made in Phase 11 show what streaming adds beyond turn-based RAG, namely early-commitment errors. See
`../NOVELTY.md` for what is and is not novel.
