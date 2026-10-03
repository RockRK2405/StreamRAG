# ADR-017: Grounded Generation: Plan-Then-Write, Entailment Verification, Citations from Verification (Phase 7)

- **Status:** Accepted (Phase 7)
- **Date:** 2026-10-03
- **Refines:** ADR-007 (LLM backend), ADR-008 (grounding and citation), ADR-016 (adaptive session)

## Context

- **Goal:** turn the Phase 6 claim and evidence state into answers in which every material factual claim is traceable to supporting evidence (G4 ≥ 85 %; zero fabricated ids).
- **Team choices (Phase 7):**
  - a small *local* LLM through Ollama (`qwen3:4b`, Q4_K_M, about 2.5 GB);
  - a pinned NLI cross-encoder for entailment (`nli-deberta-v3-xsmall`, about 280 MB, ONNX, CPU).
- **Constraints:** no hosted model, deterministic replay, no fabricated results.

## Decision

1. **Plan, then write.** Claims are derived from evidence before generation (`ClaimPlanner`: verbatim sentences or entailment-checked atoms). The model only expresses the planned facts (`facts` prompt). Gaps are stated deterministically by templates, never by the model.
2. **Structured output, bounded retries, extractive fallback.** The generator returns JSON matching a schema. An invalid reply is retried once, then the section is rendered extractively. Arbitrary prose is never parsed.
3. **Verification by entailment, not similarity.**
   - Per claim and evidence: sentence, window and chunk premises are scored by the NLI model, together with deterministic number checks.
   - Contradictions count only for single sentences about the same proposition (shared-content gate).
   - Statements not supported as a whole are decomposed and verified per atom.
4. **Citations come from the verification**, down to the supporting sentence span, with location metadata from the index. The model's labels are never trusted. A claim without a valid citation is removed.
5. **Deterministic unsupported-claim policy** (keep / conflict / keep atoms / restore facts / bounded retrieval fallback / remove):
   - **strict** (default) revises sections that produced unsupported content, at most 2 times, then completes them extractively;
   - **relaxed** only acts on missing critical facts.
6. **Conflicts are presented, never resolved.** There is no source priority without metadata.
7. **Incremental.** Only sections Phase 6 marks changed are regenerated, and inside them only facts not already expressed. Kept sentences keep their ids. Drafts are extractive; finals use the LLM.
8. **Prompt-injection defence in depth.**
   - Instruction-like sentences are neither facts nor premises.
   - Data is delimited and quoted.
   - The output is schema-bound.
   - Citations come from verification.
9. **Replayable.** LLM outputs are recorded in `LLM_CALL` and replayed by request hash.
10. **The LLM socket is loopback only.** The `generation` package is the only package allowed to open a socket.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Free answer, then search for citations | Rationalised citations (ADR-008). Measured in the ablation as arm C (labels trusted) vs D/E/F. |
| Similarity / term overlap as support | Similar chunks need not state the fact (brief §10). Overlap is kept only as WEAK, never support. |
| LLM-as-judge verification | Same failure modes as the generator, slow, needs a second call per claim. An NLI cross-encoder is about 5 ms per pair, deterministic and local. |
| Hosted LLM | The team chose local (no key, no data leaves the machine). The gateway interface accepts a hosted adapter later. |
| Using `qwen3.6` (23 GB, already installed) | The dev machine has 24 GB RAM. A 4B model is about 10× smaller and fast enough on CPU/GPU. |
| Resolving conflicts by recency | No metadata in the corpus. "Newer = correct" is not justified. |
| LLM-written uncertainty sentences | Uncertainty must not depend on the model admitting ignorance (ADR-008 §4). Templates are deterministic. |

## Consequences

- **Guarantee and its limit.** Every released factual claim has a verified STRONG / MODERATE premise and a valid index citation. Its accuracy is bounded by the NLI model's precision (measured: report §6). Groundedness is not truth.
- **Cost.** Generation dominates latency (seconds per answer on the dev machine). Verification is tens of milliseconds per answer.
- **Model quality.** The small model sometimes produces garbled but entailed text ("in the orch: overnight"). Verification keeps it because the meaning is supported; fluency is not checked.
- **Known gaps** (report §22):
  - entailment errors on dropped qualifiers and modality shifts;
  - heuristic injection markers;
  - English-only decomposition;
  - generation blocks the realtime event loop.
