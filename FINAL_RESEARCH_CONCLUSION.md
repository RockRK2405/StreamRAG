# Final Research Conclusion

**Question.** Can a RAG system answer a *live, changing* spoken request:
* retrieving while the user speaks;
* handling several needs, late details and corrections incrementally;
* stating only what its sources support?

And what does that cost compared with turn-based RAG?

**Evidence base.**
* Phase 10: v1, 87 turns, later used as development data.
* Phase 11: the held-out set v2 (81 turns, new domain), written and hash-frozen before any Phase 11 change and run once.
* Paired tests on both.
* Every number is from fictional fixture corpora with implementer labels. **The results are indicative and NOT
  REPORTABLE as official results.** Tables: `FINAL_BENCHMARK_RESULTS/README.md`, `PHASE_10_EVALUATION_REPORT.md`.

## Answers to the research questions

| RQ | answer | evidence |
|---|---|---|
| **RQ1** Does adaptive retrieval improve the quality–efficiency tradeoff? | **Partly, and consistently across both held-out sets.** Evidence precision doubles (v2: 0.24 → 0.55, p < 0.001) with fewer embeddings (1.06 → 0.83 per turn, p = 0.007), and stale values disappear (7 / 13 → 0 / 13). Recall@5 is lower (0.98 → 0.94, n.s. on v2; significant on v1). There are more searches (+0.28 per turn, p = 0.006). Answer correctness is not significantly different. | v2 `topk_5` vs `adaptive_x` (extractive); v2 `hybrid_rerank` vs `adaptive_rag`; Phase 10 RQ1 |
| **RQ2** Does claim-driven retrieval improve grounding? | **Not established.** Phase 10 found a small, non-significant gain (+4 / −0 turns). It was not re-ablated on v2. | Phase 10 Exp 3 |
| **RQ3** Does delta retrieval reduce redundant work? | **Yes, without a quality cost on v2.** Full re-retrieval needs 1.49 retrieval calls per turn vs 1.35 with delta (p = 0.016), with identical answer correctness (0.718). | v2 `delta_full_restart` vs `adaptive_x`; Phase 10 RQ3 |
| **RQ4** Does cancellation reduce wasted computation? | **Yes, when the user corrects a question that is still being answered.** For the same inputs, worker time is −27% and LLM calls go 15 → 9. Final-answer correctness is unchanged (0.556 both), and the corrected answer is verified slightly sooner (p50 1,656 → 1,509 ms after the end). Superseded questions go unanswered by design. The Phase 10 v1 result was similar: −30% worker time and −50% LLM calls. | v2 cancellation experiment (n = 14 turns, descriptive) |
| **RQ5** Does streaming reduce perceived response latency? | **Yes for evidence and first content, no for the verified answer.** First evidence p50 is 264 vs 767 ms and first answer content 271 vs 2,477 ms, against the batch version of the same pipeline (p < 0.001). The verified answer after the user stops is about the same: p50 1,643 vs 1,659 ms, and naive RAG takes 1,411 ms. | v2 `full_system` vs `full_system_batch` |
| **RQ6** Does session memory reduce retrieval repetition? | **A little.** Validated caches answer 7.4% of turns without a search, and retrieval calls fall from 1.42 to 1.35 per turn (p = 0.031), with no quality change. But follow-up and correction turns in a new domain are a weakness: 0.61 correct in streaming vs 0.72 in batch mode. | v2 `cache_none` vs `adaptive_x`; v2 gate G5 |
| **RQ7** Does contradiction-aware retrieval improve reliability? | **Yes, and in Phase 11 also in streaming.** On v2 the full system asserts 0 stale values (baselines 2–7 of 13), reports both values when notices conflict (2 / 2), and gives the current version for superseded documents (2 / 2; hybrid + rerank 0 / 2). In Phase 10 the streaming system still asserted stale values in 2 of 3 temporal turns. The Phase 11 early-commitment fixes removed this on v2. | v2 special cases; Phase 10 RQ7 |
| **RQ8** What is the latency cost of stronger verification? | **Small in time.** Verification costs milliseconds per answer version (stage profile). Repair adds LLM calls, and Phase 10 measured +55 ms TTVA. It also removes some true claims (Phase 10: −0.068 correctness, n.s.). | Phase 10 RQ8; `FINAL_BENCHMARK_RESULTS/README.md` §8 |

## Findings

**1. Streaming RAG's distinctive failure mode is early commitment, and it can be controlled.**
* Decisions made on a partial transcript (retrieval, temporal resolution, constraint evidence) survive into the final
  answer.
* In Phase 10 this cost 7 turns against batch mode on v1. Two general rules removed most of it:
  * drop evidence that only superseded partial-transcript queries retrieved;
  * re-validate needs whose query budget is exhausted.
* After the fixes the gap was 2 turns on v1 (development data) and a net 1 turn on the held-out v2: 0.718 vs 0.732.
  On v2, batch mode alone was correct on 2 turns and streaming alone on 1.
* Stale values went from 3 of 13 (Phase 10, v1) to 0 on both sets.

**2. Verification changes the error profile more than the accuracy.**
* The verified systems state no unsupported or hallucinated values. The RAG baselines do in 6–9% of claims, and assert
  stale values in 15–31% of the relevant turns.
* Answer correctness, in the sense of all key facts stated, is not higher: 0.72 vs 0.75–0.80, n.s.
* The verified system fails by omission ("not established", an adjacent fact). The baselines fail by commission.

**3. Adaptive, need-level retrieval buys precision and cost, not recall.**
* Handing the generator half the noise does not raise correctness on corpora this small.
* Multi-constraint questions suffer where top-5 already contains everything.
* Whether the precision pays off on a large corpus is the main open question, and it needs the official corpus.

**4. Incrementality is cheap to keep correct only with explicit lifecycles.**
* Delta retrieval, caches and cancellation each save measurable work with no quality change on v2.
* Each needs version and validity checks to stay correct. Without them, reuse becomes stale reuse (Phase 6 and
  Phase 10).

**5. Rules generalise unevenly.**
* The rule-based controller generalised well: early retrieval on 98.7% of eligible v2 turns.
* Decomposition generalised well too: the multi-intent count was right in 4 of 4.
* Context-change rewriting for elliptical follow-ups and entity corrections did not transfer to the new domain. This is
  the clearest target for a small learned component.

## What this does and does not show

* **Shown, on fixture data, across two held-out sets.**
  * Streaming retrieval gives evidence and draft content within about a quarter of a second of the first word.
  * Verification removes unsupported values.
  * Incremental state and cancellation reduce work.
  * Phase 11 controlled the costs of early commitment.
* **Not shown.**
  * Better answer accuracy than strong turn-based baselines.
  * A faster *verified* answer.
  * Robust follow-up handling in unseen domains.
  * Behaviour on the official corpus, real ASR, a large index, or any device.

## Future work

1. **Official corpus and real ASR.** Re-run the frozen benchmark protocol unchanged.
2. **Learned context rewriting.** A small model for elliptical follow-ups and entity corrections, evaluated against the
   rule baseline on a fresh held-out set.
3. **Answerability.** A calibrated insufficiency detector. The LLM flag tested here traded correct answers for correct
   abstentions.
4. **Reranker and recall.** Re-evaluate the cross-encoder, which was the best baseline on v2, and adaptive k for
   multi-constraint needs, on fresh held-out data.
5. **Device split.** Measure the rule-based front end and the embedder on a phone-class device.
