# Judge Q&A

Short answers, each pointing to the evidence. Numbers are from the held-out v2 benchmark
(`FINAL_BENCHMARK_RESULTS/README.md`) unless stated: fictional fixture corpus, 81 turns, one laptop, local LLM.

**1. What is actually new here? Isn't this just RAG with streaming?**

Four things that turn-based RAG does not have:
* a need-level ledger and an evidence lifecycle, so a late detail or correction re-retrieves only the affected need;
* claim-requirement-driven adaptive retrieval with no LLM in the loop;
* verification of every streamed claim, draft or final, against its cited section;
* explicit handling of early commitment, i.e. decisions made on a partial transcript.

The retrieval models are standard (BM25, bge-small, NLI, a local LLM) and are not claimed as novel. See `NOVELTY.md`.

**2. Why rules for retrieve / wait / skip and intent splitting instead of an LLM?**
* Latency and cost. Those decisions run on every transcript chunk. Phase 1 measured that an LLM there costs hundreds
  of milliseconds per chunk; the rules cost milliseconds.
* The official guide penalises heavy orchestration.
* The cost is generality: the rules are English-only and degrade on unfamiliar phrasing (`LIMITATIONS.md` §2, §4).

**3. How do you know the answer is grounded?**
* Every claim is checked by an NLI model against the section it cites. Unsupported claims are repaired or removed,
  and citations are rebuilt from the check, never copied from the LLM.
* A model-free check finds 0 hallucinated values for the full system vs 6–9% of claims for the RAG baselines
  (p ≤ 0.008).
* Caveat: the verifier-judged 100% claim support uses the system's own verifier, so it is biased (Phase 10 §23).

**4. What happens when the user corrects themselves mid-question?**
* The context-change detector types the change: refine, correct or extend.
* The ledger marks the affected need's evidence as stale or superseded.
* In-flight work for the old version is cancelled, and only that need is searched again.
* On v2 corrections, cancellation cut worker time by 27% and LLM calls from 15 to 9, with the same final-answer
  correctness.
* Demo scenario **D** shows this live.

**5. What if the LLM is slow or fails?**
* The turn falls back to verified extractive answers, and the degraded mode is announced in the UI.
* Phase 11 fixed a bug where an LLM *timeout* left the turn without an answer (Phase 10: 0 / 10 recovered).
* The demo runs every scenario without the LLM: `demo-check --no-llm`. See `FINAL_BENCHMARK_RESULTS/README.md` §6.

**6. Is it real-time?**
* No real-time guarantee is claimed.
* Measured on one laptop with a local 4B model:
  * first evidence 0.26 s and first answer content 0.27 s after the first word (p50);
  * verified answer 1.64 s after the user stops (p50), 3.99 s at p95.
* The verified answer is **not** faster than naive RAG's (1.41 s, not significant). The gain is earlier evidence and
  draft content.

**7. How did you avoid overfitting your own test set?**
* Phase 10's test set was inspected in its error analysis, so it was relabelled as development data.
* A new held-out set (v2, new domain) was written and hash-frozen **before** any Phase 11 change
  (`experiments/datasets/streamrag_eval_v2/FROZEN.sha256`).
* Fixes were developed on other data, and the final system was run on v2 once.
* Tests check that no v2 string appears in the source code, configs or demo data.

**8. Your answer correctness is not better than the baselines'. Why should we care?**
* That is correct: 0.72 vs 0.76–0.80, not significant. We report it.
* The system wins where errors are costly or visible:
  * 0 stale values vs 15–54%;
  * 0 hallucinated values;
  * a citation that supports every claim;
  * evidence in a quarter of a second;
  * cheaper corrections.
* For rules, fees and deadlines, "not in the sources" beats a confident wrong number.

**9. Why is recall lower?**
* Adaptive retrieval hands the answer stage fewer, more precise sections: about 2.2 times the evidence precision.
* On a corpus of 14 short documents, naive top-5 contains almost everything, so its Recall@5 (0.99) is easy to reach.
* The trade-off is real: multi-constraint questions suffer (`LIMITATIONS.md` §2).

**10. Will it scale to a large corpus?**
* NOT MEASURED beyond fixture corpora.
* The design is built for it:
  * adaptive retrieval avoids embedding searches when a keyword lookup suffices (0.83 embeddings per turn vs 1.0);
  * delta retrieval avoids repeat searches;
  * every search is bounded in k, iterations and latency.
* Index size and latency on a large corpus remain to be measured.

**11. What is the Samsung relevance? Does it run on a Galaxy device?**
* No Samsung hardware, SDK or model was used, and on-device execution was **not measured**.
* The relevance is the design:
  * voice-first assistants need incremental, correction-aware answering;
  * the pipeline splits cleanly into a cheap, model-free front end (controller, decomposition, session state) that
    could run on a device and keep the raw transcript local;
  * a server back end handles corpus search, the LLM and verification.
* The measured footprints are in `docs/security/README.md` §3.

**12. What about privacy?**
* In the documented deployments, nothing leaves the machine. Retrieval, embedding and verification run in process,
  and the LLM is local (Ollama).
* The LLM client refuses non-loopback hosts unless they are explicitly allowlisted.
* Sessions live in memory only, and logs contain no transcript or answer text.

**13. Can a malicious document inject instructions?**
* Retrieved text is treated as untrusted data:
  * it is quoted and delimited in prompts;
  * instruction-like sentences are excluded from facts;
  * every claim is verified against evidence, so an injected *fact* cannot reach the answer;
  * the UI renders text without HTML.
* There is a regression test for an injection fixture.
* The marker list is heuristic. We do not claim the model ignores every instruction.

**14. How do you handle conflicting or outdated documents?**
* Superseded documents are labelled. The answer gives the current value and marks the earlier version as superseded.
* Genuinely conflicting notices are both reported. On v2, both conflict cases were reported.
* Stale values went to 0 on v2, vs 2–4 of 13 turns for the RAG baselines.

**15. What if the documents don't contain the answer?**
* This is a weakness. Only 1 of 4 unanswerable v2 questions got an explicit "not in the sources".
* An optional LLM answerability check raised that to 2 of 4, but cost answer correctness (0.66 vs 0.72), so it is
  off by default.
* What the system does guarantee: it never invents a value. It may state an adjacent true fact.

**16. What does streaming cost?**
* More work: 1.86 retrieval calls and 1.27 LLM calls per turn, vs 1.0 each for naive RAG.
* Cancellation and delta retrieval recover part of it:
  * full re-retrieval costs 1.49 retrieval calls per turn vs 1.35 with delta retrieval;
  * cancellation saves 27% of worker time on corrections.

**17. How is the adaptive retrieval decided, and can I see it?**
* Each need gets claim requirements, e.g. "an amount" or "a form".
* A router picks one plan per need from lexical fast path, filtered, semantic, hybrid, iterative, multi-hop or reuse.
* It stops when the requirements are covered or the budget is spent.
* On v2, 55% of needs took the keyword fast path.
* The UI shows the plan chip and the activity log per turn (demo scenarios **C1** and **C2**).

**18. What would you do next?**
1. Evaluate on the official corpus.
2. Feed real ASR output, with its errors.
3. Add a small learned query rewriter for elliptical follow-ups and entity corrections, the main v2 weakness.
4. Re-tune the reranker and the answerability check on a fresh held-out set.
5. Measure on-device footprints of the front end.
