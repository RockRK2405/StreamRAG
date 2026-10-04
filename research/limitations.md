# Limitations (research view)

The full list, with numbers, is in `../LIMITATIONS.md`. These are the limits that bound the research conclusions.

1. **Validity of the data.**
   * Fictional fixture corpora (14–15 short documents each) with labels written by the implementer.
   * No official corpus, no human ratings, no real ASR.
   * The conclusions are about mechanisms, not product quality.
2. **Statistical power.**
   * 81 held-out turns. Per-category n is 4–15, and most answer-correctness differences are not significant.
   * Effects on work (retrieval calls, LLM calls, worker time) and on latency are significant. Accuracy effects are
     mostly not.
3. **Instrument bias.**
   * Verifier-judged metrics use the system's own NLI model.
   * The model-free hallucination check covers only values: numbers, dates and names.
   * Answer correctness is key-string matching, so a correct paraphrase of a key can be missed. The same applies to
     every system.
4. **Domain transfer of the rules.** The session-rewriting rules for follow-ups and entity corrections did not
   transfer to the v2 domain (gate G5: 0.61 streaming vs 0.72 batch).
5. **Single machine.**
   * All latency was measured on one laptop with a local 4B model.
   * One environment slowdown was observed and disclosed (the v1 regression run). Latency claims are relative, within
     a run.
6. **The development data was re-used.** v1 was inspected in Phase 10 and used to develop the Phase 11 fixes, so v1
   improvements are optimistic. Only v2 is held-out evidence for Phase 11.
