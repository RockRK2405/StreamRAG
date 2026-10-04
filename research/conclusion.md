# Conclusion

The full conclusion, research-question answers and future work are in `../FINAL_RESEARCH_CONCLUSION.md`.

**In one paragraph.**
* Streaming RAG can give a speaking user evidence and draft content within about a quarter of a second of the first
  word, and can keep every stated value grounded in a verified, cited source.
* Need-level state, delta retrieval and cancellation make late details and corrections cheaper without a measured
  quality cost.
* Its distinctive failure mode is early commitment: decisions locked in on a partial transcript. Two general rules
  bring this close to batch-mode quality.
* On two held-out fixture sets, the verified, adaptive system did not beat strong turn-based baselines on answer
  correctness, and it did not deliver the *verified* answer sooner. Its advantage is the error profile, not accuracy:
  * no stale values;
  * no hallucinated values;
  * failures by omission rather than by commission.
* The weak points are follow-up rewriting in unseen domains and recognising unanswerable questions. Those, and an
  evaluation on the official corpus with real speech input, are the next steps.
