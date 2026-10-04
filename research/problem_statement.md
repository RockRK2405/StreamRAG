# Problem statement (research package)

The full text is in `../FINAL_PROBLEM_STATEMENT.md`. Research framing in brief:

**Task.** Answer a user from a fixed document corpus *while the user is still speaking*:
- input is a stream of transcript chunks with revisions;
- one utterance can hold several needs, late details and corrections;
- every answer statement must be grounded in and cited to a corpus section, or explicitly marked as not established.

**Research questions** (Phase 10 RQ1–RQ8, re-tested in Phase 11):
1. Does adaptive retrieval improve the quality–efficiency tradeoff?
2. Does claim-driven retrieval improve grounding?
3. Does delta retrieval reduce redundant work?
4. Does cancellation reduce wasted computation?
5. Does streaming reduce perceived response latency?
6. Does session memory reduce retrieval repetition?
7. Does contradiction-aware retrieval improve reliability?
8. What is the latency cost of stronger verification?

**Constraints.**
- Answers must be grounded and cited.
- CPU plus a local 4B LLM.
- No cloud services.
- Parsimony: the official guide penalises heavy orchestration.
- The official Theme 4 corpus is unavailable, so every result is measured on fictional fixture corpora and is NOT
  REPORTABLE as product quality.
