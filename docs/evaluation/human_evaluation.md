# Human evaluation protocol (Phase 10) — prepared, NOT RUN

No independent human annotators were available in this phase, so **no human evaluation was performed** and no
number in the Phase 10 report comes from one. This protocol and the sheet
`experiments/datasets/human_eval_sheet.csv` (generated from the stored answers) are ready for the team.

**Sample.** 28 test turns stratified by query type (2 per type, seeded random choice), each answered by naive RAG,
hybrid + rerank and the full system: 84 items shown blind and in random order. The sheet has no system column; the
item → (sample, system) key is in `experiments/datasets/human_eval_key.json` and must not be shown to annotators.
Both files are written by `experiments/runners/analyze.py`.

**Criteria (1–3 each):** correctness (1 wrong / 2 partly / 3 correct w.r.t. the corpus), relevance (answers the
question asked), grounding (every factual statement is supported by the cited section — check each citation),
completeness (all parts of the question), citation usefulness (citations point to the right sections).

**Procedure.** Two annotators independently, with the corpus open; disagreements discussed only after both finished;
report Cohen's κ per criterion and the raw disagreement list. 84 items over 28 questions are not statistically representative — the
study would calibrate the automatic metrics (key-fact correctness, verifier-judged support), not replace them.

**LLM judge.** Not used (brief §57: it would need human labels to validate it; none exist yet).
