# 07 · Multi-hop retrieval (Phase 9)

Code: `controller.py` (`_hops`, `_relation_target`, `_reference`), contract `RetrievalHop`
(hop_id, parent_hop_id, query, purpose, evidence_ids, status, bridge).

1. **Hop 0** — the question as asked; **hop 0e** — the entity alone ("Zemland" → what the corpus says about it), run in
   parallel when the need is MULTI_HOP.
2. **Bridge discovery** in the retrieved evidence: a sentence "<entity> <relation cue> <target>" (relation cues are
   configured, e.g. "is classified as", "belongs to", "falls under") or "<reference cue> <document title>" ("see the
   … Annex" → that document). The target is a short phrase (≤ 6 words, cut at punctuation); sentences that look like
   instructions (`instruction_like`) are never used.
3. **Hop n** — the question with the entity replaced by the target ("… applicant from Group B …"), or the question
   restricted to the referenced document. The original slot becomes a *link* slot (entity → target, key = entity) and
   a *bridge* slot (aspect for the target, key = target words) is added (05). HOP_CREATED is emitted.
4. Hops count against `max_hops` (bridge hops; hop 0 / 0e excluded), `max_queries` and the latency budget. A hop
   round is not counted as stalled (its value shows in the next assessment).

Hop 2 depends on hop 1 by construction: its query text comes from evidence hop 1 returned
(tests/multi_hop/test_multi_hop.py checks it). `multi_hop: false` routes MULTI_HOP needs to ITERATIVE without hops.

**Security:** corpus text contributes at most the short target phrase of a hop query; it never changes the strategy,
k, the number of hops or any budget.

**Limitations:** English relation cues only; a bridge phrase that reduces to one stem ("Group A" → "group", "a" is a
stopword) is ambiguous between members of the class; the analyzer's entity cue needs capitalisation (01).
