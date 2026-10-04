# 04 · Adaptive top-k (Phase 9)

Code: `policy.py` (`initial_k`, `next_k`), `controller.py` (`expand_k` action).

* Initial k by complexity: SIMPLE 3, MODERATE 5, COMPLEX 5, MULTI_HOP 5 (`initial_k`).
* Expansion follows `k_schedule` (5 → 10 → 20, `max_top_k` = 20) **only** while requirements are unmet and the last
  search of the question had more fused candidates than its k (otherwise the expected gain of `expand_k` is 0).
* Targeted follow-up searches (requirement queries, filter relaxation, contradiction search) use the next k of the
  schedule; hops use the plan's k.
* The final evidence is not "top k": it is the evidence that supports a requirement (plus, when the need is not
  SUFFICIENT, the other results of the question's own searches as context), deduplicated, capped at
  `max_per_document` per source and `final_k` overall (06).
* Ablation switch `adaptive_k: false`: k = `k_schedule[0]` for every need and no expansion.

Measured effect: report §7 / research/phase9/results/ablations*.json, curves.json.
