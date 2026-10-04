"""Brief §65-§67 plumbing: latency, quality and operation counts are measured, never asserted as improvements here
(the measurements and their analysis are in research/phase9; fixture results are not reportable)."""

import json

from adaptive_helpers import controller, run

from conftest import REPO
from streamrag.adaptive.controller import AdaptiveRequest


def test_latency_is_measured_per_search_and_per_run(adaptive_env):
    r = run(adaptive_env, "What documents does an applicant from Zemland need?")
    assert r.state.latency_spent_ms > 0 and all(s["latency_ms"] > 0 for s in r.searches)
    assert sum(1 for d in r.decisions if d.latency_cost_ms >= 0) == len(r.decisions)
    stop = next(p for t, p in r.events if t == "RETRIEVAL_STOPPED")
    assert stop["latency_ms"] == r.state.latency_spent_ms


def test_operation_counts_are_consistent(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        for q in ("How often must a residence permit be renewed?", "What documents does an applicant from Zemland need?",
                  "When does the Permit Processing Office open?"):
            r = ctl.run(AdaptiveRequest("q", q))
            o = r.ops
            assert o.searches == len(r.searches) == sum(1 for t, _ in r.events if t == "RETRIEVAL_STARTED")
            assert o.lexical_searches == sum("bm25" in s["retrievers"] for s in r.searches)
            assert o.dense_searches == sum("dense" in s["retrievers"] for s in r.searches)   # = embeddings computed
            assert o.chunks_returned == sum(len(s["items"]) for s in r.searches)
            assert o.llm_calls == 0 and o.reranker_calls == 0                               # no reranker loaded
    finally:
        ctl.close()


def test_quality_not_degraded_on_fixture_questions(adaptive_env):
    """Non-degradation check on the fixture corpus (hashing embedder): adaptive coverage of the gold sections is not
    below fixed hybrid k=5 by more than 0.1 on the A-cases with gold."""
    cfg, svc = adaptive_env
    ctl = controller(adaptive_env)
    cov_a, cov_f = [], []
    try:
        for p in sorted((REPO / "eval" / "dev_adaptive_retrieval").glob("A*.json")):
            t = json.loads(p.read_text())["turns"][0]
            gold = set(t["gold"]["citations"])
            if not gold:
                continue
            r = ctl.run(AdaptiveRequest(p.stem, t["utterance_text"]))
            cov_a.append(len(gold & {e.citation for e in r.evidence.items}) / len(gold))
            es = svc.retrieve(t["utterance_text"])
            cov_f.append(len(gold & {e.citation for e in es.items[:5]}) / len(gold))
    finally:
        ctl.close()
    assert sum(cov_a) / len(cov_a) >= sum(cov_f) / len(cov_f) - 0.1
