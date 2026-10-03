"""DeltaPlanner, DeltaQueryGenerator, semantic cache, query lineage (Phase 6, brief §10-18). TEST FIXTURE corpus."""

from streamrag.delta.planner import SemanticCache

from session_helpers import engine_of, p6_stack, run_turns  # noqa: F401  (fixture)

LADDERS = "What are the rules for ladders in the orchard?"


def test_delta_query_equals_full_build_terms(p6_stack):
    """The delta query (previous query + semantic delta) has the same terms as a from-scratch build."""
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Actually, ignore the overnight restriction.",
                                 "Only for the night shift."])
    eng = engine_of(p)
    it = eng.tracker.intents["I1"]
    full = eng.qb.build(it, eng.tracker.constraints_for(it))
    delta = eng.last_query["I1"]
    assert sorted(delta.terms) == sorted(full.terms)
    assert [c.source for c in delta.components][-1] == "constraint"


def test_plan_actions_and_lineage(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Actually, ignore the overnight restriction."])
    eng = engine_of(p)
    p2, p3 = rs[1].plan, rs[2].plan
    assert [a.action for a in p2.queries_to_create] == ["retrieve"] and p2.queries_to_supersede == ["Q1"]
    assert p2.queries_to_create[0].reason == "delta_query_for_changed_need"
    # removing the constraint returns the need to its first state: the semantic cache serves Q1's evidence
    [a] = p3.queries_to_reuse
    assert a.action == "cache_hit" and a.reused_query_id == "Q1" and p3.queries_to_create == []
    q = {r.query_id: r for r in eng.ledger.all()}
    assert q["Q2"].parent_query_id == "Q1" and q["Q2"].derived_from_change_id == p2.change_ids[0]
    assert q["Q3"].status == "reused" and q["Q3"].reused_from == "Q1" and q["Q3"].parent_query_id == "Q2"
    assert q["Q1"].stale and q["Q2"].stale and not q["Q3"].stale
    assert len({r.semantic_key for r in eng.ledger.all()}) == 2 and q["Q1"].semantic_key == q["Q3"].semantic_key


def test_reuse_active_when_semantics_unchanged(p6_stack):
    """A repeated late detail changes nothing retrieval-relevant: the active query is reused, nothing re-run."""
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Specifically overnight."])
    r3 = rs[2]
    assert r3.retrievals == 0 and r3.cache_hits == 0
    assert r3.plan is None or all(a.action == "reuse_active" for a in r3.plan.queries_to_reuse)


def test_semantic_cache_key_rules():
    c = SemanticCache("idx", "opt")
    assert c.key(["ladder", "rule", "orchard"]) == c.key(["orchard", "ladder", "rule", "rule"])   # order-free
    assert c.key(["ladder", "rule"]) != c.key(["crate", "rule"])                                   # entity change
    assert c.key(["ladder"]) != SemanticCache("idx2", "opt").key(["ladder"])                       # corpus snapshot
    assert c.key(["ladder"]) != SemanticCache("idx", "opt2").key(["ladder"])                       # retrieval config
    c.put(c.key(["a"]), "Q1", ["E1", "E2"])
    c.put(c.key(["b"]), "Q2", ["E3"])
    assert c.get(c.key(["a"])) == "Q1"
    assert c.invalidate_evidence({"E2"}) == 1 and c.get(c.key(["a"])) is None and c.get(c.key(["b"])) == "Q2"
    assert c.invalidate("index_changed") == 1 and c.entries == {}
    assert SemanticCache("idx", "opt", enabled=False).get(c.key(["b"])) is None


def test_full_restart_plan_retrieves_every_need_of_the_frame(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "And how are the wicks trimmed, especially overnight?",
                                 "Specifically for the keeper."], restart=True)
    for r in rs:
        assert r.plan is None or r.plan.full_restart
    assert rs[-1].retrievals >= 1 and rs[-1].cache_hits == 0 and rs[-1].reused_active == 0
