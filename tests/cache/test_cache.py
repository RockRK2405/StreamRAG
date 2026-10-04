"""Cache reuse (brief §61) and invalidation (§62): reuse only what is still valid; every invalidation has a reason."""

import dataclasses

from adaptive_helpers import controller

from streamrag.adaptive.cache import AdaptiveQueryCache, ValidatedClaimCache
from streamrag.adaptive.controller import AdaptiveRequest


def test_equivalent_query_reuses_cached_evidence(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        first = ctl.run(AdaptiveRequest("q1", "What is the residence permit application fee?"))
        second = ctl.run(AdaptiveRequest("q2", "What is the application fee for a residence permit?"))
        assert first.ops.searches >= 1 and first.cache == "miss"
        assert second.cache == "hit" and second.ops.searches == 0 and second.plan.strategy.value == "CACHE_REUSE"
        assert [e.evidence_id for e in second.evidence.items] == [e.evidence_id for e in first.evidence.items]
        assert ctl.cache.stats["hits"] == 1 and ("CACHE_HIT" in [t for t, _ in second.events])
        saved = first.ops.searches - second.ops.searches
        assert saved == first.ops.searches                # retrieval calls saved by the hit
    finally:
        ctl.close()


def test_entity_b_does_not_reuse_entity_a(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        a = ctl.run(AdaptiveRequest("q1", "Do applicants from Zemland need an interview?"))
        b = ctl.run(AdaptiveRequest("q2", "Do applicants from Norvia need an interview?"))
        assert b.cache == "miss" and b.ops.searches >= 1
        assert {h.bridge for h in b.hops if h.bridge} != {h.bridge for h in a.hops if h.bridge}
        inv = ctl.cache.invalidate_entities({"zemland"})
        assert inv["reason"] == "entity_changed" and inv["n"] >= 1
        assert all("zemland" not in e.entity_terms for e in ctl.cache.entries.values())
    finally:
        ctl.close()


def test_source_version_change_invalidates(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        ctl.run(AdaptiveRequest("q1", "What is the residence permit application fee?"))
        info = ctl.catalog.docs["ELIG-2026"]
        changed = dataclasses.replace(ctl.catalog, docs={**ctl.catalog.docs, "ELIG-2026": dataclasses.replace(
            info, meta={**info.meta, "version": 3})})
        ctl2 = controller(adaptive_env)
        ctl2.catalog, ctl2.cache = changed, ctl.cache
        r = ctl2.run(AdaptiveRequest("q2", "What is the residence permit application fee?"))
        assert r.cache == "invalidated" and r.ops.searches >= 1
        assert ctl.cache.log[-1]["reason"] == "source_version_changed"
        assert any(t == "RETRIEVAL_INVALIDATED" for t, _ in r.events)
        ctl2.close()
    finally:
        ctl.close()


def test_temporal_validity_invalidates(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        ctl.run(AdaptiveRequest("q1", "Which public holidays are there in 2026?"))
        key, entry = next(iter(ctl.cache.entries.items()))
        entry.valid_at = "2027-06-01"                     # the entry is now asked for a date its sources do not cover
        hit, inv = ctl.cache.get(key, ctl.catalog)
        assert hit is None and inv["reason"] == "temporal_validity"
    finally:
        ctl.close()


def test_stale_evidence_and_constraint_and_corpus_invalidation(adaptive_env):
    from streamrag.adaptive.cache import CacheEntry
    ctl = controller(adaptive_env)
    try:
        r = ctl.run(AdaptiveRequest("q1", "How long does processing take for domestic applicants?"))
        assert r.evidence.items
        key = next(iter(ctl.cache.entries))
        hit, inv = ctl.cache.get(key, ctl.catalog, usable=lambda i: False)     # Phase 6 marked the evidence stale
        assert hit is None and inv["reason"] == "stale_evidence"
        ctl.cache.put(CacheEntry("kc", "q", "I1", {"permit"}, {"intern"}, None, {}, [], "SUFFICIENT", 0.0))
        ctl.cache.put(CacheEntry("ke", "q", "I2", {"zemland"}, set(), None, {}, [], "SUFFICIENT", 0.0))
        inv = ctl.cache.invalidate_constraints({"intern"})                     # constraint retracted / replaced
        assert inv["reason"] == "constraint_changed" and inv["keys"] == ["kc"]
        inv = ctl.cache.invalidate_corpus("new-corpus-hash")                   # new corpus snapshot
        assert inv["reason"] == "corpus_changed" and inv["keys"] == ["ke"] and not ctl.cache.entries
        assert [x["reason"] for x in ctl.cache.log] == ["stale_evidence", "constraint_changed", "corpus_changed"]
    finally:
        ctl.close()


def test_cache_bounded_lru():
    c = AdaptiveQueryCache("h", max_entries=2)
    from streamrag.adaptive.cache import CacheEntry
    for i in range(3):
        c.put(CacheEntry(f"k{i}", "q", None, set(), set(), None, {}, [], "SUFFICIENT", 0.0))
    assert list(c.entries) == ["k1", "k2"] and c.stats["evicted"] == 1


def test_validated_claim_cache():
    cc = ValidatedClaimCache()
    cc.add("The fee is 55 euros.", {"fee", "55", "euro"}, ["E1"], "SUPPORTED")
    cc.add("Unsupported claim.", {"x"}, ["E2"], "UNSUPPORTED")
    assert len(cc.claims) == 1
    assert cc.evidence_for({"fee", "euro"}, 0.6, usable=lambda i: True) == ["E1"]
    assert cc.evidence_for({"fee", "euro"}, 0.6, usable=lambda i: False) == []
    assert cc.invalidate_evidence({"E1"}) == 1 and not cc.claims


def test_key_ignores_word_order_but_not_filters():
    c = AdaptiveQueryCache("h")
    assert c.key(["fee", "permit"], None, None, True) == c.key(["permit", "fee"], None, None, True)
    assert c.key(["fee"], None, None, True) != c.key(["fee"], {"metadata": {"applicant_type": ["domestic"]}}, None, True)
    assert c.key(["fee"], None, "2025-12-01", True) != c.key(["fee"], None, None, True)
