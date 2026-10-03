"""Phase 6 dev-suite scoring (adaptive sessions; eval/dev_adaptive). TEST FIXTURE DOMAIN ONLY - NOT REPORTABLE.

Scoring is against annotator gold per turn (see research/phase6/build_dev_suite.py). Gold query terms are written
as words and compared after the index analyzer (stemming / stopwords), so "ladders" matches the term "ladder".
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

BANNER = "DEV SUITE ON TEST FIXTURE DOMAIN - NOT AN OFFICIAL BENCHMARK RESULT - NOT HELD-OUT"


def load_adaptive_cases(path: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(path).glob("*.json"))]


def change_ok(detected: list[str], gold: dict) -> bool:
    options = gold.get("change_types_any") or [gold["change_types"]]
    return any(Counter(detected) == Counter(o) for o in options)


def active_queries(engine) -> dict[str, list[str]]:
    """Active session needs -> analyzed terms of their active query."""
    out = {}
    for it in engine.tracker.active_session_intents():
        q = engine.ledger.active_for_intent(it.intent_id)
        if q is not None:
            out[it.intent_id] = list(q.terms)
    return out


def need_matches(queries: dict[str, list[str]], gold_needs: list[dict], terms_fn: Callable[[str], list[str]]):
    """For each gold need: the first active need whose query has every required and no forbidden term (or None)."""
    out = []
    for g in gold_needs:
        req = {t for w in g["required"] for t in terms_fn(w)}
        forb = {t for w in g["forbidden"] for t in terms_fn(w)}
        hit = next((i for i, ts in queries.items() if req <= set(ts) and not (forb & set(ts))), None)
        out.append(hit)
    return out


def retrieval_verdict(gold: str, retrievals: int, reused: int) -> str:
    """ok | unnecessary_retrieval | missed_retrieval | missed_reuse."""
    if gold == "required":
        return "ok" if retrievals > 0 else "missed_retrieval"
    if retrievals > 0:
        return "unnecessary_retrieval"
    if gold == "reuse" and reused == 0:
        return "missed_reuse"
    return "ok"


def score_turn(engine, res, gold: dict, terms_fn: Callable[[str], list[str]]) -> dict:
    detected = [c.change_type for c in res.changes]
    queries = active_queries(engine)
    matches = need_matches(queries, gold["need_queries"], terms_fn)
    created = res.plan.queries_to_create if res.plan else []
    forb = {t for w in gold["forbidden_new"] for t in terms_fn(w)}
    leaks = [a.query.text for a in created if forb & set(a.query.terms)]
    cites = set()
    for iid in matches:
        if iid is not None:
            cites |= {engine.store.records[a.evidence_id].citation for a in engine.store.usable(iid)}
    gold_ev = gold["gold_evidence"]
    reval_tags = res.plan.evidence_to_revalidate if res.plan else []
    reval_cites = {engine.store.records[t.split("@")[0]].citation for t in reval_tags}
    ans = engine.answers.get_current_answer_state()
    conflict = bool(ans and any(u.kind == "conflict" for u in ans.uncertainty))
    return {
        "detected": detected, "gold": gold.get("change_types_any") or [gold["change_types"]],
        "change_ok": change_ok(detected, gold),
        "retrieval_gold": gold["retrieval"], "retrievals": res.retrievals,
        "reused": res.cache_hits + res.reused_active,
        "retrieval_verdict": retrieval_verdict(gold["retrieval"], res.retrievals, res.cache_hits + res.reused_active),
        "needs_gold": len(matches), "needs_ok": sum(1 for m in matches if m is not None),
        "isolation_leaks": leaks,
        "gold_evidence": len(gold_ev), "gold_evidence_found": len(set(gold_ev) & cites),
        "revalidation_gold": len(gold["expect_revalidation"]),
        "revalidation_found": len(set(gold["expect_revalidation"]) & reval_cites),
        "conflict_gold": gold["expect_conflict"], "conflict_flagged": conflict,
        "ambiguous": gold.get("ambiguous", False),
        "queries": {i: " ".join(t) for i, t in queries.items()},
    }


def aggregate(rows: list[dict]) -> dict:
    def rate(num, den):
        return round(num / den, 4) if den else None
    unamb = [r for r in rows if not r["ambiguous"]]
    verdicts = Counter(r["retrieval_verdict"] for r in rows)
    by_type: dict[str, list[bool]] = {}
    for r in rows:
        by_type.setdefault("|".join(r["gold"][0]) or "-", []).append(r["change_ok"])
    return {
        "turns": len(rows), "turns_unambiguous": len(unamb),
        "change_type_accuracy": rate(sum(r["change_ok"] for r in rows), len(rows)),
        "change_type_accuracy_unambiguous": rate(sum(r["change_ok"] for r in unamb), len(unamb)),
        "change_type_accuracy_by_gold": {k: {"n": len(v), "acc": rate(sum(v), len(v))} for k, v in sorted(by_type.items())},
        "need_query_correctness": rate(sum(r["needs_ok"] for r in rows), sum(r["needs_gold"] for r in rows)),
        "retrieval_verdicts": dict(sorted(verdicts.items())),
        "retrieval_decision_accuracy": rate(verdicts["ok"], len(rows)),
        "isolation_leaks": sum(len(r["isolation_leaks"]) for r in rows),
        "gold_evidence_recall_fixture": rate(sum(r["gold_evidence_found"] for r in rows),
                                             sum(r["gold_evidence"] for r in rows)),
        "expected_revalidation_recall": rate(sum(r["revalidation_found"] for r in rows),
                                             sum(r["revalidation_gold"] for r in rows)),
        "conflicts": {"gold": sum(r["conflict_gold"] for r in rows),
                      "flagged_when_gold": sum(r["conflict_flagged"] for r in rows if r["conflict_gold"]),
                      "flagged_without_gold": sum(r["conflict_flagged"] for r in rows if not r["conflict_gold"])},
        "retrievals": sum(r["retrievals"] for r in rows), "reused": sum(r["reused"] for r in rows),
    }
