"""Streaming metrics computed only from structured events (never from log text).

Per utterance (all times on the session stream clock, ms):
  lead_time_ms                   = utterance_finalized_at - first retrieval_started_at   (>0 => early retrieval)
  ttfr_ms                        = first retrieval_started_at - first chunk_received_at
  post_final_retrieval_latency_ms= max(0, final-query retrieval_completed_at - utterance_finalized_at)
  evidence_ready_slack_ms        = utterance_finalized_at - final-query retrieval_completed_at (negative => wait)

Multi-intent mode (Phase 5, QUERY_GENERATED events): the same metrics per intent under ``intents``; the utterance's
"final query" is then the set of final queries of its active intents, and readiness waits for the slowest one.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from streamrag.models.events import TelemetryEvent


def _t(ev: TelemetryEvent) -> float:
    return float(ev.t_session_ms or 0.0)


def utterance_stats(events: list[TelemetryEvent]) -> dict[str, dict]:
    by_u: dict[str, dict] = defaultdict(lambda: {"retrievals": {}, "decisions": Counter(), "skips": Counter(),
                                                  "first_chunk_ms": None, "finalized_ms": None, "chunks": 0,
                                                  "start_ms": None, "final_query_id": None, "cancelled": 0,
                                                  "controller_wall_ms": [], "end_reason": None,
                                                  "intent_final": {}, "inactive_intents": set(),
                                                  "intent_versions": 0, "fused": None, "ledger_hits": set()})
    for ev in events:
        uid = ev.utterance_id
        if uid is None:
            continue
        u = by_u[uid]
        p = ev.payload
        if ev.type.value == "CHUNK_RECEIVED" and not p.get("status", "").startswith("rejected"):
            if p.get("status") not in ("duplicate",):
                u["chunks"] += 1
            if u["first_chunk_ms"] is None:
                u["first_chunk_ms"] = _t(ev)
                u["start_ms"] = _t(ev) - (ev.t_stream_s or 0.0) * 1000.0
        elif ev.type.value == "RETRIEVAL_DECISION":
            u["decisions"][p["decision"]] += 1
            if p["decision"] == "SKIP":
                u["skips"][p.get("skip_kind") or "other"] += 1
            if "wall" in p:
                u["controller_wall_ms"].append(p["wall"]["controller_ms"])
        elif ev.type.value in ("QUERY_UPDATED", "QUERY_GENERATED"):
            u["retrievals"][p["query_id"]] = {"query_id": p["query_id"], "query": p["query_text"], "trigger": p["trigger"],
                                              "created_ms": _t(ev), "start_ms": None, "start_s": None, "end_ms": None,
                                              "status": None, "stale": False, "supersedes": p.get("supersedes"),
                                              "intent_id": ev.intent_id}
            u["final_query_id"] = p["query_id"]
            if ev.intent_id:
                u["intent_final"][ev.intent_id] = p["query_id"]
        elif ev.type.value == "INTENTS_UPDATED":
            u["intent_versions"] += 1
        elif ev.type.value == "INTENT_SUPERSEDED" or (ev.type.value == "INTENT_UPDATED"
                                                      and p.get("status") == "DROPPED"):
            u["inactive_intents"].add(ev.intent_id)
        elif ev.type.value == "EVIDENCE_FUSED" and p.get("final"):
            u["fused"] = p
        elif ev.type.value == "RETRIEVAL_SKIPPED" and p.get("reason") == "ledger_hit" and ev.intent_id:
            u["ledger_hits"].add(ev.intent_id)
        elif ev.type.value == "RETRIEVAL_CANCELLED":
            r = u["retrievals"].get(p["query_id"])
            if r:
                r["status"] = "cancelled"
            u["cancelled"] += 1
            if u["final_query_id"] == p["query_id"]:
                u["final_query_id"] = p.get("superseded_by")
            if ev.intent_id and u["intent_final"].get(ev.intent_id) == p["query_id"]:
                u["intent_final"][ev.intent_id] = p.get("superseded_by")
        elif ev.type.value == "RETRIEVAL_STARTED":
            r = u["retrievals"].get(p["query_id"])
            if r:
                r["start_ms"], r["start_s"] = _t(ev), ev.t_stream_s
        elif ev.type.value == "RETRIEVAL_COMPLETED":
            r = u["retrievals"].get(p["query_id"])
            if r:
                r["end_ms"], r["status"], r["stale"] = _t(ev), p["status"], p.get("stale", False)
        elif ev.type.value == "UTTERANCE_FINALIZED":
            u["finalized_ms"], u["end_reason"] = _t(ev), p.get("reason")
    out = {}
    for uid, u in by_u.items():
        rets = list(u["retrievals"].values())
        started = [r for r in rets if r["start_ms"] is not None]
        first_start = min((r["start_ms"] for r in started), default=None)
        fin = u["finalized_ms"]
        final = u["retrievals"].get(u["final_query_id"]) if u["final_query_id"] else None
        lead = round(fin - first_start, 3) if (fin is not None and first_start is not None) else None
        texts = Counter(r["query"] for r in started)
        slack = round(fin - final["end_ms"], 3) if (final and final["end_ms"] is not None and fin is not None) else None
        intents = {}
        for iid in u["ledger_hits"]:                 # intents served by an earlier utterance's evidence
            u["intent_final"].setdefault(iid, None)
        for iid, fq in u["intent_final"].items():
            mine = [r for r in started if r.get("intent_id") == iid]
            fs = min((r["start_ms"] for r in mine), default=None)
            fr = u["retrievals"].get(fq) if fq else None
            ilead = round(fin - fs, 3) if (fin is not None and fs is not None) else None
            islack = round(fin - fr["end_ms"], 3) if (fr and fr["end_ms"] is not None and fin is not None) else None
            intents[iid] = {"active": iid not in u["inactive_intents"], "queries": len(mine),
                            "first_retrieval_start_ms": fs, "lead_time_ms": ilead,
                            "retrieved_early": bool(ilead is not None and ilead > 0), "final_query_id": fq,
                            "ledger_hit": iid in u["ledger_hits"],
                            "evidence_ready_slack_ms": islack}
        act = [v for v in intents.values() if v["active"]]
        if act:                                   # multi-intent: readiness waits for the slowest active intent
            slacks = [v["evidence_ready_slack_ms"] for v in act]
            slack = min(slacks) if all(x is not None for x in slacks) else None
        ctl = u["controller_wall_ms"]
        out[uid] = {
            "chunks": u["chunks"], "first_chunk_ms": u["first_chunk_ms"], "finalized_ms": fin, "end_reason": u["end_reason"],
            "retrieval_count": len(started), "cancelled": u["cancelled"],
            "first_retrieval_start_ms": first_start, "lead_time_ms": lead, "retrieved_early": bool(lead is not None and lead > 0),
            "ttfr_ms": round(first_start - u["first_chunk_ms"], 3) if (first_start is not None and u["first_chunk_ms"] is not None) else None,
            "final_query_id": u["final_query_id"], "evidence_ready_slack_ms": slack,
            "post_final_retrieval_latency_ms": (max(0.0, -slack) if slack is not None else None),
            "duplicate_retrievals": sum(c - 1 for c in texts.values() if c > 1),
            "stale_completions": sum(1 for r in started if r["stale"] and r["end_ms"] is not None),
            "decisions": dict(u["decisions"]), "skips": dict(u["skips"]),
            **({"intents": intents, "intent_count": len(act), "intent_set_versions": u["intent_versions"],
                "unified_items": len(u["fused"]["items"]) if u["fused"] else None,
                "intent_coverage": u["fused"]["intent_coverage"] if u["fused"] else None} if intents else {}),
            "wall": {"controller_ms_max": round(max(ctl), 4) if ctl else None},
            "retrievals": started,
        }
    return out
