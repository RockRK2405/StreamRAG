"""Latency metrics. Milestones per turn (ms from the turn's first transcript chunk, wall clock):
  TTFT  time to first user-visible event of the turn (first event the system emits for it)
  TTFE  time to first evidence (first retrieval result with hits)
  TTFA  time to first answer content (draft or final)
  TTVA  time to validated answer (final answer released after verification)
  total time to the last event of the turn
Also measured from the end of the utterance (what a user waits after speaking).
Percentiles: p50 always; p90 needs n >= 10, p95 n >= 20, p99 n >= 100 - otherwise None (sample too small).
"""

from __future__ import annotations

import statistics

MIN_N = {"p90": 10, "p95": 20, "p99": 100}


def _q(v: list[float], p: float) -> float:
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def percentiles(values) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0, "mean": None, "std": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    out = {"n": len(v), "mean": round(statistics.fmean(v), 3), "std": round(statistics.stdev(v), 3) if len(v) > 1
           else None, "p50": round(_q(v, 0.5), 3), "max": round(v[-1], 3)}
    for name, p in (("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        out[name] = round(_q(v, p), 3) if len(v) >= MIN_N[name] else None
    return out


def milestones(events, first_input_ms: float, utterance_id: str) -> dict:
    """From one turn's runtime events (TelemetryEvent objects)."""
    m = {"ttft": None, "ttfe": None, "ttfa": None, "ttva": None, "total": None, "utterance_end": None}
    for e in events:
        if e.utterance_id != utterance_id:
            continue
        dt = round(e.t_wall_ms - first_input_ms, 3)
        t = e.type.value
        m["total"] = dt
        if m["ttft"] is None and t not in ("CHUNK_RECEIVED", "TRANSCRIPT_UPDATED"):
            m["ttft"] = dt
        if m["ttfe"] is None and ((t == "RETRIEVAL_PARTIAL" and e.payload.get("hits")) or
                                  (t == "RETRIEVAL_COMPLETED" and (e.payload.get("evidence_ids")
                                                                   or e.payload.get("n_results")))):
            m["ttfe"] = dt
        if m["ttfa"] is None and t in ("ANSWER_COMPLETED", "ANSWER_CLAIM_READY", "ANSWER_STARTED"):
            m["ttfa"] = dt
        if m["ttva"] is None and t == "ANSWER_COMMITTED":
            m["ttva"] = dt
        if m["utterance_end"] is None and t == "UTTERANCE_FINALIZED":
            m["utterance_end"] = dt
    if m["ttva"] is not None and m["utterance_end"] is not None:
        m["ttva_after_end"] = round(m["ttva"] - m["utterance_end"], 3)
    if m["ttfa"] is not None and m["utterance_end"] is not None:
        m["ttfa_after_end"] = round(m["ttfa"] - m["utterance_end"], 3)
    return m
