"""Streaming metrics from a session's runtime event log (Phase 8 events).

  update_gaps_ms        time between successive answer updates of a turn (ANSWER_COMPLETED drafts / finals,
                        ANSWER_DELTA, ANSWER_COMMITTED)
  revisions             answer versions released for the turn after the first (ANSWER_REVISED + extra commits)
  time_to_useful_ms     first answer update whose text states >= 1 expected key fact (None: never)
  time_to_final_ms      the turn's last ANSWER_COMMITTED
  stale_update_rate     STALE_RESULT_DISCARDED / (commits applied + discarded) over the session
  ordering_errors       user-visible events whose output_seq is not strictly increasing
  interrupted           1 if a turn that received input never reached TURN_COMPLETED
"""

from __future__ import annotations

UPDATE = ("ANSWER_COMPLETED", "ANSWER_DELTA", "ANSWER_COMMITTED")


def turn(events, first_input_ms: float, utterance_id: str, keys: list[list[str]]) -> dict:
    ups, useful, final, revised = [], None, None, 0
    commits = 0
    for e in events:
        if e.utterance_id != utterance_id:
            continue
        t = e.type.value
        dt = e.t_wall_ms - first_input_ms
        if t in UPDATE:
            ups.append(dt)
            text = (e.payload.get("text") or "").lower()
            if useful is None and text and any(all(k.lower() in text for k in ks) for ks in keys if ks):
                useful = dt
        if t == "ANSWER_COMMITTED":
            final = dt
            commits += 1
        if t == "ANSWER_REVISED":
            revised += 1
    gaps = [round(b - a, 3) for a, b in zip(ups, ups[1:])]
    return {"answer_updates": len(ups), "update_gaps_ms": gaps, "revisions": revised + max(0, commits - 1),
            "time_to_useful_ms": None if useful is None else round(useful, 3),
            "time_to_final_ms": None if final is None else round(final, 3),
            "completed": any(e.type.value == "TURN_COMPLETED" and e.utterance_id == utterance_id for e in events)}


def session(events) -> dict:
    seqs = [e.output_seq for e in events if e.output_seq is not None]
    errors = sum(1 for a, b in zip(seqs, seqs[1:]) if b <= a)
    stale = sum(1 for e in events if e.type.value == "STALE_RESULT_DISCARDED")
    commits = sum(1 for e in events if e.type.value in ("ANSWER_COMMITTED", "RETRIEVAL_COMPLETED"))
    return {"ordering_errors": errors, "stale_updates": stale,
            "stale_update_rate": (stale / (stale + commits)) if stale + commits else None}
