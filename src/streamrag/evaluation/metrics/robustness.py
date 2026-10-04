"""Robustness metrics over fault-injection runs (one row per injected scenario x question).

  recovered            the turn completed and released an answer (validated, or an explicit
                       "not established" / degraded answer) despite the fault
  degraded_success     recovered while a DEGRADED_MODE_CHANGED event was emitted (the fault was noticed)
  incorrect_answer     the released answer matches a forbidden pattern or misses every expected key fact while
                       stating some value (a wrong answer, not an abstention)
  propagated           a fault in one session changed the outcome of a concurrent, unfaulted session
"""

from __future__ import annotations


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)

    def rate(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return (sum(v) / len(v)) if v else None
    return {"runs": n, "recovery_rate": rate("recovered"), "degraded_success_rate": rate("degraded_success"),
            "incorrect_answer_rate": rate("incorrect_answer"), "failure_propagation_rate": rate("propagated")}
