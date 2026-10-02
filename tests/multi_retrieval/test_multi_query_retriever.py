"""Per-intent retrieval through the Phase 3 service: dispatch strategies, budget, error isolation (brief §16-19)."""

import time

import pytest

from streamrag.models.intents import IntentQuery
from streamrag.multi_retrieval import MultiQueryRetriever

from streaming_helpers import hashing_stack, make_es

SPAN = {"utterance_id": "u1", "start": 0, "end": 1, "text": "x"}


def q(iid, text):
    return IntentQuery(intent_id=iid, intent_version=1, utterance_id="u1", text=text,
                       components=[{"text": text, "source": "intent", "span": SPAN}])


QUERIES = [q("I1", "how high are wicks trimmed"), q("I2", "where are ladders stored overnight"),
           q("I3", "when is the telescope recalibrated")]


@pytest.fixture(scope="module")
def service(fixture_bundle):
    cfg, b = fixture_bundle
    return hashing_stack(cfg, b).service


def test_all_dispatch_modes_return_the_phase3_results(service):
    """Same Phase 3 retrieval for every intent, whatever the dispatch strategy."""
    mr = MultiQueryRetriever(service, max_concurrent=3)
    expected = {x.intent_id: [e.chunk_id for e in service.retrieve(x.text).items] for x in QUERIES}
    for mode in ("sequential", "parallel", "batched"):
        r = mr.retrieve(QUERIES, mode)
        assert {o.intent_id: [e.chunk_id for e in o.evidence.items] for o in r.outcomes} == expected, mode
        assert r.skipped == [] and all(o.error is None for o in r.outcomes)


def test_budget_dispatches_in_priority_order_and_records_skips(service):
    mr = MultiQueryRetriever(service, max_queries=2)
    r = mr.retrieve(QUERIES, "sequential", priority={"I1": 3, "I2": 1, "I3": 2})
    assert [o.intent_id for o in r.outcomes] == ["I2", "I3"]
    assert r.skipped == [("I1", "max_queries=2")]


class SlowBackend:
    def __init__(self, delay_s=0.05, fail_on=None):
        self.delay_s, self.fail_on = delay_s, fail_on

    def retrieve(self, req, options=None):
        time.sleep(self.delay_s)
        if self.fail_on and self.fail_on in req.query:
            raise RuntimeError("backend down")
        return make_es(req.query, ["a", "b"])


def test_parallel_overlaps_and_shortens_wall_time():
    mr = MultiQueryRetriever(SlowBackend(0.05), max_concurrent=3)
    seq, par = mr.retrieve(QUERIES, "sequential"), mr.retrieve(QUERIES, "parallel")
    assert seq.max_concurrency == 1 and par.max_concurrency == 3
    assert par.total_ms < seq.total_ms * 0.7                  # 3 x 50 ms sleeps overlap
    assert par.critical_path_ms >= 45


def test_one_failing_intent_does_not_fail_the_others():
    r = MultiQueryRetriever(SlowBackend(0.0, fail_on="ladders"), max_concurrent=3).retrieve(QUERIES, "parallel")
    by = r.by_intent()
    assert by["I2"].error.startswith("RuntimeError") and by["I2"].evidence is None
    assert by["I1"].evidence is not None and by["I3"].evidence is not None
