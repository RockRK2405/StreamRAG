from streamrag.ledger import QueryLedger, containment, jaccard


def test_versions_lineage_and_relations():
    led = QueryLedger("s")
    q1, prev = led.create("u1", "fog signal", "t1", [(0, 3)], ["fog", "signal"], 100, "provisional", 1, "chunk", "r")
    assert q1.query_id == "Q1" and q1.version == 1 and prev is None and q1.relation == "initial" and q1.lineage_root == "Q1"
    q2, prev = led.create("u1", "fog signal storm", "t2", [], ["fog", "signal", "storm"], 200, "provisional", 2, "chunk", "r")
    assert q2.supersedes == "Q1" and q2.relation == "refines" and q2.lineage_root == "Q1" and q2.version == 2
    assert prev.superseded_by == "Q2" and prev.stale and led.get("Q1").stale
    q3, _ = led.create("u1", "telescope", "t3", [], ["telescop"], 300, "final", None, "utterance_end", "r")
    assert q3.relation == "replaces" and q3.lineage_root == "Q1"
    assert led.active("u1").query_id == "Q3"
    other, _ = led.create("u2", "lamp", "t", [], ["lamp"], 400, "provisional", 0, "chunk", "r")
    assert other.version == 1 and other.supersedes is None


def test_status_tracking_counts_and_similarity():
    led = QueryLedger("s")
    led.create("u1", "a", "t", [], ["fog", "signal"], 100, "provisional", 0, "chunk", "r")
    led.update("Q1", status="in_flight", retrieval_started_at_ms=150)
    assert [r.query_id for r in led.in_flight()] == ["Q1"] and led.last_issued_ms("u1") == 100   # decision time, not start
    led.create("u1", "b", "t", [], ["lamp"], 300, "provisional", 0, "chunk", "r")
    led.update("Q2", status="cancelled")
    assert led.retrieval_count("u1") == 1 and led.active("u1").query_id == "Q1"
    best, sim = led.most_similar(["fog", "signal", "storm"])
    assert best.query_id == "Q1" and abs(sim - 2 / 3) < 1e-9


def test_evidence_view_keeps_stale_lineage_after_active():
    led = QueryLedger("s")
    led.create("u1", "a", "t", [], ["fog"], 0, "provisional", 0, "chunk", "r")
    led.create("u1", "b", "t", [], ["fog", "storm"], 1, "provisional", 1, "chunk", "r")
    led.update("Q1", status="completed", evidence_ids=["E1", "E2"])
    led.update("Q2", status="completed", evidence_ids=["E2", "E3"])
    view = led.evidence_view("u1")
    assert [v["evidence_id"] for v in view] == ["E2", "E3", "E1"]
    assert view[0]["query_ids"] == ["Q2", "Q1"] and not view[0]["stale"] and view[2]["stale"]


def test_similarity_helpers():
    assert jaccard(set(), set()) == 1.0 and jaccard({"a"}, {"b"}) == 0.0
    assert containment({"a", "b"}, {"a", "b", "c"}) == 1.0 and containment(set(), {"a"}) == 0.0
