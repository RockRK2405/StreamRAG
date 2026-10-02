import pytest
from pydantic import ValidationError

from streamrag.models import EventType, SessionEnd, SessionStart, TelemetryEvent, TranscriptChunk, UtteranceEnd
from streamrag.streaming import simulator as sim
from streamrag.streaming.events import EventBus, canonical_for_replay


def test_event_bus_envelope_ids_and_clock(tmp_path):
    clock = {"t": 0.0}
    bus = EventBus("sess", lambda: clock["t"], tmp_path / "t.jsonl")
    clock["t"] = 1500.0
    e = bus.emit(EventType.RETRIEVAL_DECISION, "retrieval_controller", {"decision": "WAIT", "wall": {"x": 1}}, "u1", 1000.0)
    bus.close()
    assert e.event_id == "sess:000000" and e.t_session_ms == 1500.0 and e.t_stream_s == 0.5
    assert (tmp_path / "t.jsonl").read_text().count("\n") == 1
    c = canonical_for_replay(e)
    assert "t_wall_ms" not in c and "wall" not in c["payload"] and c["payload"]["decision"] == "WAIT"


def test_telemetry_event_rejects_bad_ids_and_types():
    with pytest.raises(ValidationError):
        TelemetryEvent(event_id="x", type="RETRIEVAL_DECISION", session_id="s", seq=0, t_wall_ms=0, component="c")
    with pytest.raises(ValidationError):
        TelemetryEvent(event_id="s:000000", type="NOT_A_TYPE", session_id="s", seq=0, t_wall_ms=0, component="c")


def test_stream_fixed_interval():
    evs = sim.stream(["I want to know", "about the policy", "for X"], interval_ms=200, end_gap_ms=500)
    assert isinstance(evs[0], SessionStart) and isinstance(evs[-1], SessionEnd)
    chunks = [e for e in evs if isinstance(e, TranscriptChunk)]
    assert [c.payload.timestamp_s for c in chunks] == [0.0, 0.2, 0.4]
    end = next(e for e in evs if isinstance(e, UtteranceEnd))
    assert end.payload.timestamp_s == pytest.approx(0.9) and end.payload.last_chunk_index == 2


def test_timed_is_deterministic_and_monotonic():
    a = sim.timed(["one two", "three four five", "six"], jitter=0.2, seed=3)
    b = sim.timed(["one two", "three four five", "six"], jitter=0.2, seed=3)
    assert [e.canonical_json() for e in a] == [e.canonical_json() for e in b]
    ts = [e.payload.timestamp_s for e in a if isinstance(e, TranscriptChunk)]
    assert ts == sorted(ts) and ts[0] > 0


def test_inputs_roundtrip(tmp_path):
    evs = sim.stream(["a b", "c d"], interval_ms=250)
    sim.write_inputs(tmp_path / "in.jsonl", evs)
    back = sim.read_inputs(tmp_path / "in.jsonl")
    assert [e.canonical_json() for e in back] == [e.canonical_json() for e in evs]
    assert sim.parse_inline(" a | b  |  | c ") == ["a", "b", "c"]
