import json

import pytest

from streamrag.bench.streaming import run_streaming_benchmark
from streamrag.cli import main
from streamrag.replay import ReplayEngine, dump_trace, inputs_from_trace, read_trace
from streamrag.streaming import run_virtual, utterance_stats
from streamrag.streaming import simulator as sim

from conftest import FIX, REPO
from streaming_helpers import hashing_stack, with_controller


@pytest.fixture(scope="module")
def stack(fixture_bundle):
    cfg, b = fixture_bundle
    return hashing_stack(cfg, b)


INPUTS = sim.stream(["so I wanted to ask", "how high the wicks", "should be trimmed"], interval_ms=400)


def test_replay_is_identical_in_virtual_mode(stack, tmp_path):
    r = run_virtual(stack.cfg, stack.service, stack.policy, INPUTS, index_hash=stack.index_hash)
    dump_trace(r.events, tmp_path / "t.jsonl")
    eng = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash)
    rep = eng.replay_file(tmp_path / "t.jsonl")
    assert rep.identical and rep.n_original == rep.n_replayed == len(r.events)


def test_replay_detects_behavior_change(stack):
    r = run_virtual(stack.cfg, stack.service, stack.policy, INPUTS, index_hash=stack.index_hash)
    other = hashing_stack(with_controller(stack.cfg, strategy="end_only"), stack.bundle)
    rep = ReplayEngine(other.cfg, other.service, other.policy, other.index_hash).replay(r.events)
    assert not rep.identical and rep.differences


def test_realtime_trace_replays_with_identical_behavior(stack):
    """Realtime timestamps jitter, so exact equality is not expected; decisions, queries and outcomes must match."""
    from streamrag.streaming import run_realtime
    r = run_realtime(stack.cfg, stack.service, stack.policy, INPUTS, index_hash=stack.index_hash)
    rep = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash).replay(r.events)
    assert rep.source_mode == "realtime" and not rep.identical
    assert rep.behavior_identical and rep.ok, rep.behavior_differences
    tampered = [e.model_copy(deep=True) for e in r.events]
    dec = next(e for e in tampered if e.type.value == "RETRIEVAL_DECISION" and e.payload["decision"] == "RETRIEVE")
    dec.payload["decision"] = "WAIT"
    rep2 = ReplayEngine(stack.cfg, stack.service, stack.policy, stack.index_hash).replay(tampered)
    assert not rep2.behavior_identical and not rep2.ok


def test_inputs_reconstructed_including_duplicates(stack, tmp_path):
    evs = INPUTS[:3] + [INPUTS[2]] + INPUTS[3:]
    r = run_virtual(stack.cfg, stack.service, stack.policy, evs, index_hash=stack.index_hash)
    dump_trace(r.events, tmp_path / "t.jsonl")
    rebuilt = inputs_from_trace(read_trace(tmp_path / "t.jsonl"))
    assert [e.type for e in rebuilt] == [e.type for e in evs]


def test_metrics_from_structured_events(stack):
    r = run_virtual(stack.cfg, stack.service, stack.policy, sim.stream(["so I wanted to ask", "how high the wicks"],
                                                                        interval_ms=1000, end_gap_ms=500),
                    index_hash=stack.index_hash)
    st = utterance_stats(r.events)["u1"]
    assert st["first_chunk_ms"] == 0.0 and st["finalized_ms"] == 1500.0
    assert st["first_retrieval_start_ms"] == 1000.0 and st["lead_time_ms"] == 500.0 and st["ttfr_ms"] == 1000.0
    assert st["post_final_retrieval_latency_ms"] == 0.0 and st["retrieval_count"] >= 1


def test_streaming_benchmark_harness_is_not_reportable(stack, tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    for name in ("fog_storm-authored", "ack_thanks-authored", "shorter-authored"):
        (cases / f"{name}.json").write_text((REPO / "eval" / "dev_streaming" / f"{name}.json").read_text())
    m = run_streaming_benchmark(stack, cases, ["controller", "end_only"], tmp_path / "out")
    assert m["REPORTABLE"] is False and "NOT AN OFFICIAL" in m["banner"]
    c, a = m["policies"]["controller"], m["policies"]["end_only"]
    assert c["false_retrieval_rate"] == 0.0 and a["false_retrieval_rate"] == 1.0
    assert (tmp_path / "out" / "traces" / "controller" / "fog_storm-authored.jsonl").exists()
    assert json.loads((tmp_path / "out" / "run_manifest.json").read_text())["REPORTABLE"] is False


def test_cli_stream_and_replay(tmp_path, capsys):
    common = ["--config", str(REPO / "configs/default.yaml"), "--set", f"paths.index_root={tmp_path}",
              "--set", "telemetry.log_level=ERROR", "--set", "dense.embedder=hashing"]
    assert main(common + ["build-index", "--corpus", str(FIX / "corpus")]) == 0
    trace = tmp_path / "trace.jsonl"
    assert main(common + ["stream", "--corpus", str(FIX / "corpus"), "--text", "I need | information | about the fog signal",
                          "--interval-ms", "300", "--trace", str(trace)]) == 0
    out = capsys.readouterr().out
    assert "CONTROLLER WAIT" in out and "RETRIEVAL  START" in out and trace.exists()
    assert main(common + ["replay", str(trace), "--corpus", str(FIX / "corpus")]) == 0
    assert '"identical": true' in capsys.readouterr().out
