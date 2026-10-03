import json

from streamrag.cli import main

from conftest import FIX, REPO


def test_corpus_status_reports_official_not_available(capsys, tmp_path):
    rc = main(["--config", str(REPO / "configs/default.yaml"), "corpus-status", "--corpus", str(tmp_path / "none")])
    assert rc == 0 and "OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE" in capsys.readouterr().out
    main(["--config", str(REPO / "configs/default.yaml"), "corpus-status", "--corpus", str(FIX / "corpus")])
    out = capsys.readouterr().out
    assert "OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE" in out and "detected = TEST_FIXTURE" in out


def test_build_and_search_json(capsys, tmp_path):
    common = ["--config", str(REPO / "configs/default.yaml"), "--set", f"paths.index_root={tmp_path}",
              "--set", "telemetry.log_level=WARNING"]
    assert main(common + ["build-index", "--corpus", str(FIX / "corpus"), "--embedder", "hashing"]) == 0
    assert "TEST FIXTURE" in capsys.readouterr().out
    assert main(common + ["--set", "dense.embedder=hashing", "search", "fog signal", "--corpus", str(FIX / "corpus"),
                          "--json"]) == 0
    es = json.loads(capsys.readouterr().out)
    assert es["items"][0]["citation"] == "fixture_lighthouse_manual §3"


def test_errors_are_explicit(capsys, tmp_path):
    rc = main(["--config", str(REPO / "configs/default.yaml"), "build-index", "--corpus", str(tmp_path / "missing")])
    assert rc == 2 and "CorpusNotFoundError" in capsys.readouterr().err


def test_stream_session_mode_prints_changes_and_replays(capsys, tmp_path):
    common = ["--config", str(REPO / "configs/default.yaml"), "--set", f"paths.index_root={tmp_path}",
              "--set", "telemetry.log_level=WARNING", "--set", "dense.embedder=hashing"]
    assert main(common + ["build-index", "--corpus", str(FIX / "corpus")]) == 0
    capsys.readouterr()
    trace = tmp_path / "t.jsonl"
    assert main(common + ["stream", "--corpus", str(FIX / "corpus"), "--session", "--interval-ms", "300",
                          "--text", "What are the rules | for ladders | in the orchard?",
                          "--text", "Specifically | overnight.", "--trace", str(trace)]) == 0
    out = capsys.readouterr().out
    assert "CONSTRAINT_ADDITION" in out and "ANSWER     A2" in out and "net=['CONSTRAINT_ADDITION']" in out
    assert main(common + ["replay", "--corpus", str(FIX / "corpus"), str(trace)]) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True
