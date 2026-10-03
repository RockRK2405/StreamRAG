"""Shared harness for the Phase 8 runtime measurements (fixture domain; NOT REPORTABLE).

Stacks use the real Phase 3-7 components: bge-small ONNX retrieval, the pinned NLI verifier, and either the local
LLM (Ollama ``qwen3:4b``) or the SYNTHETIC ``SimulatedLLM``. Workloads stream the dev-suite questions
(``eval/dev_grounded``) chunk by chunk. Every result file states which LLM and which injected latencies were used.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
CORPORA = {"fixture": "tests/fixtures/corpus", "grounding": "tests/fixtures/corpus_grounding",
           "conflict": "tests/fixtures/corpus_conflict", "injection": "tests/fixtures/corpus_injection"}
BANNER = "DEV FIXTURE DOMAIN - SYNTHETIC WORKLOAD - NOT AN OFFICIAL BENCHMARK RESULT - NOT REPORTABLE"

from streamrag.answer_state.resources import GroundingResources  # noqa: E402
from streamrag.citations.mapper import ChunkCatalog  # noqa: E402
from streamrag.claims.nli import NliModel  # noqa: E402
from streamrag.config import load_config  # noqa: E402
from streamrag.retrieval import build_index  # noqa: E402
from streamrag.streaming.factory import build_stack  # noqa: E402

_NLI = None


def nli():
    global _NLI
    if _NLI is None:
        _NLI = NliModel.load(REPO / "models", "nli-deberta-v3-xsmall")
    return _NLI


def stack_for(name: str, index_root: Path, backend=None, **overrides):
    ov = {"paths.corpus": str(REPO / CORPORA[name]), "paths.index_root": str(index_root / name),
          "telemetry.log_level": "ERROR", "multi_intent.enabled": True, "session.enabled": True,
          "generation.enabled": True, "generation.backend": "extractive"}
    ov.update(overrides)
    cfg = load_config(REPO / "configs" / "default.yaml", ov, base_dir=REPO)
    st = build_stack(cfg, build_index(cfg).path)
    an = st.bundle.analyzer
    st._grounding = GroundingResources(ChunkCatalog.from_bundle(st.bundle), nli(), backend,
                                       lambda t: list(dict.fromkeys(an.tokens(t))))
    return st


def with_cfg(stack, **dotted):
    """A copy of ``stack`` whose config has the dotted overrides (shares index, models, intent stack)."""
    cfg = stack.cfg
    for key, val in dotted.items():
        parts = key.split(".")
        objs = [cfg]
        for p in parts[:-1]:
            objs.append(getattr(objs[-1], p))
        new = objs[-1].model_copy(update={parts[-1]: val})
        for obj, p in zip(reversed(objs[:-1]), reversed(parts[:-1])):
            new = obj.model_copy(update={p: new})
        cfg = new
    st = stack.with_config(cfg)
    st._intent_stack = stack.intent_stack
    st._grounding = stack.grounding
    return st


def suite(corpus: str | None = None) -> list[dict]:
    cases = [json.loads(p.read_text()) for p in sorted((REPO / "eval" / "dev_grounded").glob("G*.json"))]
    return [c for c in cases if corpus is None or c["corpus"] == corpus]


def chunks(text: str, words: int = 3) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + words]) for i in range(0, len(w), words)]


def pct(values) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}

    def q(p):
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    return {"n": len(v), "p50": round(q(0.5), 3), "p95": round(q(0.95), 3), "max": round(v[-1], 3),
            "mean": round(statistics.fmean(v), 3)}


class ResourceProbe:
    """Process CPU time, peak RSS, peak thread count (sampled) over a measured region."""

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval = interval_s
        self.peak_threads = threading.active_count()
        self._stop = threading.Event()

    def __enter__(self) -> "ResourceProbe":
        self.cpu0, self.wall0 = time.process_time(), time.perf_counter()
        self.rss0 = _rss_mb()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_threads = max(self.peak_threads, threading.active_count() - 1)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        self.cpu_s = time.process_time() - self.cpu0
        self.wall_s = time.perf_counter() - self.wall0
        self.rss_peak_mb = _rss_mb()

    def summary(self) -> dict:
        return {"cpu_s": round(self.cpu_s, 3), "wall_s": round(self.wall_s, 3),
                "cpu_utilisation": round(self.cpu_s / self.wall_s, 3) if self.wall_s else None,
                "peak_threads": self.peak_threads, "rss_peak_mb_process_lifetime": round(self.rss_peak_mb, 1)}


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if platform.system() == "Darwin" else r / 1024


def open_fds() -> int:
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return -1


def meta(**extra) -> dict:
    return {"REPORTABLE": False, "banner": BANNER, "machine": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "embedder": "bge-small-en-v1.5 (ONNX)", "verifier": "nli-deberta-v3-xsmall",
            **extra}


def write(name: str, obj: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def turn_metrics(events, first_input: dict[str, float]) -> dict[str, dict]:
    from streamrag.runtime.telemetry import turn_milestones
    return turn_milestones(events, first_input)


def first_inputs(events) -> dict[str, float]:
    """Wall time of each utterance's first processed chunk (same reference for every pipeline)."""
    out: dict[str, float] = {}
    for e in events:
        if e.type.value == "CHUNK_RECEIVED" and e.utterance_id not in out:
            out[e.utterance_id] = e.t_wall_ms
    return out
