"""Phase 6 test helpers: adaptive session over the TEST FIXTURE corpus (fictional; hashing embedder, no models).

Everything here is TEST_FIXTURE_ONLY: the fixture documents and the utterances below exercise mechanics (change
types, lifecycles, lineage), never retrieval quality.
"""

from __future__ import annotations

import pytest

from streamrag.models.events import SessionEnd, SessionStart
from streamrag.session import AdaptivePipeline, FullRestartPipeline
from streamrag.streaming import run_virtual
from streamrag.streaming import simulator as sim

from streaming_helpers import hashing_stack


def session_cfg(cfg, **session_kw):
    return cfg.model_copy(update={
        "multi_intent": cfg.multi_intent.model_copy(update={"enabled": True}),
        "session": cfg.session.model_copy(update={"enabled": True, **session_kw})})


@pytest.fixture(scope="session")
def p6_stack(fixture_bundle):
    cfg, b = fixture_bundle
    return hashing_stack(session_cfg(cfg), b)


def run_turns(stack, turns: list[str], restart: bool = False, gap_ms: float = 3000.0):
    """Synchronous adaptive (or full-restart) session; returns (pipeline, [TurnResult])."""
    p = FullRestartPipeline(stack) if restart else AdaptivePipeline(stack)
    results = [p.process(f"u{n}", t, n * gap_ms) for n, t in enumerate(turns, start=1)]
    return p, results


def engine_of(p):
    return p.last.engine if isinstance(p, FullRestartPipeline) else p.engine


def stream_turns(stack, turns: list[list[str]], interval_ms: float = 400, gap_ms: float = 2500, **kw):
    """Virtual streaming run of a multi-utterance session (each turn = list of chunks)."""
    evs = [SessionStart(session_id="sim")]
    off = 0.0
    for n, chunks in enumerate(turns, start=1):
        evs += sim.stream(chunks, interval_ms=interval_ms, utterance_id=f"u{n}", offset_ms=off, wrap_session=False)
        off += interval_ms * len(chunks) + gap_ms
    evs.append(SessionEnd(session_id="sim"))
    return run_virtual(stack.cfg, stack.service, stack.policy, evs, index_hash=stack.index_hash,
                       intent_stack=stack.intent_stack, **kw)


def of(events, type_name: str):
    return [e for e in events if e.type.value == type_name]
