"""Controller contracts: the transparent decision object (Phase 4 brief §10; spec §8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract

Decision = Literal["WAIT", "RETRIEVE", "SKIP"]
Tick = Literal["chunk", "stability_timer", "utterance_end"]
SignalValue = float | int | str | bool | None


class RetrievalDecision(Contract):
    """One controller decision. ``signals`` holds the computed inputs; ``confidence`` is derived from them by a
    documented formula (docs/streaming/03). Nothing here is hand-set."""

    decision: Decision
    reason: str                                   # primary reason code (closed vocabulary)
    reasons: list[str] = []
    skip_kind: Literal["suppressed", "redundant", "budget", "not_worthy"] | None = None
    confidence: float = Field(ge=0, le=1)
    signals: dict[str, SignalValue] = {}
    tick: Tick
    trigger: Literal["provisional", "final"] | None = None
    query_text: str | None = None
    ledger_ref: str | None = None                 # query reused / compared against
    policy: str = "rules"
    session_id: str
    utterance_id: str
    trigger_chunk: int | None = None
    t_session_ms: float = Field(ge=0)


@dataclass
class ControllerInput:
    session_id: str
    utterance_id: str
    transcript: str
    tick: Tick
    now_ms: float
    trigger_chunk: int | None = None
    prev_tick_terms: list[str] = field(default_factory=list)   # query terms at the previous decision
    quiet: bool = False                                        # stability timer fired (no new words for quiet_ms)
    has_gaps: bool = False
