"""Late-arriving-detail gate (Phase 6; docs/session/03).

The Phase 4 controller judges an utterance on its own; "Specifically at night.", "Only for students." or "Actually,
ignore the student restriction." are not request-shaped, so it may call them not (yet) retrieval-worthy. When the
session has an active need, such a decision still opens the interpretation gate if the utterance carries a
late-detail cue (lexicon-driven, configs/intent_lexicon.yaml):

  a correction or retraction phrase anywhere ("actually I meant", "ignore", "never mind", ...)
  a focus / condition marker or a restriction preposition opening the utterance ("specifically", "only", "if",
  "for ...", "and for ...")

and the utterance has content beyond the cue words (a bare "Specifically" or "Actually, ignore" mid-stream waits for
its content). Suppressed acts (backchannels, fillers, closings) never open it: their decision reason is not in
``LATE_DETAIL_GATE_REASONS``. Phase 5's correction gate is the special case "correction phrase only".
"""

from __future__ import annotations

from collections.abc import Callable

from streamrag.intents.text import match_phrase, tokenize

LATE_DETAIL_GATE_REASONS = frozenset({"not_retrieval_worthy", "not_yet_retrieval_worthy", "low_specificity",
                                      "awaiting_stability"})


def late_detail_cue(transcript: str, lex, terms_fn: Callable[[str], list[str]] | None = None) -> bool:
    lows = [t.lower for t in tokenize(transcript) if not t.punct and t.lower not in lex.base.fillers]
    if not lows:
        return False
    cue = [False] * len(lows)
    for i in range(len(lows)):
        n = max(match_phrase(lows, i, lex.correction_phrases), match_phrase(lows, i, lex.retraction_phrases))
        for j in range(i, i + n):
            cue[j] = True
    found = any(cue)
    if not found:
        n = max(match_phrase(lows, 0, lex.focus_phrases), match_phrase(lows, 0, lex.condition_phrases))
        if lows[0] in lex.coordinators and len(lows) > 1 and lows[1] in lex.restriction_prepositions:
            n = 2
        elif lows[0] in lex.restriction_prepositions:
            n = max(n, 1)
        found = n > 0
        for j in range(n):
            cue[j] = True
    if not found:
        return False
    if terms_fn is None:
        return True
    rest = [w for w, c in zip(lows, cue) if not c and w not in lex.constraint_words]
    return bool(terms_fn(" ".join(rest)))


def late_detail_gate(decision, has_active_need: bool, transcript: str, lex,
                     terms_fn: Callable[[str], list[str]] | None = None) -> bool:
    return decision.reason in LATE_DETAIL_GATE_REASONS and has_active_need \
        and late_detail_cue(transcript, lex, terms_fn)
