"""QueryBuilder: current transcript -> one retrieval-ready query (Phase 4: single active query, no decomposition).

Removes conversational filler (anywhere) and the leading preamble (greetings, backchannels, request heads such
as "can you tell me"), strips dangling trailing function words, and keeps everything else verbatim: entities,
numbers, constraints, terminology and **negation** (never removed, never stripped). The output records the
character spans of every kept token in the transcript, so the query is fully traceable to what was said.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from streamrag.controller.lexicon import Lexicon, find_phrases, tokenize

_LEADING_CONNECTIVES = {"and", "so", "but", "also", "then", "about", "like", "or"}


@dataclass
class BuiltQuery:
    text: str
    spans: list[tuple[int, int]]
    removed: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.text


class QueryBuilder:
    def __init__(self, lex: Lexicon) -> None:
        self.lex = lex

    def build(self, transcript: str) -> BuiltQuery:
        lx = self.lex
        toks = tokenize(transcript)
        lows = [t.lower for t in toks]
        drop = [False] * len(toks)
        for i, j, _ in find_phrases(lows, lx.filler_phrases):
            for k in range(i, j):
                drop[k] = True
        for k, w in enumerate(lows):
            if w in lx.fillers and w not in lx.negations:
                drop[k] = True
        # leading preamble: request heads, greetings, backchannels, connectives
        heads = {i: j for i, j, _ in find_phrases(lows, lx.request_heads + lx.social_phrases + lx.backchannel_phrases)}
        k = 0
        while k < len(toks):
            if drop[k]:
                k += 1
            elif k in heads:
                for m in range(k, heads[k]):
                    drop[m] = True
                k = heads[k]
            elif lows[k] in lx.social or lows[k] in lx.backchannel or lows[k] in _LEADING_CONNECTIVES:
                drop[k] = True
                k += 1
            else:
                break
        # trailing dangling function words (never strip a negation or an auxiliary verb: "what do workers do")
        k = len(toks) - 1
        while k >= 0 and (drop[k] or (lows[k] in lx.dangling and lows[k] not in lx.negations
                                      and lows[k] not in lx.aux_questions)):
            drop[k] = True
            k -= 1
        kept = [t for t, d in zip(toks, drop) if not d]
        return BuiltQuery(text=" ".join(t.text for t in kept), spans=[(t.start, t.end) for t in kept],
                          removed=[t.text for t, d in zip(toks, drop) if d])
