"""Tokenization for decomposition: words *and* punctuation, with character spans (clause boundaries need commas,
full stops and question marks, which the Phase 4 word tokenizer discards)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[,.;:?!&]")
HARD_PUNCT = frozenset({".", "?", "!", ";"})


@dataclass(frozen=True)
class Tok:
    text: str
    lower: str
    start: int
    end: int

    @property
    def punct(self) -> bool:
        return not self.text[0].isalnum()


def tokenize(text: str) -> list[Tok]:
    return [Tok(m.group(0), m.group(0).lower().replace("’", "'"), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def match_phrase(lows: list[str], i: int, phrases: tuple[tuple[str, ...], ...]) -> int:
    """Length of the longest phrase in ``phrases`` starting at ``lows[i]`` (0 if none). Phrases are sorted longest
    first, so the first hit is the longest."""
    for ph in phrases:
        n = len(ph)
        if tuple(lows[i:i + n]) == ph:
            return n
    return 0
