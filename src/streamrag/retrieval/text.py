"""Lexical analyzer shared by index and query side (identical processing => consistent matching).

lowercase + NFKC -> tokens (letters / numbers) -> spoken number words to digits -> stopwords -> Snowball stem.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from functools import lru_cache
from importlib import resources

import snowballstemmer

_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+")
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1000}


@lru_cache(maxsize=8)
def load_stopwords(name: str) -> frozenset[str]:
    text = resources.files("streamrag.resources").joinpath(name).read_text(encoding="utf-8")
    return frozenset(ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#"))


def number_words_to_digits(tokens: list[str]) -> list[str]:
    out: list[str] = []
    total = current = 0
    active = False
    for t in tokens + [""]:
        if t in _UNITS or t in _TENS:
            current += _UNITS.get(t, 0) + _TENS.get(t, 0)
            active = True
        elif t in _SCALES and active:
            current = max(current, 1) * _SCALES[t]
            if _SCALES[t] >= 1000:
                total, current = total + current, 0
        else:
            if active:
                out.append(str(total + current))
                total = current = 0
                active = False
            if t:
                out.append(t)
    return out


class Analyzer:
    def __init__(self, stemming: str = "snowball", stopwords: str = "stopwords_en.txt",
                 normalize_number_words: bool = True) -> None:
        self.stemming = stemming
        self.stopwords = load_stopwords(stopwords)
        self.normalize_number_words = normalize_number_words
        # Snowball stemmer objects keep per-call state: one per thread (the Phase 8 runtime analyses text on several
        # worker threads at once - a shared instance raised IndexError under concurrency)
        self._stemmer = snowballstemmer.stemmer("english") if stemming == "snowball" else None
        self._stemmers: dict[int, object] = {}

    def raw_tokens(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        toks = [t.replace(",", "") if t[0].isdigit() else t for t in _TOKEN.findall(text)]
        return number_words_to_digits(toks) if self.normalize_number_words else toks

    def tokens(self, text: str) -> list[str]:
        toks = [t for t in self.raw_tokens(text) if t not in self.stopwords]
        if self._stemmer is None:
            return toks
        tid = threading.get_ident()
        st = self._stemmers.get(tid)
        if st is None:
            st = self._stemmers[tid] = snowballstemmer.stemmer("english")
        return st.stemWords(toks)

    def config(self) -> dict:
        return {"stemming": self.stemming, "normalize_number_words": self.normalize_number_words,
                "stopwords_count": len(self.stopwords)}
