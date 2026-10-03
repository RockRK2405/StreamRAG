"""Deterministic, section-bounded chunking.

Strategies (``chunking.strategy``):
  * ``section``   — one chunk per section; sections longer than ``max_tokens`` fall back to ``paragraph``.
  * ``paragraph`` — pack whole paragraphs up to ``target_tokens`` (hard cap ``max_tokens``); oversized
                    paragraphs are split at sentence boundaries; a paragraph ending with ":" (list lead-in)
                    stays with the following paragraph; overlap = trailing whole sentences (only when a split
                    falls inside a paragraph).
  * ``token``     — fixed sliding windows of ``target_tokens`` words with ``overlap_tokens`` overlap (baseline).

Tokens are regex words (``\\w+``). A chunk never crosses a section boundary and its text is always an exact
slice of the normalized document text (traceability invariant).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from streamrag.config.settings import ChunkingConfig
from streamrag.corpus.headings import LIST_ITEM
from streamrag.models.corpus import CorpusSection

_WORD = re.compile(r"\w+")
_SENT_END = re.compile(r"[.!?]+[\"”’)\]]*\s+")
_LIST_BREAK = re.compile(r"\n(?=" + LIST_ITEM.pattern.lstrip("^") + ")")
_ABBREV = {"e.g", "i.e", "etc", "vs", "no", "nos", "dr", "mr", "mrs", "ms", "st", "approx", "fig", "sec",
           "art", "inc", "ltd", "co", "cf", "al", "jr", "sr", "dept", "est", "max", "min"}


def count_tokens(text: str) -> int:
    return len(_WORD.findall(text))


def sentence_spans(text: str, base: int = 0) -> list[tuple[int, int]]:
    """Sentence (and list-item) spans of ``text`` as absolute offsets; never splits after abbreviations."""
    cuts: list[tuple[int, int]] = []   # (end_of_sentence, start_of_next)
    for m in _SENT_END.finditer(text):
        before = re.search(r"(\w+(?:\.\w+)*)$", text[: m.start()])
        word = before.group(1).lower() if before else ""
        nxt = text[m.end(): m.end() + 1]
        if word in _ABBREV or (len(word) == 1 and word.isalpha()):
            continue
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"“(['‘-•*"):
            continue
        cuts.append((m.start() + len(m.group(0).rstrip()), m.end()))
    for m in _LIST_BREAK.finditer(text):
        cuts.append((m.start(), m.end()))
    cuts.sort()
    spans, start = [], 0
    for end, nxt in cuts:
        if end > start and text[start:end].strip():
            spans.append((start, end))
        start = max(start, nxt)
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    # trim whitespace inside spans
    out = []
    for s, e in spans:
        seg = text[s:e]
        s2 = s + (len(seg) - len(seg.lstrip()))
        e2 = e - (len(seg) - len(seg.rstrip()))
        if e2 > s2:
            out.append((base + s2, base + e2))
    return out


@dataclass
class _Unit:
    start: int
    end: int
    tokens: int
    para: int
    para_start: bool
    glue_next: bool
    mid_sentence: bool = False


@dataclass
class ChunkSpan:
    start: int
    end: int
    starts_mid_sentence: bool


def _word_windows(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    words = [(start + m.start(), start + m.end()) for m in _WORD.finditer(text[start:end])]
    if not words:
        return []
    step = max(1, size - overlap)
    out, i = [], 0
    while True:
        j = min(len(words), i + size)
        out.append((words[i][0], words[j - 1][1]))
        if j == len(words):
            break
        i += step
    return out


def _units(section: CorpusSection, doc_text: str, cfg: ChunkingConfig) -> list[_Unit]:
    units: list[_Unit] = []
    paras = section.paragraphs
    for pi, p in enumerate(paras):
        glue = cfg.keep_lead_in_with_list and p.text.rstrip().endswith(":") and pi + 1 < len(paras)
        ptok = count_tokens(p.text)
        if ptok <= cfg.max_tokens:
            units.append(_Unit(p.char_start, p.char_end, ptok, pi, True, glue))
            continue
        pieces: list[tuple[int, int, bool]] = []
        for s, e in sentence_spans(doc_text[p.char_start:p.char_end], p.char_start):
            if count_tokens(doc_text[s:e]) <= cfg.max_tokens:
                pieces.append((s, e, False))
            else:  # pathological run-on sentence: fall back to word windows
                for k, (ws, we) in enumerate(_word_windows(doc_text, s, e, cfg.max_tokens, 0)):
                    pieces.append((ws, we, k > 0))
        for k, (s, e, mid) in enumerate(pieces):
            units.append(_Unit(s, e, count_tokens(doc_text[s:e]), pi, k == 0,
                               glue and k == len(pieces) - 1, mid))
    return units


def _pack(units: list[_Unit], doc_text: str, cfg: ChunkingConfig) -> list[ChunkSpan]:
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    for u in units:
        if not cur:
            cur = [u]
            continue
        cur_tok = sum(x.tokens for x in cur)
        fits_target = cur_tok + u.tokens <= cfg.target_tokens
        fits_max = cur_tok + u.tokens <= cfg.max_tokens
        if fits_target or (fits_max and (cur[-1].glue_next or cur_tok < cfg.min_tokens)):
            cur.append(u)
            continue
        groups.append(cur)
        overlap: list[_Unit] = []
        if cfg.overlap_tokens and not u.para_start:          # split inside a paragraph -> carry context
            budget = cfg.overlap_tokens
            for prev in reversed(cur):
                if prev.para != u.para or prev.tokens > budget:
                    break
                overlap.insert(0, prev)
                budget -= prev.tokens
            while overlap and sum(x.tokens for x in overlap) + u.tokens > cfg.max_tokens:
                overlap.pop(0)
        cur = overlap + [u]
    if cur:
        groups.append(cur)
    spans = [ChunkSpan(g[0].start, g[-1].end, g[0].mid_sentence) for g in groups]
    # merge a tiny trailing piece into its predecessor when it fits
    if len(spans) >= 2:
        last, prev = spans[-1], spans[-2]
        if count_tokens(doc_text[last.start:last.end]) < cfg.min_tokens and \
                count_tokens(doc_text[prev.start:last.end]) <= cfg.max_tokens:
            spans[-2:] = [ChunkSpan(prev.start, last.end, prev.starts_mid_sentence)]
    return spans


def chunk_section(section: CorpusSection, doc_text: str, cfg: ChunkingConfig) -> list[ChunkSpan]:
    if not section.paragraphs:
        return []
    body_start, body_end = section.paragraphs[0].char_start, section.paragraphs[-1].char_end
    if cfg.strategy == "token":
        sent_starts = {s for s, _ in sentence_spans(doc_text[body_start:body_end], body_start)}
        return [ChunkSpan(s, e, s not in sent_starts)
                for s, e in _word_windows(doc_text, body_start, body_end, cfg.target_tokens, cfg.overlap_tokens)]
    if cfg.strategy == "section" and count_tokens(doc_text[body_start:body_end]) <= cfg.max_tokens:
        return [ChunkSpan(body_start, body_end, False)]
    return _pack(_units(section, doc_text, cfg), doc_text, cfg)
