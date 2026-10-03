"""Deterministic, logged document normalization.

What it does (each step configurable): Unicode normalization, invisible-character removal, line-ending
and whitespace cleanup, consecutive duplicate-line removal (PDF extraction artifact), repeated
header/footer removal and page-number removal (paged sources only), de-hyphenation across line breaks,
joining of hard-wrapped lines (headings and list items are protected), page joining with page-offset
tracking.

What it never does: drop content lines that are not headers/footers/page markers, reorder text, or
rewrite words. Every removal is recorded in the document's ``normalization_log``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from streamrag.config.settings import NormalizationConfig, SectionsConfig
from streamrag.corpus.headings import LIST_ITEM, MD_HEADING, match_caps_heading, match_heading

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_PAGE_NUMBER = re.compile(r"^\s*(?:[-–—]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?(?:\s*[-–—])?\s*$", re.I)
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", '"', "”", ")")


@dataclass
class NormalizedText:
    text: str
    page_offsets: list[tuple[int, int, int]]   # (page_number, char_start, char_end); empty if not paged
    log: list[str]


def _basic(text: str, form: str) -> str:
    text = unicodedata.normalize(form, text).translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [re.sub(r"[  ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines)


def _drop_consecutive_duplicate_lines(text: str, log: list[str], where: str) -> str:
    out, prev, dropped = [], None, 0
    for ln in text.split("\n"):
        if ln and ln == prev:
            dropped += 1
            continue
        out.append(ln)
        prev = ln if ln else prev
    if dropped:
        log.append(f"{where}: removed {dropped} consecutive duplicate line(s)")
    return "\n".join(out)


def _edges(lines: list[str]) -> tuple[list[int], list[int]]:
    """Indices of top/bottom edge lines: 2 per side, or 1 on short pages (<6 non-empty lines) so that body text
    on short pages is never mistaken for running headers/footers."""
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    k = 2 if len(nonempty) >= 6 else 1
    return nonempty[:k], nonempty[-k:] if len(nonempty) > k else []


def _key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


def _header_footer_keys(pages: list[str], cfg: NormalizationConfig) -> tuple[set[str], set[str]]:
    if len(pages) < cfg.header_footer_min_pages:
        return set(), set()
    top: Counter[str] = Counter()
    bottom: Counter[str] = Counter()
    for page in pages:
        lines = page.split("\n")
        t, b = _edges(lines)
        top.update({_key(lines[i]) for i in t})
        bottom.update({_key(lines[i]) for i in b})
    need = cfg.header_footer_min_fraction * len(pages)
    return {k for k, c in top.items() if c >= need}, {k for k, c in bottom.items() if c >= need}


def _strip_edges(page: str, keys: tuple[set[str], set[str]], strip_numbers: bool) -> tuple[str, list[str]]:
    lines = page.split("\n")
    t, b = _edges(lines)
    drop = {i for i in t if _key(lines[i]) in keys[0]} | {i for i in b if _key(lines[i]) in keys[1]}
    if strip_numbers:
        drop |= {i for i in t + b if _PAGE_NUMBER.match(lines[i])}
    removed = [lines[i] for i in sorted(drop)]
    return "\n".join(ln for i, ln in enumerate(lines) if i not in drop), removed


def _is_protected_line(line: str, sec: SectionsConfig) -> bool:
    if MD_HEADING.match(line) or line.startswith("|") or LIST_ITEM.match(line):
        return True
    return (match_heading(line, sec.max_heading_words, allow_markdown=False) is not None
            or match_caps_heading(line, sec.max_heading_words) is not None)


def _join_wrapped(text: str, sec: SectionsConfig) -> str:
    """Join hard-wrapped lines inside paragraphs; headings become standalone blocks."""
    blocks = re.split(r"\n{2,}", text)
    out_blocks = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        merged: list[str] = []
        for ln in lines:
            if not merged or _is_protected_line(ln, sec):
                merged.append(ln)                       # first line, or a heading / list item / table row
                continue
            prev = merged[-1]
            # Trade-off: a wrapped list item whose first line looks like a title ("1. Pick Fruit") is not
            # joined; plain-text headings without a following blank line are the more common case.
            prev_is_heading = bool(MD_HEADING.match(prev)) or prev.startswith("|") or (
                match_heading(prev, sec.max_heading_words, allow_markdown=False) is not None) or (
                match_caps_heading(prev, sec.max_heading_words) is not None)
            if prev_is_heading:
                merged.append(ln)                       # never glue body text onto a heading or table row
            else:
                merged[-1] = prev + " " + ln            # wrapped continuation (incl. of a list item)
        # headings get their own block so the section parser sees them cleanly
        cur: list[str] = []
        for ln in merged:
            if MD_HEADING.match(ln) or match_heading(ln, sec.max_heading_words, allow_markdown=False):
                if cur:
                    out_blocks.append("\n".join(cur))
                    cur = []
                out_blocks.append(ln)
            else:
                cur.append(ln)
        if cur:
            out_blocks.append("\n".join(cur))
    return "\n\n".join(out_blocks)


def normalize_pages(pages: list[str], paged: bool, cfg: NormalizationConfig, sec: SectionsConfig,
                    fmt: str) -> NormalizedText:
    log: list[str] = []
    cleaned = [_basic(p, cfg.unicode_form) for p in pages]
    cleaned = [_drop_consecutive_duplicate_lines(p, log, f"page {i + 1}" if paged else "document")
               for i, p in enumerate(cleaned)]
    if paged:
        keys = _header_footer_keys(cleaned, cfg) if cfg.strip_repeated_headers_footers else (set(), set())
        stripped = []
        for i, p in enumerate(cleaned):
            p2, removed = _strip_edges(p, keys, cfg.strip_page_numbers)
            if removed:
                log.append(f"page {i + 1}: removed header/footer/page-marker line(s): {removed}")
            stripped.append(p2)
        cleaned = stripped
    if cfg.dehyphenate:
        cleaned = [_HYPHEN_BREAK.sub(r"\1\2", p) for p in cleaned]
    if cfg.join_wrapped_lines:
        cleaned = [_join_wrapped(p, sec) for p in cleaned]
    else:
        cleaned = [re.sub(r"\n{3,}", "\n\n", p).strip() for p in cleaned]

    # Join pages, tracking offsets. A paragraph that continues across a page break is joined by a space.
    text, offsets = "", []
    for i, p in enumerate(cleaned):
        p = p.strip()
        if text and p:
            continues = not text.endswith(_TERMINAL_PUNCT) and p[:1].islower()
            text += " " if continues else "\n\n"
        start = len(text)
        text += p
        if paged:
            offsets.append((i + 1, start, len(text)))
    return NormalizedText(text=text, page_offsets=offsets, log=log)


def page_at(offsets: list[tuple[int, int, int]], char_pos: int) -> int | None:
    for page, start, end in offsets:
        if start <= char_pos <= end:
            return page
    return offsets[-1][0] if offsets else None
