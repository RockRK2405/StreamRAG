"""Heading recognition shared by normalization (to protect headings from line-joining) and section
parsing. Purely lexical, generic, deterministic; no corpus-specific vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SECTION_SIGN = re.compile(r"^§\s*(\d+(?:\.\d+)*)\s*[-–—:.)]?\s*(.*)$")
SECTION_WORD = re.compile(r"^(?:section|article|clause|part|chapter)\s+(\d+(?:\.\d+)*)\s*[-–—:.)]?\s*(.*)$", re.I)
NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,4})[.)]?\s+(\S.*)$")
LIST_ITEM = re.compile(r"^\s*(?:[-*•▪◦·]|\(?\d+[.)]|\(?[a-zA-Z][.)]|\([ivxIVX]+\))\s+")
_TERMINAL = (".", ",", ";", ":", "?", "!")


@dataclass(frozen=True)
class HeadingMatch:
    kind: str            # markdown | section_sign | section_word | numbered | caps
    level: int
    number: str | None   # native section number, if any
    title: str


def _words(s: str) -> int:
    return len(re.findall(r"\w+", s))


def match_heading(line: str, max_words: int, allow_markdown: bool = True, allow_numbered: bool = True) -> HeadingMatch | None:
    s = line.strip()
    if not s or "\n" in s:
        return None
    if allow_markdown:
        m = MD_HEADING.match(s)
        if m:
            title = m.group(2).strip()
            num = NUMBERED.match(title) or SECTION_SIGN.match(title) or SECTION_WORD.match(title)
            if num and _words(num.group(2)) <= max_words:
                return HeadingMatch("markdown", len(m.group(1)), num.group(1), num.group(2).strip() or title)
            return HeadingMatch("markdown", len(m.group(1)), None, title)
    if not allow_numbered:
        return None
    for kind, rx in (("section_sign", SECTION_SIGN), ("section_word", SECTION_WORD)):
        m = rx.match(s)
        if m and _words(m.group(2)) <= max_words:
            return HeadingMatch(kind, m.group(1).count(".") + 1, m.group(1), m.group(2).strip())
    m = NUMBERED.match(s)
    if m:
        title = m.group(2).strip()
        if (_words(title) <= max_words and not title.endswith(_TERMINAL)
                and (title[0].isupper() or title[0].isdigit())):
            return HeadingMatch("numbered", m.group(1).count(".") + 1, m.group(1), title)
    return None


def match_caps_heading(line: str, max_words: int) -> HeadingMatch | None:
    s = line.strip()
    letters = [c for c in s if c.isalpha()]
    if (len(letters) >= 3 and all(c.isupper() for c in letters) and _words(s) <= max_words
            and not s.endswith((".", ",", ";"))):
        return HeadingMatch("caps", 1, None, s.title())
    return None
