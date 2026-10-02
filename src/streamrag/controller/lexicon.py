"""Generic lexicon (configs/controller_lexicon.yaml) + word tokenization with character spans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class Token:
    text: str
    lower: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.group(0).lower().replace("’", "'"), m.start(), m.end()) for m in _WORD.finditer(text)]


def _phrases(items: list[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted({tuple(p.lower().split()) for p in items}, key=lambda t: (-len(t), t)))


@dataclass(frozen=True)
class Lexicon:
    fillers: frozenset[str]
    filler_phrases: tuple[tuple[str, ...], ...]
    request_heads: tuple[tuple[str, ...], ...]
    question_words: frozenset[str]
    aux_questions: frozenset[str]
    dangling: frozenset[str]
    social: frozenset[str]
    social_phrases: tuple[tuple[str, ...], ...]
    backchannel: frozenset[str]
    backchannel_phrases: tuple[tuple[str, ...], ...]
    presentation_verbs: frozenset[str]
    presentation_phrases: tuple[tuple[str, ...], ...]
    presentation_objects: frozenset[str]
    format_words: frozenset[str]
    languages: frozenset[str]
    meta_phrases: tuple[tuple[str, ...], ...]
    more_info_phrases: tuple[tuple[str, ...], ...]
    prototypes: dict[str, list[str]] = field(default_factory=dict)
    negations: frozenset[str] = frozenset({"no", "not", "never", "nor", "none", "cannot", "without", "except",
                                           "unless", "neither", "n't"})

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        d = yaml.safe_load(Path(path).read_text()) or {}

        def low(k: str) -> frozenset[str]:
            vals = d.get(k, [])
            bad = [v for v in vals if not isinstance(v, str)]
            if bad:   # e.g. unquoted yes/no/on parsed as booleans by YAML
                raise ValueError(f"lexicon '{k}' has non-string entries {bad}; quote them in {path}")
            return frozenset(w.lower() for w in vals)
        return cls(fillers=low("fillers"), filler_phrases=_phrases(d.get("filler_phrases", [])),
                   request_heads=_phrases(d.get("request_heads", [])), question_words=low("question_words"),
                   aux_questions=low("aux_questions"), dangling=low("dangling"), social=low("social"),
                   social_phrases=_phrases(d.get("social_phrases", [])), backchannel=low("backchannel"),
                   backchannel_phrases=_phrases(d.get("backchannel_phrases", [])),
                   presentation_verbs=low("presentation_verbs"),
                   presentation_phrases=_phrases(d.get("presentation_phrases", [])),
                   presentation_objects=low("presentation_objects"), format_words=low("format_words"),
                   languages=low("languages"), meta_phrases=_phrases(d.get("meta_phrases", [])),
                   more_info_phrases=_phrases(d.get("more_info_phrases", [])),
                   prototypes=dict(d.get("prototypes", {})))


def find_phrases(lowers: list[str], phrases: tuple[tuple[str, ...], ...]) -> list[tuple[int, int, tuple[str, ...]]]:
    """Non-overlapping phrase matches (longest first) as (start_idx, end_idx_exclusive, phrase)."""
    taken = [False] * len(lowers)
    out = []
    for ph in phrases:
        n = len(ph)
        for i in range(0, len(lowers) - n + 1):
            if tuple(lowers[i:i + n]) == ph and not any(taken[i:i + n]):
                out.append((i, i + n, ph))
                for k in range(i, i + n):
                    taken[k] = True
    return sorted(out)
