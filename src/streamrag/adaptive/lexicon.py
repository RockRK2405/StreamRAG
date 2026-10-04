"""Retrieval lexicon (configs/retrieval_lexicon.yaml): domain-neutral cue phrases and bounded synonym groups."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


# Question / request words: not content the evidence has to repeat (grammar, not corpus vocabulary).
QUESTION_WORDS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "much", "many", "long", "often", "tell",
    "explain", "describe", "list", "need", "needs", "know", "want", "please", "can", "could", "would", "does", "do",
    "is", "are", "the", "about", "give", "show", "say", "detail", "details", "inform", "information", "anything",
    "something", "rule", "rules", "there", "any", "get", "take", "takes", "make", "makes", "go", "goes",
    # fillers: general words that carry no subject ("bring somebody along")
    "somebody", "someone", "anybody", "anyone", "anything", "everything", "everybody", "everyone", "along", "also",
    "really", "actually", "maybe", "perhaps", "still", "already", "just", "way", "kind", "sort", "thing", "things",
    "exactly", "ever", "okay", "ok", "hello", "hi", "thanks", "um", "uh", "like", "bit", "lot", "lots"})


@dataclass(frozen=True)
class RetrievalLexicon:
    current: tuple[str, ...] = ()
    past: tuple[str, ...] = ()
    relation_cues: tuple[str, ...] = ()
    reference_cues: tuple[str, ...] = ()
    comparison_cues: tuple[str, ...] = ()
    constraint_markers: tuple[str, ...] = ()
    synonyms: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def load(cls, path: Path) -> "RetrievalLexicon":
        d = yaml.safe_load(Path(path).read_text()) or {}
        t = d.get("temporal") or {}
        low = lambda xs: tuple(str(x).lower() for x in (xs or []))  # noqa: E731
        return cls(current=low(t.get("current")), past=low(t.get("past")),
                   relation_cues=low(d.get("relation_cues")), reference_cues=low(d.get("reference_cues")),
                   comparison_cues=low(d.get("comparison_cues")),
                   constraint_markers=low(d.get("constraint_markers")),
                   synonyms=tuple(low(g) for g in (d.get("synonyms") or [])))

    @staticmethod
    def has_phrase(text: str, phrases: tuple[str, ...]) -> str | None:
        """First phrase found as whole words in ``text`` (case-insensitive), or None."""
        low = " " + re.sub(r"[^\w:]+", " ", text.lower()) + " "
        for p in phrases:
            if f" {p} " in low:
                return p
        return None
