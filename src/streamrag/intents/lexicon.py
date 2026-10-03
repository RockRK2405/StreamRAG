"""Intent lexicon (configs/intent_lexicon.yaml) merged with the controller lexicon (fillers, request heads,
question words, auxiliaries, social/backchannel words). Generic English only; disclosed configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from streamrag.controller.lexicon import Lexicon as ControllerLexicon
from streamrag.models.intents import INTENT_TYPES


def _phrases(items) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted({tuple(str(p).lower().split()) for p in items}, key=lambda t: (-len(t), t)))


@dataclass(frozen=True)
class IntentLexicon:
    base: ControllerLexicon
    coordinators: frozenset[str]
    addition_phrases: tuple[tuple[str, ...], ...]
    alternatives: frozenset[str]
    comparison_cues: tuple[tuple[str, ...], ...]
    pair_openers: frozenset[str]
    generic_nouns: frozenset[str]
    aspect_nouns: frozenset[str]
    topic_prepositions: frozenset[str]
    restriction_prepositions: frozenset[str]
    focus_phrases: tuple[tuple[str, ...], ...]          # single markers + phrases
    condition_phrases: tuple[tuple[str, ...], ...]
    scope_all_phrases: tuple[tuple[str, ...], ...]
    correction_phrases: tuple[tuple[str, ...], ...]
    correction_replacement_phrases: tuple[tuple[str, ...], ...]
    correction_words: frozenset[str]
    anaphora: frozenset[str]
    locative_anaphora: frozenset[str]
    follow_up_openers: tuple[tuple[str, ...], ...]
    request_verbs: frozenset[str]
    embedding_verbs: frozenset[str]
    type_cues: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]   # ordered (type, phrases)
    retraction_phrases: tuple[tuple[str, ...], ...] = ()               # Phase 6
    constraint_words: frozenset[str] = frozenset()
    elliptical_heads: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def load(cls, path: Path, base: ControllerLexicon) -> "IntentLexicon":
        d = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in d.items():
            vals = v.values() if isinstance(v, dict) else [v]
            for vv in vals:
                bad = [x for x in (vv if isinstance(vv, list) else []) if not isinstance(x, str)]
                if bad:
                    raise ValueError(f"intent lexicon '{k}' has non-string entries {bad}; quote them in {path}")

        def low(k: str) -> frozenset[str]:
            return frozenset(str(w).lower() for w in d.get(k, []))
        types = d.get("intent_types", {})
        unknown = set(types) - set(INTENT_TYPES)
        if unknown:
            raise ValueError(f"unsupported intent types in {path}: {sorted(unknown)}")
        return cls(
            base=base, coordinators=low("coordinators"), addition_phrases=_phrases(d.get("addition_phrases", [])),
            alternatives=low("alternatives"), comparison_cues=_phrases(d.get("comparison_cues", [])),
            pair_openers=low("pair_openers"), generic_nouns=low("generic_nouns"), aspect_nouns=low("aspect_nouns"),
            topic_prepositions=low("topic_prepositions"), restriction_prepositions=low("restriction_prepositions"),
            focus_phrases=_phrases(list(d.get("focus_markers", [])) + list(d.get("focus_phrases", []))),
            condition_phrases=_phrases(list(d.get("condition_markers", [])) + list(d.get("condition_phrases", []))),
            scope_all_phrases=_phrases(d.get("scope_all_phrases", [])),
            correction_phrases=_phrases(d.get("correction_phrases", [])),
            correction_replacement_phrases=_phrases(d.get("correction_replacement_phrases", [])),
            correction_words=low("correction_words"), anaphora=low("anaphora"),
            locative_anaphora=low("locative_anaphora"), follow_up_openers=_phrases(d.get("follow_up_openers", [])),
            request_verbs=low("request_verbs"), embedding_verbs=low("embedding_verbs"),
            type_cues=tuple((t, _phrases(v)) for t, v in types.items()),     # lexicon order = precedence
            retraction_phrases=_phrases(d.get("retraction_markers", [])), constraint_words=low("constraint_words"),
            elliptical_heads=_phrases(d.get("elliptical_follow_up_heads", [])))

    # convenience views over the controller lexicon
    @property
    def question_words(self) -> frozenset[str]:
        return self.base.question_words

    @property
    def aux(self) -> frozenset[str]:
        return self.base.aux_questions

    @property
    def request_heads(self) -> tuple[tuple[str, ...], ...]:
        return self.base.request_heads
