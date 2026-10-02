"""Dialog-act classification for retrieval-worthiness (Phase 2 §8.3, §15).

``RuleActClassifier`` (default) combines a generic lexicon with *structure* (question form, the object of a
transformation verb) and *corpus statistics* (whether remaining content words are anchors in the indexed corpus
vocabulary). ``PrototypeActClassifier`` is the model-based alternative (embedding nearest-centroid over generic
seed phrases) kept behind the same interface for the rule-vs-model ablation (Exp 8).

Acts: INFO_REQUEST, PRESENTATION, SOCIAL, BACKCHANNEL, META (suppressible), UNKNOWN (declarative/casual).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from streamrag.controller.lexicon import Lexicon, find_phrases, tokenize

SUPPRESSED_ACTS = {"PRESENTATION": "presentation_restructure", "SOCIAL": "social_ack",
                   "BACKCHANNEL": "backchannel", "META": "meta_conversation"}
_POLITE = {"please", "you", "me", "for", "can", "could", "would", "will", "make", "give", "put", "do", "now", "again"}


@dataclass
class ActResult:
    act: str
    confidence: float
    cues: list[str] = field(default_factory=list)
    anchors_present: bool = False


class RuleActClassifier:
    name = "rules"

    def __init__(self, lex: Lexicon, stopwords: frozenset[str], is_anchor: Callable[[str], bool]) -> None:
        self.lex, self.stop, self.is_anchor = lex, stopwords, is_anchor

    def classify(self, text: str) -> ActResult:
        lx = self.lex
        lows = [t.lower for t in tokenize(text)]
        if not lows:
            return ActResult("BACKCHANNEL", 0.9, ["empty"])
        covered = set()
        for i, j, _ in find_phrases(lows, lx.filler_phrases + lx.backchannel_phrases):
            covered.update(range(i, j))
        if all(k in covered or w in lx.fillers or w in lx.backchannel for k, w in enumerate(lows)):
            return ActResult("BACKCHANNEL", 0.9, ["only_fillers_or_continuers"])
        anchors = any(self.is_anchor(w) for w in lows if w not in self.stop)
        if find_phrases(lows, lx.more_info_phrases):
            return ActResult("INFO_REQUEST", 0.75, ["asks_for_more_information"], anchors)
        if find_phrases(lows, lx.meta_phrases):
            return ActResult("META", 0.85, ["about_the_conversation"], anchors)
        pres_hits = find_phrases(lows, lx.presentation_phrases)
        if pres_hits or any(w in lx.presentation_verbs for w in lows):
            in_phrase = {k for i, j, _ in pres_hits for k in range(i, j)}
            residual = [w for k, w in enumerate(lows)
                        if k not in in_phrase and w not in self.stop and w not in lx.presentation_verbs
                        and w not in lx.presentation_objects and w not in lx.format_words and w not in lx.languages
                        and w not in lx.fillers and w not in lx.social and w not in lx.backchannel
                        and w not in lx.aux_questions and w not in _POLITE]
            if not residual:
                return ActResult("PRESENTATION", 0.9, ["transforms_prior_output"], anchors)
            if len(residual) == 1 and not any(self.is_anchor(w) for w in residual):
                return ActResult("PRESENTATION", 0.8, ["transforms_prior_output", "one_unknown_word"], anchors)
            if any(self.is_anchor(w) for w in residual):   # "summarize the X rule": transform a corpus topic
                return ActResult("INFO_REQUEST", 0.8, ["transform_verb_on_corpus_topic"], anchors)
        first = next((w for w in lows if w not in lx.fillers and w not in lx.social and w not in lx.backchannel), "")
        wh = any(w in lx.question_words for w in lows)
        head = bool(find_phrases(lows, lx.request_heads))
        if "?" in text or wh or head or first in lx.aux_questions:
            cues = [c for c, ok in (("question_mark", "?" in text), ("wh_word", wh), ("request_head", head),
                                    ("aux_inversion", first in lx.aux_questions)) if ok]
            return ActResult("INFO_REQUEST", 0.9 if ("?" in text or wh) else 0.85, cues, anchors)
        social = any(w in lx.social for w in lows) or bool(find_phrases(lows, lx.social_phrases))
        if social and not anchors:
            return ActResult("SOCIAL", 0.85, ["social_marker_without_corpus_anchor"], anchors)
        return ActResult("UNKNOWN", 0.5, ["declarative_or_casual"], anchors)


class PrototypeActClassifier:
    """Model-based alternative: nearest class centroid of embedded generic seed phrases (ablation only)."""

    name = "prototype"

    def __init__(self, lex: Lexicon, embed: Callable[[list[str]], np.ndarray], fallback: RuleActClassifier) -> None:
        self.fallback = fallback
        self.embed = embed
        self.labels = sorted(lex.prototypes)
        cents = []
        for lab in self.labels:
            v = embed(lex.prototypes[lab]).mean(axis=0)
            cents.append(v / max(np.linalg.norm(v), 1e-12))
        self.centroids = np.vstack(cents) if cents else np.zeros((0, 1))

    def classify(self, text: str) -> ActResult:
        if not tokenize(text) or not self.labels:
            return self.fallback.classify(text)
        v = self.embed([text])[0]
        sims = self.centroids @ v
        order = np.argsort(-sims)
        best, second = float(sims[order[0]]), float(sims[order[1]]) if len(order) > 1 else 0.0
        conf = float(np.clip(0.5 + (best - second) * 5.0, 0.0, 1.0))   # margin -> confidence (transparent)
        rule = self.fallback.classify(text)
        return ActResult(self.labels[order[0]], round(conf, 3), [f"prototype_sim={best:.3f}"], rule.anchors_present)
