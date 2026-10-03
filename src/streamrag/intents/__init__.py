"""Multi-intent decomposition (Phase 5): transcript -> IntentSet (docs/multi_intent/)."""

from __future__ import annotations

from streamrag.config.settings import StreamRagConfig
from streamrag.controller.lexicon import Lexicon
from streamrag.intents.decomposer import (
    Decomposition,
    DecompositionContext,
    DraftIntent,
    IntentDecomposer,
    PriorIntent,
    anchor_norm_for,
    corpus_coordination_pairs,
)
from streamrag.intents.lexicon import IntentLexicon
from streamrag.retrieval.store import IndexBundle
from streamrag.retrieval.text import load_stopwords


def make_decomposer(cfg: StreamRagConfig, bundle: IndexBundle) -> IntentDecomposer:
    """Decomposer over the loaded index: anchors use the index's own IDF table, and fixed 'x and y' phrases are
    read from the indexed corpus (so they are never split into two needs)."""
    base = Lexicon.load(cfg.controller.lexicon)
    lex = IntentLexicon.load(cfg.multi_intent.lexicon, base)
    an = bundle.analyzer
    return IntentDecomposer(
        lex, load_stopwords(cfg.lexical.stopwords), terms_fn=lambda t: list(dict.fromkeys(an.tokens(t))),
        idf_fn=bundle.bm25.term_idf, anchor_norm=anchor_norm_for(bundle.bm25.n_docs),
        anchor_floor=cfg.controller.anchor_idf_floor,
        corpus_pairs=corpus_coordination_pairs([c.text for c in bundle.chunks]),
        max_intents=cfg.multi_intent.max_intents, duplicate_jaccard=cfg.multi_intent.duplicate_jaccard,
        carryover=cfg.multi_intent.carryover, cooccur_fn=bundle.bm25.cooccurrence)


__all__ = ["Decomposition", "DecompositionContext", "DraftIntent", "IntentDecomposer", "IntentLexicon",
           "PriorIntent", "make_decomposer"]
