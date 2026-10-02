"""Retrieval controller (Phase 4): WAIT / RETRIEVE / SKIP over a streaming transcript."""

from __future__ import annotations

from streamrag.config.settings import StreamRagConfig
from streamrag.controller.acts import PrototypeActClassifier, RuleActClassifier
from streamrag.controller.controller import EndOnlyPolicy, EveryChunkPolicy, RetrievalController
from streamrag.controller.lexicon import Lexicon
from streamrag.controller.models import ControllerInput, RetrievalDecision
from streamrag.controller.query_builder import BuiltQuery, QueryBuilder
from streamrag.controller.signals import SignalComputer
from streamrag.retrieval.store import IndexBundle
from streamrag.retrieval.text import load_stopwords


def make_policy(cfg: StreamRagConfig, bundle: IndexBundle, embed=None):
    """Build the configured policy over the loaded index (anchors come from the index's own vocabulary)."""
    lex = Lexicon.load(cfg.controller.lexicon)
    builder = QueryBuilder(lex)
    signals = SignalComputer(cfg.controller, lex, bundle.analyzer, bundle.bm25)
    strategy = cfg.controller.strategy
    if strategy == "end_only":
        return EndOnlyPolicy(builder, signals)
    if strategy == "every_chunk":
        return EveryChunkPolicy(builder, signals)
    rules = RuleActClassifier(lex, load_stopwords(cfg.lexical.stopwords), signals.is_anchor_word)
    classifier = rules
    if cfg.controller.act_classifier == "prototype":
        if embed is None:
            raise ValueError("prototype act classifier needs an embedding function")
        classifier = PrototypeActClassifier(lex, embed, rules)
    return RetrievalController(cfg.controller, classifier, builder, signals)


__all__ = ["BuiltQuery", "ControllerInput", "EndOnlyPolicy", "EveryChunkPolicy", "Lexicon", "PrototypeActClassifier",
           "QueryBuilder", "RetrievalController", "RetrievalDecision", "RuleActClassifier", "SignalComputer",
           "make_policy"]
