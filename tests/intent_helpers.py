"""Decomposition fixtures over the TEST FIXTURE corpus (fictional; hashing embedder, no models needed)."""

import pytest

from streamrag.intents import make_decomposer
from streamrag.intents.query_builder import IntentQueryBuilder
from streamrag.intents.tracker import IntentTracker


@pytest.fixture(scope="session")
def dec(fixture_bundle):
    cfg, b = fixture_bundle
    return make_decomposer(cfg, b)


@pytest.fixture()
def tracker(fixture_bundle, dec):
    cfg, _ = fixture_bundle
    return IntentTracker("s", dec, cfg.multi_intent)


@pytest.fixture(scope="session")
def qb(fixture_bundle):
    _, b = fixture_bundle
    return IntentQueryBuilder(lambda t: list(dict.fromkeys(b.analyzer.tokens(t))))


