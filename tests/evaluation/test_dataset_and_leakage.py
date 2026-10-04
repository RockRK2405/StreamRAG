"""Dataset integrity (brief §4-6) and leakage checks (brief §52-53)."""

import ast
import re
from pathlib import Path

from conftest import FIX, REPO

from streamrag.evaluation.dataset import QUERY_TYPES, difficulty, load

DS = REPO / "experiments" / "datasets" / "streamrag_eval_v1"


def test_dataset_schema_categories_and_difficulty_rule():
    test = load(DS / "test.jsonl")
    dev = load(DS / "dev.jsonl")
    assert {s.query_type for s in test} == set(QUERY_TYPES)            # all 14 categories in the test split
    assert all(s.split == "test" and s.corpus == "transit" for s in test)
    assert all(s.split == "dev" and s.corpus != "transit" for s in dev)
    for s in test + dev:                                                # difficulty is the rule, not a free label
        gap = bool(s.difficulty_features.get("lexical_gap"))
        d, f = difficulty(s, gap)
        assert d == s.difficulty and f == s.difficulty_features
    ids = [s.sample_id for s in test + dev]
    assert len(ids) == len(set(ids))
    for s in test:
        if s.expected_state == "SUFFICIENT" and s.query_type != "AMBIGUOUS":
            assert s.ground_truth_evidence and s.expected_claims and all(c.key for c in s.expected_claims)


def test_test_strings_not_in_code_configs_or_prompts():
    texts = []
    for s in load(DS / "test.jsonl"):
        texts += [s.query, s.expected_answer] + [c.text for c in s.expected_claims]
    texts = {t.lower() for t in texts if len(t.split()) >= 4}
    files = [p for p in (REPO / "src").rglob("*.py")] + [p for p in (REPO / "configs").rglob("*.yaml")]
    for p in files:
        body = p.read_text().lower()
        for t in texts:
            assert t not in body, f"test string in {p}: {t}"


def _shingles(text: str, n: int = 5) -> set:
    """5-word shingles of the body text (front matter, headings and the fixture marker line excluded)."""
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    lines = [l for l in text.splitlines() if "TEST FIXTURE ONLY" not in l and not l.startswith("#")]
    w = re.findall(r"[a-z0-9]+", " ".join(lines).lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def test_document_level_separation_of_the_test_corpus():
    test_docs = {p.name: _shingles(p.read_text()) for p in (FIX / "corpus_eval_transit").glob("*.md")}
    dev_dirs = ["corpus", "corpus_adaptive", "corpus_grounding", "corpus_conflict", "corpus_injection"]
    dev_docs = {f"{d}/{p.name}": _shingles(p.read_text()) for d in dev_dirs for p in (FIX / d).glob("*")
                if p.suffix in (".md", ".txt")}
    worst = 0.0
    for a, sa in test_docs.items():
        for b, sb in dev_docs.items():
            if sa and sb:
                worst = max(worst, len(sa & sb) / len(sa | sb))
    # near-duplicate = >= 0.3 shared 5-word shingles; the highest observed is 0.125 (two holiday notices that list
    # the same three dates) - documented in docs/evaluation/dataset.md
    assert worst < 0.3, f"near-duplicate documents across splits (5-gram Jaccard {worst:.2f})"
    test_ids = {re.search(r"^id: (.+)$", p.read_text(), re.M).group(1) for p in (FIX / "corpus_eval_transit").glob("*.md")}
    dev_ids = {m.group(1) for d in dev_dirs for p in (FIX / d).glob("*.md")
               if (m := re.search(r"^id: (.+)$", p.read_text(), re.M))}
    assert not test_ids & dev_ids


def test_no_system_module_reads_evaluation_data():
    pkg = REPO / "src" / "streamrag"
    for f in pkg.rglob("*.py"):
        if "evaluation" in f.parts:
            continue
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            assert not any(m.startswith("streamrag.evaluation") for m in mods), f"{f} imports the evaluation package"
        body = f.read_text()
        assert "streamrag_eval" not in body and "experiments/datasets" not in body, f


# ---------------------------------------------------------------- Phase 11 held-out set v2
DS2 = REPO / "experiments" / "datasets" / "streamrag_eval_v2"
DEMO = REPO / "demo"


def test_v2_schema_categories_and_difficulty_rule():
    v2 = load(DS2 / "test.jsonl")
    assert {s.query_type for s in v2} == set(QUERY_TYPES)
    assert all(s.split == "test" and s.corpus == "utility" for s in v2)
    for s in v2:
        d, f = difficulty(s, bool(s.difficulty_features.get("lexical_gap")))
        assert d == s.difficulty and f == s.difficulty_features
        if s.expected_state == "SUFFICIENT" and s.query_type != "AMBIGUOUS":
            assert s.ground_truth_evidence and s.expected_claims and all(c.key for c in s.expected_claims)
    assert len({s.sample_id for s in v2}) == len(v2)


def test_v2_strings_not_in_code_configs_or_demo():
    texts = set()
    for s in load(DS2 / "test.jsonl"):
        texts |= {t.lower() for t in [s.query, s.expected_answer] + [c.text for c in s.expected_claims]
                  if len(t.split()) >= 4}
    files = [p for p in (REPO / "src").rglob("*") if p.suffix in (".py", ".html", ".js", ".css", ".txt")]
    files += [p for p in (REPO / "configs").rglob("*.yaml")]
    files += [p for p in DEMO.rglob("*") if p.is_file()] if DEMO.exists() else []
    for p in files:
        body = p.read_text(errors="ignore").lower()
        for t in texts:
            assert t not in body, f"held-out v2 string in {p}: {t}"


def test_v2_document_level_separation():
    v2_docs = {p.name: _shingles(p.read_text()) for p in (FIX / "corpus_eval_utility").glob("*.md")}
    other_dirs = [FIX / d for d in ("corpus", "corpus_adaptive", "corpus_grounding", "corpus_conflict",
                                    "corpus_injection", "corpus_eval_transit")]
    if (DEMO / "corpus").exists():
        other_dirs.append(DEMO / "corpus")
    others = {f"{d.name}/{p.name}": _shingles(p.read_text()) for d in other_dirs for p in d.glob("*")
              if p.suffix in (".md", ".txt")}
    worst = max((len(a & b) / len(a | b) for a in v2_docs.values() for b in others.values() if a and b), default=0)
    assert worst < 0.3, f"near-duplicate documents (5-gram Jaccard {worst:.2f})"
    ids = lambda d: {m.group(1) for p in d.glob("*.md") if (m := re.search(r"^id: (.+)$", p.read_text(), re.M))}  # noqa: E731
    assert not ids(FIX / "corpus_eval_utility") & set().union(*(ids(d) for d in other_dirs))
