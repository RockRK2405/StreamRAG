"""Anti-cheating guard (spec REQ-REPRO-006): no evaluation strings, gold answers, benchmark-specific branches,
or vocabulary from the official guide's example scenarios may appear in the application code or configs."""

import json
import re

from conftest import FIX, REPO

SCANNED = [p for p in (REPO / "src").rglob("*.py")] + [p for p in (REPO / "configs").rglob("*.yaml")] + \
          [p for p in (REPO / "src").rglob("*.txt")]

# Distinctive words from the Theme 4 guide's illustrative examples. Retrieval code must be domain-neutral.
GUIDE_EXAMPLE_TERMS = ["pune", "catering", "reimbursement", "workshop venue", "doc_12", "doc_31", "doc_09",
                       "senior director", "foreign currency"]
SUSPICIOUS = [re.compile(p) for p in (r"if\s+query\s*==", r"query\s+in\s*\{", r"expected_answer", r"gold_evidence\s*=",
                                       r"answers?\s*=\s*\{\s*['\"]")]


def _eval_strings(min_words: int = 4):
    """Full evaluation utterances (>= 4 words). Shorter strings ("okay", "thanks") are generic vocabulary that a
    lexicon legitimately contains; whole multi-word evaluation utterances must never appear in source."""
    out = set()
    files = list(FIX.rglob("*.jsonl"))
    if (REPO / "eval").exists():
        files += list((REPO / "eval").rglob("*.jsonl"))
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                out.add(json.loads(line)["query"].lower())
    for f in (REPO / "eval").rglob("*.json") if (REPO / "eval").exists() else []:
        case = json.loads(f.read_text())
        for turn in case.get("session", {}).get("turns", []):
            out.add(turn["utterance_text"].lower())
        for turn in case.get("turns", []):                     # Phase 6 adaptive-session / Phase 7 grounded cases
            out.add(turn["utterance_text"].lower())
            for fact in turn.get("gold", {}).get("required_facts", []):   # Phase 7 gold answers
                out.add(fact["text"].lower())
        for utt in case.get("utterances", []):                 # Phase 5 multi-intent cases
            out.add(utt["utterance_text"].lower())
            for gi in utt.get("expected_intents", []):
                out.add(gi["description"].lower())
                out.update(p.lower() for p in gi.get("paraphrases", []))
    return {s for s in out if len(s.split()) >= min_words}


def test_eval_queries_not_in_source():
    queries = _eval_strings()
    assert queries
    for p in SCANNED:
        text = p.read_text().lower()
        for q in queries:
            assert q not in text, f"evaluation query embedded in {p}"


def test_guide_example_vocabulary_not_in_source():
    for p in SCANNED:
        text = p.read_text().lower()
        for term in GUIDE_EXAMPLE_TERMS:
            assert term not in text, f"guide example term '{term}' found in {p}"


def test_no_benchmark_specific_branches():
    for p in (REPO / "src").rglob("*.py"):
        text = p.read_text()
        for rx in SUSPICIOUS:
            assert not rx.search(text), f"suspicious pattern {rx.pattern} in {p}"
