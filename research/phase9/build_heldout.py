"""Writes eval/heldout_adaptive_retrieval/ - written AFTER the Phase 9 code freeze, run once, no fixes afterwards.

Same corpus as the dev set (tests/fixtures/corpus_adaptive, which the implementer has seen), new questions and
sessions, labels by the implementer from the corpus text. TEST FIXTURE ONLY - NOT REPORTABLE.
Usage: .venv/bin/python research/phase9/build_heldout.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "eval" / "heldout_adaptive_retrieval"
A = "adaptive"
SINGLE = [
    ("H01", "semantic", "Do refugees have to pay the residence permit fee?", ["WAIVERS §1"], "SUFFICIENT"),
    ("H02", "simple", "How does an applicant request a fee waiver?", ["WAIVERS §2"], "SUFFICIENT"),
    ("H03", "simple", "What does the Civic Registry issue?", ["ADDRESS §2"], "SUFFICIENT"),
    ("H04", "simple", "Is address registration free of charge?", ["ADDRESS §1"], "SUFFICIENT"),
    ("H05", "simple", "What must vehicle owners present to register a vehicle?", ["VEH-2026 §2"], "SUFFICIENT"),
    ("H06", "simple", "How often must vehicles older than four years be inspected?", ["VEH-2026 §3"], "SUFFICIENT"),
    ("H07", "constraint", "Which form do domestic applicants use?", ["FORMS §1", "DOM-2026 §1"], "SUFFICIENT"),
    ("H08", "simple", "Where are business licence applications submitted?", ["BIZ-2026 §3"], "SUFFICIENT"),
    ("H09", "simple", "Who needs a business licence in Fixture City?", ["BIZ-2026 §1"], "SUFFICIENT"),
    ("H10", "temporal", "What was the fee for a residence permit in 2025?", ["ELIG-2024 §2"], "SUFFICIENT"),
    ("H11", "temporal", "Which requirement was added in the 2026 edition?", ["ELIG-2026 §3"], "SUFFICIENT"),
    ("H12", "multi_hop", "What extra documents do applicants from Estria provide?", ["ANNEX-C §1", "GROUP-RULES §2"],
     "SUFFICIENT"),
    ("H13", "multi_hop", "Do Norvia applicants attend an interview?", ["ANNEX-C §1", "GROUP-RULES §1"], "SUFFICIENT"),
    ("H14", "contradiction", "What time does the Permit Processing Office open?", ["NOTICE-A §1", "NOTICE-B §1"],
     "CONTRADICTORY"),
    ("H15", "exact_keyword", "Is the PPO open on 25 December?", ["FORMS §2", "HOLIDAYS §1"], "SUFFICIENT"),
    ("H16", "insufficient", "Can I appeal a decision about a business licence?", [], "INSUFFICIENT"),
    ("H17", "insufficient", "How much is the fine for late vehicle registration?", [], "INSUFFICIENT"),
    ("H18", "semantic", "How many days do international applicants wait for a decision?", ["INTL-2026 §2"],
     "SUFFICIENT"),
    ("H19", "constraint", "What is the reduced fee for applicants younger than 25?", ["ELIG-2026 §2"], "SUFFICIENT"),
    ("H20", "exact_keyword", "What must international applicants submit together with Form PX-204?", ["INTL-2026 §1"],
     "SUFFICIENT"),
]
SESSIONS = [
    ("HS1", "follow_up", [("How much is the vehicle registration fee?", ["VEH-2026 §1"]),
                          ("And how often are old vehicles inspected?", ["VEH-2026 §3"])]),
    ("HS2", "repeated_question", [("What is the processing time for international applicants?", ["INTL-2026 §2"]),
                                  ("What is the processing time for international applicants?", ["INTL-2026 §2"])]),
]
META = {"is_fixture": True, "reportable": False, "labeler": "implementer (Phase 9) - written after code freeze",
        "run_policy": "run once after freeze; no fixes based on it"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cid, cat, text, gold, expect in SINGLE:
        (OUT / f"{cid}.json").write_text(json.dumps({"case_id": cid, "corpus": A, "category": cat, **META, "turns": [
            {"utterance_id": "u1", "utterance_text": text, "gold": {"citations": gold, "expect": expect}}]},
            indent=2, ensure_ascii=False) + "\n")
    for cid, cat, turns in SESSIONS:
        (OUT / f"{cid}.json").write_text(json.dumps({"case_id": cid, "corpus": A, "category": cat, **META, "turns": [
            {"utterance_id": f"u{n}", "utterance_text": t, "gold": {"citations": g, "expect": "SUFFICIENT"}}
            for n, (t, g) in enumerate(turns, start=1)]}, indent=2, ensure_ascii=False) + "\n")
    (OUT / "README.md").write_text(
        "# eval/heldout_adaptive_retrieval (TEST FIXTURE ONLY - NOT REPORTABLE)\n\nWritten by "
        "`research/phase9/build_heldout.py` after the Phase 9 code freeze; run once by "
        "`research/phase9/run_benchmarks.py --eval heldout_adaptive_retrieval`; no fixes were made based on it. Same "
        "fixture corpus as the dev set (seen by the implementer); implementer labels.\n")
    print(len(SINGLE), "+", len(SESSIONS), "->", OUT)


if __name__ == "__main__":
    main()
