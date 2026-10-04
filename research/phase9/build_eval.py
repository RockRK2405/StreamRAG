"""Writes eval/dev_adaptive_retrieval/ (Phase 9 dev cases; TEST FIXTURE ONLY - NOT REPORTABLE).

Labels were written by the implementer (the same person who wrote the adaptive retrieval code) by reading the
fictional fixture corpora: gold = the corpus sections that state what the question asks (section-level citations),
``expect`` = the evidence state the corpus supports (SUFFICIENT / INSUFFICIENT / CONTRADICTORY). There is no
independent annotation, no held-out split and the corpora are tiny (8-60 chunks): results on this set are a
behaviour check, not evidence of real-corpus quality. Categories follow brief §58.

Usage: .venv/bin/python research/phase9/build_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "eval" / "dev_adaptive_retrieval"
A = "adaptive"
SINGLE = [
    # id, corpus, category, utterance, gold citations, expected evidence state
    ("A01", A, "simple", "What is the application fee for a residence permit?", ["ELIG-2026 §2"], "SUFFICIENT"),
    ("A02", A, "simple", "How old must applicants be?", ["ELIG-2026 §1"], "SUFFICIENT"),
    ("A03", A, "simple", "How often must a residence permit be renewed?", ["RENEW §1"], "SUFFICIENT"),
    ("A04", A, "simple", "How long does an applicant have to appeal a refused application?", ["APPEALS §1"],
     "SUFFICIENT"),
    ("A05", A, "simple", "Who hears the appeals?", ["APPEALS §2"], "SUFFICIENT"),
    ("A06", A, "simple", "How much is the business licence fee?", ["BIZ-2026 §2"], "SUFFICIENT"),
    ("A07", A, "simple", "How soon must a vehicle be registered after purchase?", ["VEH-2026 §1"], "SUFFICIENT"),
    ("A08", A, "simple", "Within how many days must new residents register their address?", ["ADDRESS §1"],
     "SUFFICIENT"),
    ("A09", A, "exact_keyword", "What is Form PX-204 used for?", ["FORMS §1", "INTL-2026 §1"], "SUFFICIENT"),
    ("A10", A, "exact_keyword", "Where do I submit Form BL-310?", ["BIZ-2026 §3"], "SUFFICIENT"),
    ("A11", A, "exact_keyword", "Who has to present Form VR-12?", ["VEH-2026 §2"], "SUFFICIENT"),
    ("A12", A, "exact_keyword", "When is the PPO closed?", ["FORMS §2", "HOLIDAYS §1"], "SUFFICIENT"),
    ("A13", A, "exact_keyword", "Which form is used to renew a residence permit?", ["FORMS §1", "RENEW §1"],
     "SUFFICIENT"),
    ("A14", A, "semantic", "How much do I have to pay to apply for residency?", ["ELIG-2026 §2"], "SUFFICIENT"),
    ("A15", A, "semantic", "Is there any way to avoid paying the permit charge?", ["WAIVERS §1"], "SUFFICIENT"),
    ("A16", A, "semantic", "Can I bring somebody along to the review hearing?", ["APPEALS §2"], "SUFFICIENT"),
    ("A17", A, "semantic", "Do old cars need to be checked regularly?", ["VEH-2026 §3"], "SUFFICIENT"),
    ("A18", A, "constraint", "What are the eligibility requirements for international applicants?",
     ["ELIG-2026 §1", "INTL-2026 §1"], "SUFFICIENT"),
    ("A19", A, "constraint", "How long does processing take for domestic applicants?", ["DOM-2026 §2"],
     "SUFFICIENT"),
    ("A20", A, "constraint", "How long does processing take for international applicants?", ["INTL-2026 §2"],
     "SUFFICIENT"),
    ("A21", A, "constraint", "Which documents do domestic applicants submit?", ["DOM-2026 §1"], "SUFFICIENT"),
    ("A22", A, "constraint", "What is the reduced business licence fee for market stalls?", ["BIZ-2026 §2"],
     "SUFFICIENT"),
    ("A23", A, "temporal", "What was the application fee in December 2025?", ["ELIG-2024 §2"], "SUFFICIENT"),
    ("A24", A, "temporal", "What changed in the 2026 edition of the eligibility rules?", ["ELIG-2026 §3"],
     "SUFFICIENT"),
    ("A25", A, "temporal", "What is the current application fee for a residence permit?", ["ELIG-2026 §2"],
     "SUFFICIENT"),
    ("A26", A, "temporal", "What was the previous application fee for a residence permit?",
     ["ELIG-2024 §2", "ELIG-2026 §3"], "SUFFICIENT"),
    ("A27", A, "temporal", "Which public holidays are there in 2026?", ["HOLIDAYS §1"], "SUFFICIENT"),
    ("A28", A, "multi_hop", "What documents does an applicant from Zemland need?", ["ANNEX-C §1", "GROUP-RULES §2"],
     "SUFFICIENT"),
    ("A29", A, "multi_hop", "Do applicants from Norvia need additional documents?", ["ANNEX-C §1", "GROUP-RULES §1"],
     "SUFFICIENT"),
    ("A30", A, "multi_hop", "Does an applicant from Estria have to attend an interview?",
     ["ANNEX-C §1", "GROUP-RULES §2"], "SUFFICIENT"),
    ("A31", A, "multi_intent", "What is the application fee and how long does processing take for domestic applicants?",
     ["ELIG-2026 §2", "DOM-2026 §2"], "SUFFICIENT"),
    ("A32", A, "multi_intent", "What are the eligibility requirements and which form do international applicants use?",
     ["ELIG-2026 §1", "INTL-2026 §1", "FORMS §1"], "SUFFICIENT"),
    ("A33", A, "multi_intent", "How much is the vehicle registration fee and what documents are needed for registration?",
     ["VEH-2026 §1", "VEH-2026 §2"], "SUFFICIENT"),
    ("A34", A, "contradiction", "When does the Permit Processing Office open?", ["NOTICE-A §1", "NOTICE-B §1"],
     "CONTRADICTORY"),
    ("A35", A, "contradiction", "What time does the PPO open on weekdays?", ["NOTICE-A §1", "NOTICE-B §1"],
     "CONTRADICTORY"),
    ("A36", A, "insufficient", "Is there parking at the Permit Processing Office?", [], "INSUFFICIENT"),
    ("A37", A, "insufficient", "How much does it cost to renew a residence permit?", [], "INSUFFICIENT"),
    ("A38", A, "insufficient", "What is the fee for a vehicle safety inspection?", [], "INSUFFICIENT"),
    ("A39", A, "insufficient", "Can applicants pay the fee by credit card?", [], "INSUFFICIENT"),
    ("A40", A, "ambiguous", "What about the fee?", ["ELIG-2026 §2", "BIZ-2026 §2", "VEH-2026 §1"], "SUFFICIENT"),
    ("A41", A, "ambiguous", "How long does it take?", ["DOM-2026 §2", "INTL-2026 §2", "BIZ-2026 §3"], "SUFFICIENT"),
    ("B01", "fixture", "simple", "How high should the wicks be trimmed?", ["fixture_lighthouse_manual §1.1"],
     "SUFFICIENT"),
    ("B02", "fixture", "simple", "How often does the fog signal sound during a storm?", ["fixture_lighthouse_manual §3"],
     "SUFFICIENT"),
    ("B03", "fixture", "simple", "What is the maximum number of crates a picker may fill per shift?", ["Doc_07 §3"],
     "SUFFICIENT"),
    ("B04", "fixture", "constraint", "Are ladders allowed in the orchard overnight?", ["Doc_07 §2.2"], "SUFFICIENT"),
    ("B05", "fixture", "simple", "When is the main telescope recalibrated?", ["fixture_observatory_notes §1"],
     "SUFFICIENT"),
    ("B06", "fixture", "simple", "What must each worker do before leaving the tool shed?", ["Doc_07 §4"],
     "SUFFICIENT"),
    ("B07", "fixture", "semantic", "What should the keeper use to clean the lens?", ["fixture_lighthouse_manual §1.2"],
     "SUFFICIENT"),
    ("B08", "grounding", "contradiction", "What is the application fee for a new permit?",
     ["fixture_permit_notice_2023 §1", "fixture_permit_notice_2025 §1"], "CONTRADICTORY"),
    ("B09", "grounding", "simple", "How often must permits be renewed?", ["fixture_permit_handbook §3"], "SUFFICIENT"),
    ("B10", "grounding", "simple", "How are permit applications submitted?", ["fixture_permit_handbook §2"],
     "SUFFICIENT"),
    ("B11", "conflict", "contradiction", "Below which wind speed does the observatory open its dome?",
     ["fixture_dome_notes_new §1", "fixture_dome_notes_old §1"], "CONTRADICTORY"),
    ("B12", "injection", "simple", "Is the permit office open on public holidays?", ["fixture_permit_bulletin §1"],
     "SUFFICIENT"),
    ("B13", "injection", "simple", "How often do permits have to be renewed?", ["fixture_permit_handbook §3"],
     "SUFFICIENT"),
]
SESSIONS = [
    ("S01", "contextual_follow_up", [("What are the eligibility requirements for a residence permit?", ["ELIG-2026 §1"]),
                                     ("What about international applicants?", ["ELIG-2026 §1", "INTL-2026 §1"])]),
    ("S02", "repeated_question", [("What is the application fee for a residence permit?", ["ELIG-2026 §2"]),
                                  ("How long does processing take for domestic applicants?", ["DOM-2026 §2"]),
                                  ("What is the application fee for a residence permit?", ["ELIG-2026 §2"])]),
    ("S03", "entity_correction", [("Do applicants from Zemland need an interview?", ["ANNEX-C §1", "GROUP-RULES §2"]),
                                  ("Sorry, I meant Norvia, not Zemland.", ["ANNEX-C §1", "GROUP-RULES §1"])]),
    ("S04", "constraint_change", [("How long does processing take for international applicants?", ["INTL-2026 §2"]),
                                  ("Actually, for domestic applicants instead.", ["DOM-2026 §2"])]),
    ("S05", "late_constraint", [("How long does processing take for a residence permit?",
                                 ["DOM-2026 §2", "INTL-2026 §2"]),
                                ("For international applicants.", ["INTL-2026 §2"])]),
    ("S06", "equivalent_repeat", [("What is the residence permit application fee?", ["ELIG-2026 §2"]),
                                  ("Which public holidays are there in 2026?", ["HOLIDAYS §1"]),
                                  ("What is the application fee for a residence permit?", ["ELIG-2026 §2"])]),
    ("S07", "cross_query_reuse", [("What is the application fee for a residence permit?", ["ELIG-2026 §2"]),
                                  ("Is there a reduced application fee for applicants under 25?", ["ELIG-2026 §2"])]),
    ("S08", "follow_up", [("How much is the business licence fee?", ["BIZ-2026 §2"]),
                          ("And for market stalls?", ["BIZ-2026 §2"])]),
]
META = {"is_fixture": True, "reportable": False, "labeler": "implementer (Phase 9) - not independent"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    for cid, corpus, cat, text, gold, expect in SINGLE:
        case = {"case_id": cid, "corpus": corpus, "category": cat, **META,
                "turns": [{"utterance_id": "u1", "utterance_text": text,
                           "gold": {"citations": gold, "expect": expect}}]}
        (OUT / f"{cid}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    for cid, cat, turns in SESSIONS:
        case = {"case_id": cid, "corpus": A, "category": cat, **META,
                "turns": [{"utterance_id": f"u{n}", "utterance_text": t, "gold": {"citations": g, "expect": "SUFFICIENT"}}
                          for n, (t, g) in enumerate(turns, start=1)]}
        (OUT / f"{cid}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    (OUT / "README.md").write_text(
        "# eval/dev_adaptive_retrieval (TEST FIXTURE ONLY - NOT REPORTABLE)\n\n"
        "Phase 9 dev cases written by `research/phase9/build_eval.py`. Labels were written by the implementer of the "
        "adaptive retrieval code by reading the fictional fixture corpora (no independent annotation, no held-out "
        "split, tiny corpora). Gold = section-level citations that state what is asked; `expect` = the evidence "
        "state the corpus supports. A-cases and S-sessions use `tests/fixtures/corpus_adaptive`; B-cases reuse the "
        "Phase 3/7 fixture corpora (gold taken from `tests/fixtures/eval/fixture_retrieval.jsonl` and "
        "`eval/dev_grounded` where they overlap).\n")
    print(len(SINGLE), "single-turn cases,", len(SESSIONS), "sessions ->", OUT)


if __name__ == "__main__":
    main()
