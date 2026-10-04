"""Builds experiments/datasets/streamrag_eval_v1/{test,dev}.jsonl + manifest.json (Phase 10).

TEST split: new questions over a NEW fictional corpus (tests/fixtures/corpus_eval_transit, Northvale transit), written
for Phase 10 before any system was run on it; no component was developed or tuned on this corpus (document-level
separation from every development corpus, checked by tests/evaluation/test_leakage.py). Run once per experiment.

DEV split: the earlier development sets converted to the same schema - Phase 9 dev + held-out
(eval/dev_adaptive_retrieval, eval/heldout_adaptive_retrieval; corpus_adaptive) and Phase 7 grounded cases
(eval/dev_grounded; fixture / grounding / conflict / injection corpora). They were used while building the system,
so dev numbers are optimistic.

All labels are implementer-written from the corpus text (no independent annotation). TEST FIXTURE ONLY.
Usage: .venv/bin/python experiments/datasets/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from streamrag.evaluation.dataset import EvalSample, ExpectedClaim, StreamChunk, difficulty, dumps, manifest  # noqa: E402

OUT = REPO / "experiments" / "datasets" / "streamrag_eval_v1"
T = "transit"
C = ExpectedClaim


def S(sid, qtype, query, gold, claims, answer, intents, entities=(), constraints=(), state="SUFFICIENT",
      conflict=(), forbidden=(), semantics="all", stream=(), context=(), turn=1, session=None):
    return dict(sample_id=sid, split="test", source="phase10_test", corpus=T, session_id=session or sid,
                turn_index=turn, query=query, stream=[StreamChunk(text=c) if isinstance(c, str) else
                                                      StreamChunk(text=c[1], replaces=c[0]) for c in stream],
                conversation_context=list(context), query_type=qtype, expected_intents=list(intents),
                required_entities=list(entities), required_constraints=list(constraints),
                ground_truth_evidence=list(gold), gold_semantics=semantics, expected_claims=list(claims),
                expected_answer=answer, expected_state=state, conflict_values=list(conflict),
                forbidden=list(forbidden), source_documents=sorted({g.split(" §")[0] for g in gold}))


TEST = [
    # ---------------------------------------------------------------- SIMPLE
    S("T01", "SIMPLE", "How long is a single ride ticket valid?", ["FARES-2026 §2"],
      [C(text="A single ride ticket is valid for 120 minutes after validation.", citation="FARES-2026 §2",
         key=["120 minutes"])], "A single ride ticket is valid for 120 minutes after validation.",
      ["validity of a single ride ticket"], ["single ride ticket"]),
    S("T02", "SIMPLE", "How long are found items kept?", ["LOST §1"],
      [C(text="Items found on vehicles are kept for 30 days.", citation="LOST §1", key=["30 days"])],
      "Found items are kept for 30 days at the central depot.", ["how long found items are kept"], ["found items"]),
    S("T03", "SIMPLE", "How many wheelchair spaces does each bus have?", ["ACCESS §1"],
      [C(text="Every bus has two wheelchair spaces.", citation="ACCESS §1", key=["two"])],
      "Every bus has two wheelchair spaces near the middle door.", ["number of wheelchair spaces per bus"],
      ["wheelchair spaces"]),
    S("T04", "SIMPLE", "How much does a replacement travel card cost?", ["TCO §1"],
      [C(text="A replacement card costs 5 euros.", citation="TCO §1", key=["5 euros"])],
      "A replacement travel card costs 5 euros.", ["price of a replacement card"], ["replacement travel card"]),
    S("T05", "SIMPLE", "Within how many days are refunds paid?", ["REFUNDS §2"],
      [C(text="Refunds are paid within 21 days.", citation="REFUNDS §2", key=["21 days"])],
      "Refunds are paid within 21 days.", ["refund payment time"], ["refunds"]),
    # ---------------------------------------------------------------- SEMANTIC (paraphrase, little word overlap)
    S("T06", "SEMANTIC", "If I forget my wallet on the bus, how long will you hold on to it?", ["LOST §1"],
      [C(text="Items found on vehicles are kept for 30 days.", citation="LOST §1", key=["30 days"])],
      "Items found on vehicles are kept for 30 days.", ["how long lost items are held"], ["lost wallet"]),
    S("T07", "SEMANTIC", "Can I bring my bike on the train during the morning rush?", ["BIKES §1"],
      [C(text="Bicycles are allowed on trains outside peak hours; peak hours are 7:00 to 9:00 on weekdays.",
         citation="BIKES §1", key=["7:00", "9:00"])],
      "No - bicycles are only allowed outside peak hours (7:00 to 9:00 and 16:00 to 18:00 on weekdays).",
      ["bicycles on trains at peak time"], ["bicycle", "train"]),
    S("T08", "SEMANTIC", "What happens if I get caught riding without paying?", ["PENALTIES §1"],
      [C(text="A rider without a valid ticket pays a penalty fare of 60 euros.", citation="PENALTIES §1",
         key=["60 euros"])], "A rider without a valid ticket pays a penalty fare of 60 euros.",
      ["consequence of riding without a ticket"], ["fare evasion"]),
    S("T09", "SEMANTIC", "Do older people have to pay to ride?", ["CONC-SENIOR §1"],
      [C(text="Riders aged 65 or older travel free outside peak hours.", citation="CONC-SENIOR §1",
         key=["65", "free"])], "Riders aged 65 or older travel free outside peak hours.",
      ["fares for older riders"], ["older riders"]),
    S("T10", "SEMANTIC", "How far ahead do I need to arrange help at the station?", ["ACCESS §2"],
      [C(text="Travel assistance must be booked at least 24 hours in advance.", citation="ACCESS §2",
         key=["24 hours"])], "Travel assistance must be booked at least 24 hours in advance.",
      ["advance notice for station assistance"], ["station assistance"]),
    # ---------------------------------------------------------------- EXACT_TERM
    S("T11", "EXACT_TERM", "What is Form TR-410 for?", ["REFUNDS §2"],
      [C(text="Refunds are requested with Form TR-410.", citation="REFUNDS §2", key=["refund"])],
      "Form TR-410 is used to request refunds.", ["purpose of Form TR-410"], ["Form TR-410"]),
    S("T12", "EXACT_TERM", "Who uses Form TR-305?", ["CONC-SENIOR §2"],
      [C(text="Seniors apply with Form TR-305 and proof of age.", citation="CONC-SENIOR §2", key=["senior"])],
      "Seniors use Form TR-305 (with proof of age) to apply for the senior concession.",
      ["users of Form TR-305"], ["Form TR-305"]),
    S("T13", "EXACT_TERM", "When is the TCO closed?", ["TCO §2"],
      [C(text="The TCO is closed on Sundays.", citation="TCO §2", key=["Sunday"])],
      "The Travel Card Office is closed on Sundays.", ["closing day of the TCO"], ["TCO"]),
    S("T14", "EXACT_TERM", "What has to be submitted together with Form TR-210?", ["CONC-STUDENT §2"],
      [C(text="Students apply with Form TR-210 and a current enrolment certificate.", citation="CONC-STUDENT §2",
         key=["enrolment certificate"])], "Form TR-210 is submitted with a current enrolment certificate.",
      ["documents to submit with Form TR-210"], ["Form TR-210"]),
    S("T15", "EXACT_TERM", "What does the TCO issue?", ["TCO §1"],
      [C(text="The TCO issues all concession cards.", citation="TCO §1", key=["concession cards"])],
      "The Travel Card Office issues all concession cards and replaces lost travel cards.",
      ["services of the TCO"], ["TCO"]),
    # ---------------------------------------------------------------- MULTI_INTENT
    S("T16", "MULTI_INTENT", "How much is a day pass and how long is a single ticket valid?",
      ["FARES-2026 §1", "FARES-2026 §2"],
      [C(text="A day pass costs 8.00 euros.", citation="FARES-2026 §1", key=["8.00"]),
       C(text="A single ride ticket is valid for 120 minutes.", citation="FARES-2026 §2", key=["120 minutes"])],
      "A day pass costs 8.00 euros, and a single ride ticket is valid for 120 minutes.",
      ["price of a day pass", "validity of a single ticket"], ["day pass", "single ticket"]),
    S("T17", "MULTI_INTENT", "What is the penalty fare and how can I reduce it?", ["PENALTIES §1", "PENALTIES §2"],
      [C(text="The penalty fare is 60 euros.", citation="PENALTIES §1", key=["60 euros"]),
       C(text="It is reduced to 30 euros if paid within 7 days.", citation="PENALTIES §2", key=["30 euros"])],
      "The penalty fare is 60 euros; it is reduced to 30 euros if paid within 7 days.",
      ["amount of the penalty fare", "how to reduce the penalty"], ["penalty fare"]),
    S("T18", "MULTI_INTENT", "How long are lost items kept and is there a fee to collect them?", ["LOST §1", "LOST §2"],
      [C(text="Lost items are kept for 30 days.", citation="LOST §1", key=["30 days"]),
       C(text="A collection fee of 3 euros applies.", citation="LOST §2", key=["3 euros"])],
      "Lost items are kept for 30 days, and a collection fee of 3 euros applies.",
      ["how long lost items are kept", "collection fee"], ["lost items"]),
    S("T19", "MULTI_INTENT", "Which form is needed for a refund and how quickly is the refund paid?", ["REFUNDS §2"],
      [C(text="Refunds are requested with Form TR-410.", citation="REFUNDS §2", key=["TR-410"]),
       C(text="Refunds are paid within 21 days.", citation="REFUNDS §2", key=["21 days"])],
      "Refunds are requested with Form TR-410 and paid within 21 days.", ["refund form", "refund payment time"],
      ["refund"]),
    S("T20", "MULTI_INTENT", "When are peak hours and are folding bikes allowed then?", ["BIKES §1", "BIKES §2"],
      [C(text="Peak hours are 7:00 to 9:00 and 16:00 to 18:00 on weekdays.", citation="BIKES §1",
         key=["7:00", "16:00"]),
       C(text="Folded bicycles are allowed at all times.", citation="BIKES §2", key=["all times"])],
      "Peak hours are 7:00-9:00 and 16:00-18:00 on weekdays; folded bicycles are allowed at all times.",
      ["peak hours", "folding bicycles at peak"], ["peak hours", "folding bike"]),
    # ---------------------------------------------------------------- MULTI_CONSTRAINT
    S("T21", "MULTI_CONSTRAINT", "How long does processing take for students?", ["CONC-STUDENT §2"],
      [C(text="Student applications are processed within 5 working days.", citation="CONC-STUDENT §2",
         key=["5 working days"])], "Student applications are processed within 5 working days.",
      ["processing time"], [], ["students"], forbidden=[r"\b10 working days"]),
    S("T22", "MULTI_CONSTRAINT", "What discount do students get on the monthly pass?", ["CONC-STUDENT §1"],
      [C(text="Students receive a 50 percent discount on the monthly pass.", citation="CONC-STUDENT §1",
         key=["50 percent"])], "Students receive a 50 percent discount on the monthly pass.",
      ["student discount"], ["monthly pass"], ["students", "monthly pass"]),
    S("T23", "MULTI_CONSTRAINT", "How long does processing take for seniors applying for the concession?",
      ["CONC-SENIOR §2"],
      [C(text="Senior applications are processed within 10 working days.", citation="CONC-SENIOR §2",
         key=["10 working days"])], "Senior applications are processed within 10 working days.",
      ["processing time"], ["concession"], ["seniors"], forbidden=[r"\b5 working days"]),
    S("T24", "MULTI_CONSTRAINT", "What does a monthly pass cost for a student?", ["FARES-2026 §1", "CONC-STUDENT §1"],
      [C(text="A monthly pass costs 72 euros.", citation="FARES-2026 §1", key=["72 euros"]),
       C(text="Students receive a 50 percent discount on the monthly pass.", citation="CONC-STUDENT §1",
         key=["50 percent"])],
      "A monthly pass costs 72 euros; students receive a 50 percent discount on it.",
      ["monthly pass price for students"], ["monthly pass"], ["students", "monthly pass"]),
    S("T25", "MULTI_CONSTRAINT", "Which documents do seniors need to apply, and how long until approval?",
      ["CONC-SENIOR §2"],
      [C(text="Seniors apply with Form TR-305 and proof of age.", citation="CONC-SENIOR §2",
         key=["TR-305", "proof of age"]),
       C(text="Senior applications are processed within 10 working days.", citation="CONC-SENIOR §2",
         key=["10 working days"])],
      "Seniors apply with Form TR-305 and proof of age; applications are processed within 10 working days.",
      ["documents for seniors", "processing time for seniors"], [], ["seniors"]),
    # ---------------------------------------------------------------- TEMPORAL
    S("T26", "TEMPORAL", "What did a single ride cost in 2025?", ["FARES-2025 §1"],
      [C(text="In 2025 a single ride cost 2.40 euros.", citation="FARES-2025 §1", key=["2.40"])],
      "In 2025 a single ride cost 2.40 euros.", ["single ride price in 2025"], ["single ride"],
      forbidden=[r"single ride (costs|cost|is) 2\.80"]),
    S("T27", "TEMPORAL", "What is the current price of a single ride?", ["FARES-2026 §1"],
      [C(text="A single ride costs 2.80 euros.", citation="FARES-2026 §1", key=["2.80"])],
      "A single ride currently costs 2.80 euros.", ["current single ride price"], ["single ride"],
      forbidden=[r"single ride (costs|cost|is) 2\.40"]),
    S("T28", "TEMPORAL", "What was the monthly pass price in January 2026?", ["FARES-2025 §1"],
      [C(text="Until 31 January 2026 a monthly pass cost 65 euros.", citation="FARES-2025 §1", key=["65 euros"])],
      "In January 2026 the 2025 fare table still applied: a monthly pass cost 65 euros (the 2026 table, published "
      "10 January, took effect on 1 February).", ["monthly pass price in January 2026"], ["monthly pass"],
      forbidden=[r"monthly pass (costs|cost|was|is) 72"]),
    S("T29", "TEMPORAL", "What changed in the 2026 fare table?", ["FARES-2026 §3"],
      [C(text="The 2026 table raised the single fare from 2.40 to 2.80 euros.", citation="FARES-2026 §3",
         key=["2.80"]),
       C(text="Single ride validity was extended from 90 to 120 minutes.", citation="FARES-2026 §3", key=["120"])],
      "The single ride fare rose from 2.40 to 2.80 euros and single ride validity was extended from 90 to 120 "
      "minutes.", ["changes in the 2026 fare table"], ["2026 fare table"]),
    S("T30", "TEMPORAL", "Which days have a holiday timetable in 2026?", ["HOLIDAY-SERVICE §1"],
      [C(text="On 1 January, 1 May and 25 December 2026 all lines run the Sunday timetable.",
         citation="HOLIDAY-SERVICE §1", key=["1 January", "1 May", "25 December"])],
      "1 January, 1 May and 25 December 2026 (Sunday timetable).", ["holiday timetable days in 2026"],
      ["holiday timetable"]),
    # ---------------------------------------------------------------- MULTI_HOP (station -> zone -> supplement)
    S("T31", "MULTI_HOP", "What supplement applies for a trip to Millbrook?", ["ZONES §1", "ZONE-RULES §1"],
      [C(text="Millbrook is classified as Zone A.", citation="ZONES §1", key=["Zone A"]),
       C(text="Trips that end in Zone A need no supplement.", citation="ZONE-RULES §1", key=["no supplement"])],
      "Millbrook is in Zone A, and trips ending in Zone A need no supplement.", ["supplement for Millbrook"],
      ["Millbrook"]),
    S("T32", "MULTI_HOP", "Do I need to validate my ticket before boarding for a trip to Harbor Point?",
      ["ZONES §1", "ZONE-RULES §3"],
      [C(text="Harbor Point is classified as Zone C.", citation="ZONES §1", key=["Zone C"]),
       C(text="For Zone C the ticket must be validated before boarding.", citation="ZONE-RULES §3",
         key=["validated before boarding"])],
      "Yes - Harbor Point is in Zone C, where the ticket must be validated before boarding.",
      ["validation rule for Harbor Point"], ["Harbor Point"]),
    S("T33", "MULTI_HOP", "How much extra is a trip to Quarry Lane?", ["ZONES §1", "ZONE-RULES §2"],
      [C(text="Quarry Lane is classified as Zone B.", citation="ZONES §1", key=["Zone B"]),
       C(text="Trips that end in Zone B need a supplement of 0.50 euros.", citation="ZONE-RULES §2", key=["0.50"])],
      "Quarry Lane is in Zone B: a supplement of 0.50 euros.", ["supplement for Quarry Lane"], ["Quarry Lane"]),
    S("T34", "MULTI_HOP", "Which zone is Millbrook in and what does that mean for my fare?",
      ["ZONES §1", "ZONE-RULES §1"],
      [C(text="Millbrook is classified as Zone A.", citation="ZONES §1", key=["Zone A"]),
       C(text="Trips that end in Zone A need no supplement.", citation="ZONE-RULES §1", key=["no supplement"])],
      "Millbrook is in Zone A; trips ending there need no supplement.", ["zone of Millbrook", "fare consequence"],
      ["Millbrook"]),
    S("T35", "MULTI_HOP", "Is the office that issues concession cards open on Sundays?", ["TCO §1", "TCO §2"],
      [C(text="The Travel Card Office issues all concession cards.", citation="TCO §1", key=["Travel Card Office"]),
       C(text="The TCO is closed on Sundays.", citation="TCO §2", key=["closed on Sundays"])],
      "No - the Travel Card Office, which issues concession cards, is closed on Sundays.",
      ["Sunday opening of the card office"], ["concession cards"]),
    # ---------------------------------------------------------------- AMBIGUOUS (any plausible section is relevant)
    S("T36", "AMBIGUOUS", "How much is it?", ["FARES-2026 §1", "TCO §1", "PENALTIES §1", "LOST §2"], [], "",
      ["a price (unspecified)"], semantics="any"),
    S("T37", "AMBIGUOUS", "How long does it take?", ["CONC-STUDENT §2", "CONC-SENIOR §2", "REFUNDS §2"], [], "",
      ["a duration (unspecified)"], semantics="any"),
    S("T38", "AMBIGUOUS", "Which form do I need?", ["CONC-STUDENT §2", "CONC-SENIOR §2", "REFUNDS §2"], [], "",
      ["a form (unspecified)"], semantics="any"),
    S("T39", "AMBIGUOUS", "Is it free?", ["CONC-SENIOR §1", "ZONE-RULES §1"], [], "", ["whether something is free"],
      semantics="any"),
    S("T40", "AMBIGUOUS", "When is it closed?", ["TCO §2"], [], "", ["a closing time (unspecified)"], semantics="any"),
    # ---------------------------------------------------------------- CONTRADICTORY
    S("T41", "CONTRADICTORY", "How often does the N4 night bus run?", ["NOTICE-X §1", "NOTICE-Y §1"], [],
      "The sources disagree: one notice says every 20 minutes, the other every 30 minutes.",
      ["frequency of the N4"], ["N4"], state="CONTRADICTORY", conflict=["20 minutes", "30 minutes"]),
    S("T42", "CONTRADICTORY", "What is the N4 frequency after midnight?", ["NOTICE-X §1", "NOTICE-Y §1"], [],
      "The sources disagree: every 20 or every 30 minutes.", ["N4 frequency"], ["N4"], state="CONTRADICTORY",
      conflict=["20 minutes", "30 minutes"]),
    S("T43", "CONTRADICTORY", "How long do I have to wait for the night bus N4?", ["NOTICE-X §1", "NOTICE-Y §1"], [],
      "The sources disagree: 20 or 30 minutes between departures.", ["N4 waiting time"], ["N4"],
      state="CONTRADICTORY", conflict=["20 minutes", "30 minutes"]),
    S("T44", "CONTRADICTORY", "How much is a single ride?", ["FARES-2026 §1"],
      [C(text="A single ride costs 2.80 euros.", citation="FARES-2026 §1", key=["2.80"])],
      "A single ride costs 2.80 euros (the 2025 price of 2.40 euros no longer applies).", ["single ride price"],
      ["single ride"], forbidden=[r"single ride (costs|cost|is) 2\.40"]),
    S("T45", "CONTRADICTORY", "How much is a monthly pass now?", ["FARES-2026 §1"],
      [C(text="A monthly pass costs 72 euros.", citation="FARES-2026 §1", key=["72 euros"])],
      "A monthly pass costs 72 euros.", ["current monthly pass price"], ["monthly pass"],
      forbidden=[r"monthly pass (costs|cost|is) 65"]),
    # ---------------------------------------------------------------- INSUFFICIENT_EVIDENCE
    S("T46", "INSUFFICIENT_EVIDENCE", "Is there free wifi on the buses?", [], [],
      "The documents do not say whether buses have wifi.", ["wifi on buses"], state="INSUFFICIENT"),
    S("T47", "INSUFFICIENT_EVIDENCE", "How much does it cost to take a dog on the train?", [], [],
      "The documents do not state a fare for dogs.", ["fare for dogs"], state="INSUFFICIENT",
      forbidden=[r"dog[^.]*\d+(?:\.\d+)? euros"]),
    S("T48", "INSUFFICIENT_EVIDENCE", "Can I pay the penalty fare by credit card?", [], [],
      "The documents do not say how the penalty can be paid.", ["payment method for penalties"],
      state="INSUFFICIENT"),
    S("T49", "INSUFFICIENT_EVIDENCE", "What is the student discount on a day pass?", [], [],
      "The documents state a student discount only for the monthly pass, not for the day pass.",
      ["student discount on a day pass"], ["day pass"], ["students"], state="INSUFFICIENT",
      forbidden=[r"day pass[^.]*50 percent", r"50 percent[^.]*day pass"]),
    S("T50", "INSUFFICIENT_EVIDENCE", "Until what time is the TCO open on Saturdays?", [], [],
      "The documents say the TCO is open Monday to Saturday but give no closing time.", ["Saturday closing time"],
      ["TCO"], state="INSUFFICIENT"),
]


def SESSION(sid, turns):
    out = []
    ctx: list[str] = []
    for n, t in enumerate(turns, start=1):
        d = dict(t)
        d.update(sample_id=f"{sid}.{n}", session_id=sid, turn_index=n, conversation_context=list(ctx))
        out.append(d)
        ctx.append(d["query"])
    return out


def turn(qtype, query, gold, claims, answer, intents, entities=(), constraints=(), forbidden=()):
    return S("_", qtype, query, gold, claims, answer, intents, entities, constraints, forbidden=forbidden)


MONTHLY = C(text="A monthly pass costs 72 euros.", citation="FARES-2026 §1", key=["72 euros"])
SESSIONS = [
    # CONTEXTUAL_FOLLOWUP
    SESSION("S01", [turn("SIMPLE", "What does a monthly pass cost?", ["FARES-2026 §1"], [MONTHLY],
                         "A monthly pass costs 72 euros.", ["monthly pass price"], ["monthly pass"]),
                    turn("CONTEXTUAL_FOLLOWUP", "And for students?", ["CONC-STUDENT §1"],
                         [C(text="Students receive a 50 percent discount on the monthly pass.",
                            citation="CONC-STUDENT §1", key=["50 percent"])],
                         "Students get a 50 percent discount on the monthly pass.", ["monthly pass price for students"],
                         ["monthly pass"], ["students"])]),
    SESSION("S02", [turn("MULTI_CONSTRAINT", "How do students apply for the concession?", ["CONC-STUDENT §2"],
                         [C(text="Students apply with Form TR-210 and a current enrolment certificate.",
                            citation="CONC-STUDENT §2", key=["TR-210"])],
                         "Students apply with Form TR-210 and a current enrolment certificate.",
                         ["how students apply"], ["concession"], ["students"]),
                    turn("CONTEXTUAL_FOLLOWUP", "What about seniors?", ["CONC-SENIOR §2"],
                         [C(text="Seniors apply with Form TR-305 and proof of age.", citation="CONC-SENIOR §2",
                            key=["TR-305"])], "Seniors apply with Form TR-305 and proof of age.",
                         ["how seniors apply"], ["concession"], ["seniors"])]),
    SESSION("S03", [turn("SIMPLE", "What is the penalty fare?", ["PENALTIES §1"],
                         [C(text="The penalty fare is 60 euros.", citation="PENALTIES §1", key=["60 euros"])],
                         "The penalty fare is 60 euros.", ["penalty fare amount"], ["penalty fare"]),
                    turn("CONTEXTUAL_FOLLOWUP", "Is there a way to pay less?", ["PENALTIES §2"],
                         [C(text="The penalty is reduced to 30 euros if paid within 7 days.", citation="PENALTIES §2",
                            key=["30 euros"])], "Yes - it is reduced to 30 euros if paid within 7 days.",
                         ["reducing the penalty"], ["penalty fare"])]),
    SESSION("S04", [turn("SIMPLE", "How long is a single ride ticket valid?", ["FARES-2026 §2"],
                         [C(text="A single ride ticket is valid for 120 minutes.", citation="FARES-2026 §2",
                            key=["120 minutes"])], "120 minutes after validation.", ["single ticket validity"],
                         ["single ride ticket"]),
                    turn("CONTEXTUAL_FOLLOWUP", "And how long was it valid in 2025?", ["FARES-2025 §2"],
                         [C(text="In 2025 a single ride ticket was valid for 90 minutes.", citation="FARES-2025 §2",
                            key=["90 minutes"])], "In 2025 it was valid for 90 minutes.",
                         ["single ticket validity in 2025"], ["single ride ticket"])]),
    SESSION("S05", [turn("MULTI_CONSTRAINT", "Which documents do students need for the concession?",
                         ["CONC-STUDENT §2"],
                         [C(text="Students apply with Form TR-210 and a current enrolment certificate.",
                            citation="CONC-STUDENT §2", key=["enrolment certificate"])],
                         "Form TR-210 and a current enrolment certificate.", ["documents for students"],
                         ["concession"], ["students"]),
                    turn("CONTEXTUAL_FOLLOWUP", "How long does it take to process?", ["CONC-STUDENT §2"],
                         [C(text="Student applications are processed within 5 working days.",
                            citation="CONC-STUDENT §2", key=["5 working days"])], "Within 5 working days.",
                         ["processing time for students"], ["concession"], ["students"])]),
    # ENTITY_CORRECTION
    SESSION("S06", [turn("MULTI_HOP", "What is the supplement for trips to Harbor Point?", ["ZONES §1", "ZONE-RULES §3"],
                         [C(text="Harbor Point is classified as Zone C.", citation="ZONES §1", key=["Zone C"]),
                          C(text="Zone C trips need a supplement of 1.20 euros.", citation="ZONE-RULES §3",
                            key=["1.20"])], "Harbor Point is in Zone C: a supplement of 1.20 euros.",
                         ["supplement for Harbor Point"], ["Harbor Point"]),
                    turn("ENTITY_CORRECTION", "Sorry, I meant Quarry Lane, not Harbor Point.",
                         ["ZONES §1", "ZONE-RULES §2"],
                         [C(text="Quarry Lane is classified as Zone B.", citation="ZONES §1", key=["Zone B"]),
                          C(text="Zone B trips need a supplement of 0.50 euros.", citation="ZONE-RULES §2",
                            key=["0.50"])], "Quarry Lane is in Zone B: a supplement of 0.50 euros.",
                         ["supplement for Quarry Lane"], ["Quarry Lane"], forbidden=[r"\b1\.20\b"])]),
    SESSION("S07", [turn("MULTI_CONSTRAINT", "How long does processing take for students?", ["CONC-STUDENT §2"],
                         [C(text="Student applications are processed within 5 working days.",
                            citation="CONC-STUDENT §2", key=["5 working days"])], "Within 5 working days.",
                         ["processing time"], [], ["students"]),
                    turn("ENTITY_CORRECTION", "Sorry, I meant seniors, not students.", ["CONC-SENIOR §2"],
                         [C(text="Senior applications are processed within 10 working days.",
                            citation="CONC-SENIOR §2", key=["10 working days"])], "Within 10 working days.",
                         ["processing time for seniors"], [], ["seniors"], forbidden=[r"\b5 working days"])]),
    SESSION("S08", [turn("SIMPLE", "How much does a day pass cost?", ["FARES-2026 §1"],
                         [C(text="A day pass costs 8.00 euros.", citation="FARES-2026 §1", key=["8.00"])],
                         "A day pass costs 8.00 euros.", ["day pass price"], ["day pass"]),
                    turn("ENTITY_CORRECTION", "Sorry, I meant the monthly pass, not the day pass.", ["FARES-2026 §1"],
                         [MONTHLY], "A monthly pass costs 72 euros.", ["monthly pass price"], ["monthly pass"])]),
    SESSION("S09", [turn("SIMPLE", "Which form is used for refunds?", ["REFUNDS §2"],
                         [C(text="Refunds are requested with Form TR-410.", citation="REFUNDS §2", key=["TR-410"])],
                         "Form TR-410.", ["refund form"], ["refunds"]),
                    turn("ENTITY_CORRECTION", "Sorry, I meant the form for the student concession instead.",
                         ["CONC-STUDENT §2"],
                         [C(text="Students apply with Form TR-210.", citation="CONC-STUDENT §2", key=["TR-210"])],
                         "The student concession uses Form TR-210.", ["student concession form"],
                         ["student concession"])]),
    SESSION("S10", [turn("SIMPLE", "Are bicycles allowed on trains at 8:00?", ["BIKES §1"],
                         [C(text="Peak hours are 7:00 to 9:00; bicycles are allowed only outside peak hours.",
                            citation="BIKES §1", key=["7:00"])], "No - 8:00 is within peak hours (7:00-9:00).",
                         ["bicycles at 8:00"], ["bicycles"]),
                    turn("ENTITY_CORRECTION", "I meant folding bicycles, not ordinary ones.", ["BIKES §2"],
                         [C(text="Folded bicycles are allowed at all times.", citation="BIKES §2", key=["all times"])],
                         "Folded bicycles are allowed at all times.", ["folding bicycles"], ["folding bicycles"])]),
    # REPEATED_QUERY
    SESSION("S11", [turn("SIMPLE", "What does a day pass cost?", ["FARES-2026 §1"],
                         [C(text="A day pass costs 8.00 euros.", citation="FARES-2026 §1", key=["8.00"])],
                         "8.00 euros.", ["day pass price"], ["day pass"]),
                    turn("EXACT_TERM", "When is the TCO closed?", ["TCO §2"],
                         [C(text="The TCO is closed on Sundays.", citation="TCO §2", key=["Sunday"])], "On Sundays.",
                         ["TCO closing day"], ["TCO"]),
                    turn("REPEATED_QUERY", "What does a day pass cost?", ["FARES-2026 §1"],
                         [C(text="A day pass costs 8.00 euros.", citation="FARES-2026 §1", key=["8.00"])],
                         "8.00 euros.", ["day pass price"], ["day pass"])]),
    SESSION("S12", [turn("SIMPLE", "How long are lost items kept?", ["LOST §1"],
                         [C(text="Lost items are kept for 30 days.", citation="LOST §1", key=["30 days"])], "30 days.",
                         ["how long lost items are kept"], ["lost items"]),
                    turn("REPEATED_QUERY", "How long are lost items kept?", ["LOST §1"],
                         [C(text="Lost items are kept for 30 days.", citation="LOST §1", key=["30 days"])], "30 days.",
                         ["how long lost items are kept"], ["lost items"])]),
    SESSION("S13", [turn("SIMPLE", "What is the refund period for a monthly pass?", ["REFUNDS §1"],
                         [C(text="An unused monthly pass can be refunded within 14 days.", citation="REFUNDS §1",
                            key=["14 days"])], "Within 14 days of purchase.", ["refund period"], ["monthly pass"]),
                    turn("SIMPLE", "Which form is used for refunds?", ["REFUNDS §2"],
                         [C(text="Refunds are requested with Form TR-410.", citation="REFUNDS §2", key=["TR-410"])],
                         "Form TR-410.", ["refund form"], ["refunds"]),
                    turn("REPEATED_QUERY", "What is the refund period for an unused monthly pass?", ["REFUNDS §1"],
                         [C(text="An unused monthly pass can be refunded within 14 days.", citation="REFUNDS §1",
                            key=["14 days"])], "Within 14 days of purchase.", ["refund period"], ["monthly pass"])]),
    SESSION("S14", [turn("MULTI_HOP", "Do I need a supplement for Quarry Lane?", ["ZONES §1", "ZONE-RULES §2"],
                         [C(text="Quarry Lane trips need a supplement of 0.50 euros.", citation="ZONE-RULES §2",
                            key=["0.50"])], "Yes - Quarry Lane is in Zone B: 0.50 euros.",
                         ["supplement for Quarry Lane"], ["Quarry Lane"]),
                    turn("REPEATED_QUERY", "How much extra is a trip to Quarry Lane?", ["ZONES §1", "ZONE-RULES §2"],
                         [C(text="Quarry Lane trips need a supplement of 0.50 euros.", citation="ZONE-RULES §2",
                            key=["0.50"])], "0.50 euros (Zone B).", ["supplement for Quarry Lane"], ["Quarry Lane"])]),
    SESSION("S15", [turn("TEMPORAL", "Which days have a holiday timetable in 2026?", ["HOLIDAY-SERVICE §1"],
                         [C(text="1 January, 1 May and 25 December 2026.", citation="HOLIDAY-SERVICE §1",
                            key=["1 May"])], "1 January, 1 May and 25 December 2026.", ["holiday timetable days"],
                         ["holiday timetable"]),
                    turn("REPEATED_QUERY", "On which days in 2026 does the holiday timetable apply?",
                         ["HOLIDAY-SERVICE §1"],
                         [C(text="1 January, 1 May and 25 December 2026.", citation="HOLIDAY-SERVICE §1",
                            key=["1 May"])], "1 January, 1 May and 25 December 2026.", ["holiday timetable days"],
                         ["holiday timetable"])]),
]
STREAMING = [
    S("SC1", "STREAMING_CORRECTION", "How long does processing take for students?", ["CONC-STUDENT §2"],
      [C(text="Student applications are processed within 5 working days.", citation="CONC-STUDENT §2",
         key=["5 working days"])], "Within 5 working days.", ["processing time"], [], ["students"],
      forbidden=[r"\b10 working days"],
      stream=["How long does processing take", "for seniors", (1, "for students")]),
    S("SC2", "STREAMING_CORRECTION", "What does a monthly pass cost?", ["FARES-2026 §1"], [MONTHLY],
      "A monthly pass costs 72 euros.", ["monthly pass price"], ["monthly pass"],
      forbidden=[r"day pass (costs|cost|is)"], stream=["What does a", "day pass cost", (1, "monthly pass cost")]),
    S("SC3", "STREAMING_CORRECTION", "What supplement applies for Harbor Point?", ["ZONES §1", "ZONE-RULES §3"],
      [C(text="Zone C trips need a supplement of 1.20 euros.", citation="ZONE-RULES §3", key=["1.20"])],
      "Harbor Point is in Zone C: 1.20 euros.", ["supplement for Harbor Point"], ["Harbor Point"],
      stream=["What supplement applies", "for Quarry Lane", (1, "for Harbor Point")]),
    S("SC4", "STREAMING_CORRECTION", "When is the TCO closed?", ["TCO §2"],
      [C(text="The TCO is closed on Sundays.", citation="TCO §2", key=["Sunday"])], "On Sundays.",
      ["TCO closing day"], ["TCO"], stream=["When is the", "TCO open", (1, "TCO closed")]),
    S("SC5", "STREAMING_CORRECTION", "How long are lost items kept?", ["LOST §1"],
      [C(text="Lost items are kept for 30 days.", citation="LOST §1", key=["30 days"])], "30 days.",
      ["how long lost items are kept"], ["lost items"], stream=["How long are", "refunds", (1, "lost items"), "kept?"]),
]

# ---------------------------------------------------------------- DEV split (converted earlier sets)
P9_TYPES = {"simple": "SIMPLE", "exact_keyword": "EXACT_TERM", "semantic": "SEMANTIC", "constraint": "MULTI_CONSTRAINT",
            "temporal": "TEMPORAL", "multi_hop": "MULTI_HOP", "multi_intent": "MULTI_INTENT",
            "contradiction": "CONTRADICTORY", "insufficient": "INSUFFICIENT_EVIDENCE", "ambiguous": "AMBIGUOUS",
            "contextual_follow_up": "CONTEXTUAL_FOLLOWUP", "late_constraint": "CONTEXTUAL_FOLLOWUP",
            "follow_up": "CONTEXTUAL_FOLLOWUP", "cross_query_reuse": "CONTEXTUAL_FOLLOWUP",
            "entity_correction": "ENTITY_CORRECTION", "constraint_change": "ENTITY_CORRECTION",
            "repeated_question": "REPEATED_QUERY", "equivalent_repeat": "REPEATED_QUERY"}


def dev_samples() -> list[dict]:
    out = []
    for d, src, pre in (("dev_adaptive_retrieval", "phase9_dev", "p9d-"),
                        ("heldout_adaptive_retrieval", "phase9_heldout", "p9h-")):
        for p in sorted((REPO / "eval" / d).glob("*.json")):
            c = json.loads(p.read_text())
            ctx: list[str] = []
            for n, t in enumerate(c["turns"], start=1):
                cat = P9_TYPES[c["category"]]
                if len(c["turns"]) > 1 and n == 1:
                    cat = "SIMPLE"
                if cat == "MULTI_HOP":
                    srcx = src + "_multi_hop"
                else:
                    srcx = src
                out.append(dict(sample_id=f"{pre}{c['case_id']}.{n}", split="dev", source=srcx,
                                corpus=c["corpus"], session_id=pre + c["case_id"], turn_index=n,
                                query=t["utterance_text"],
                                conversation_context=list(ctx), query_type=cat,
                                ground_truth_evidence=t["gold"]["citations"],
                                gold_semantics="any" if cat == "AMBIGUOUS" else "all",
                                expected_state=t["gold"]["expect"],
                                source_documents=sorted({g.split(" §")[0] for g in t["gold"]["citations"]})))
                ctx.append(t["utterance_text"])
    for p in sorted((REPO / "eval" / "dev_grounded").glob("G*.json")):
        c = json.loads(p.read_text())
        ctx = []
        for n, t in enumerate(c["turns"], start=1):
            g = t["gold"]
            facts = g["required_facts"]
            state = "CONTRADICTORY" if g["conflict_values"] else "INSUFFICIENT" if g["expect_uncertainty"] and not facts \
                else "SUFFICIENT"
            cat = ("CONTRADICTORY" if state == "CONTRADICTORY" else "INSUFFICIENT_EVIDENCE" if state == "INSUFFICIENT"
                   else "MULTI_INTENT" if g.get("intents", 1) > 1 else "CONTEXTUAL_FOLLOWUP" if n > 1 else "SIMPLE")
            out.append(dict(sample_id=f"p7-{c['case_id']}.{n}", split="dev", source="phase7_dev_grounded",
                            corpus=c["corpus"], session_id="p7-" + c["case_id"], turn_index=n,
                            query=t["utterance_text"],
                            conversation_context=list(ctx), query_type=cat,
                            ground_truth_evidence=sorted({f["citation"] for f in facts}),
                            expected_claims=[ExpectedClaim(text=f["text"], citation=f["citation"]) for f in facts],
                            expected_state=state, conflict_values=list(g["conflict_values"]),
                            forbidden=list(g["forbidden"]),
                            source_documents=sorted({f["citation"].split(" §")[0] for f in facts})))
            ctx.append(t["utterance_text"])
    return out


def lexical_gap_fn():
    """Share of the query's content terms that occur in its gold evidence (analyzer of the Phase 3 index)."""
    from streamrag.adaptive.lexicon import QUESTION_WORDS
    from streamrag.config import load_config
    from streamrag.retrieval import build_index
    sys.path.insert(0, str(REPO / "experiments" / "runners"))
    from corpora import CORPORA
    cache = {}

    def texts(corpus):
        if corpus not in cache:
            cfg = load_config(REPO / "configs" / "default.yaml",
                              {"paths.corpus": str(REPO / CORPORA[corpus]), "paths.index_root": "/tmp/p10_dataset_idx/"
                               + corpus, "telemetry.log_level": "ERROR", "dense.embedder": "hashing"}, base_dir=REPO)
            b = build_index(cfg)
            cache[corpus] = ({c.citation: f"{c.title} {c.section_title or ''} {c.text}" for c in b.chunks}, b.analyzer)
        return cache[corpus]

    def gap(sample: dict) -> bool:
        if not sample["ground_truth_evidence"]:
            return False
        cit, an = texts(sample["corpus"])
        q = {t for t in an.tokens(sample["query"])} - {t for w in QUESTION_WORDS for t in an.tokens(w)}
        if not q:
            return False
        gold = set().union(*(set(an.tokens(cit.get(g, ""))) for g in sample["ground_truth_evidence"]))
        return len(q & gold) / len(q) < 0.34
    return gap


def main() -> None:
    gap = lexical_gap_fn()
    test = [x for x in TEST + STREAMING] + [t for s in SESSIONS for t in s]
    rows = {"test": [], "dev": []}
    for split, items in (("test", test), ("dev", dev_samples())):
        for d in items:
            s = EvalSample(**d)
            diff, feats = difficulty(s, gap(d))
            rows[split].append(s.model_copy(update={"difficulty": diff, "difficulty_features": feats}))
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, ss in rows.items():
        p = OUT / f"{split}.jsonl"
        p.write_text(dumps(ss))
        paths[split] = p
    man = manifest(paths, {"construction": "experiments/datasets/build_dataset.py",
                           "labels": "implementer-written from the fixture corpus text; no independent annotation",
                           "test_corpus": "tests/fixtures/corpus_eval_transit (new in Phase 10; never used in development)",
                           "reportable": False})
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
