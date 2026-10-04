"""Builds experiments/datasets/streamrag_eval_v2/test.jsonl + manifest.json (Phase 11 held-out set).

Why a second held-out set:
* The Phase 10 test split (streamrag_eval_v1) was inspected during the Phase 10 error analysis, and Phase 11 fixes
  are derived from those failures, so v1 is no longer blind for Phase 11.
* v2 is a NEW fictional domain (Lakeside water and waste utility, tests/fixtures/corpus_eval_utility).
* The corpus and these labels were written and frozen (hash in manifest.json) BEFORE any Phase 11 system change, and
  no component is developed or tuned on it. It is run once on the final system and the baselines.

Same schema and difficulty rule as v1 (src/streamrag/evaluation/dataset.py). All labels are implementer-written from
the corpus text, with no independent annotation. TEST FIXTURE ONLY - NOT REPORTABLE.

Usage: .venv/bin/python experiments/datasets/build_dataset_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import lexical_gap_fn  # noqa: E402

from streamrag.evaluation.dataset import EvalSample, ExpectedClaim, StreamChunk, difficulty, dumps, manifest  # noqa: E402

OUT = REPO / "experiments" / "datasets" / "streamrag_eval_v2"
U = "utility"
C = ExpectedClaim


def S(sid, qtype, query, gold, claims, answer, intents, entities=(), constraints=(), state="SUFFICIENT",
      conflict=(), forbidden=(), semantics="all", stream=()):
    return dict(sample_id=sid, split="test", source="phase11_test_v2", corpus=U, session_id=sid, turn_index=1,
                query=query, stream=[StreamChunk(text=c) if isinstance(c, str) else StreamChunk(text=c[1], replaces=c[0])
                                     for c in stream],
                conversation_context=[], query_type=qtype, expected_intents=list(intents),
                required_entities=list(entities), required_constraints=list(constraints),
                ground_truth_evidence=list(gold), gold_semantics=semantics, expected_claims=list(claims),
                expected_answer=answer, expected_state=state, conflict_values=list(conflict),
                forbidden=list(forbidden), source_documents=sorted({g.split(" §")[0] for g in gold}))


def SESSION(sid, turns):
    out, ctx = [], []
    for n, t in enumerate(turns, start=1):
        d = dict(t)
        d.update(sample_id=f"{sid}.{n}", session_id=sid, turn_index=n, conversation_context=list(ctx))
        out.append(d)
        ctx.append(d["query"])
    return out


def turn(qtype, query, gold, claims, answer, intents, entities=(), constraints=(), forbidden=()):
    return S("_", qtype, query, gold, claims, answer, intents, entities, constraints, forbidden=forbidden)


BASE = C(text="The monthly base charge is 13.50 euros.", citation="TARIFF-2026 §1", key=["13.50"])
STALE_BASE = [r"base charge is 12\.00"]
METER_TEST = C(text="A meter test costs 25 euros.", citation="METER §2", key=["25 euros"])
CONNECTION = C(text="A new water connection costs 450 euros.", citation="CONNECT §1", key=["450 euros"])
HARDSHIP_TIME = C(text="Hardship applications are decided within 15 working days.", citation="RELIEF-HARDSHIP §2",
                  key=["15 working days"])
PENSIONER_TIME = C(text="Pensioner applications are decided within 20 working days.", citation="RELIEF-PENSIONER §2",
                   key=["20 working days"])
REMINDER = C(text="A payment reminder costs 5 euros.", citation="LATE §1", key=["5 euros"])
UCC_HOURS = C(text="The UCC is open Monday to Friday from 8:00 to 16:00.", citation="UCC §2", key=["8:00", "16:00"])

TEST = [
    # ---------------------------------------------------------------- SIMPLE
    S("V01", "SIMPLE", "How much is the monthly base charge?", ["TARIFF-2026 §1"], [BASE],
      "The monthly base charge is 13.50 euros.", ["monthly base charge"], ["base charge"], forbidden=STALE_BASE),
    S("V02", "SIMPLE", "How much does a meter test cost?", ["METER §2"], [METER_TEST], "A meter test costs 25 euros.",
      ["price of a meter test"], ["meter test"]),
    S("V03", "SIMPLE", "How long does a new water connection take?", ["CONNECT §1"],
      [C(text="New connection applications take 6 weeks to complete.", citation="CONNECT §1", key=["6 weeks"])],
      "A new water connection takes 6 weeks.", ["duration of a new connection"], ["new connection"]),
    S("V04", "SIMPLE", "How much is a payment reminder?", ["LATE §1"], [REMINDER], "A payment reminder costs 5 euros.",
      ["reminder fee"], ["payment reminder"]),
    S("V05", "SIMPLE", "How often is garden waste collected?", ["COLLECTION §2"],
      [C(text="Garden waste is collected every second week from April to October.", citation="COLLECTION §2",
         key=["second week"])], "Every second week from April to October.", ["garden waste frequency"],
      ["garden waste"]),
    S("V06", "SIMPLE", "How far in advance must a bulky waste pickup be booked?", ["BULKY §1"],
      [C(text="Bulky waste pickups are booked at least 5 days in advance.", citation="BULKY §1", key=["5 days"])],
      "At least 5 days in advance.", ["booking lead time for bulky waste"], ["bulky waste"]),
    S("V07", "SIMPLE", "What does a new water connection cost?", ["CONNECT §1"], [CONNECTION],
      "A new water connection costs 450 euros.", ["price of a new connection"], ["new connection"]),
    S("V08", "SIMPLE", "How long can a payment plan last?", ["LATE §3"],
      [C(text="A payment plan can last up to 12 months.", citation="LATE §3", key=["12 months"])],
      "Up to 12 months.", ["payment plan length"], ["payment plan"]),
    # ---------------------------------------------------------------- SEMANTIC
    S("V09", "SEMANTIC", "Will you shut off my water completely if I cannot pay?", ["LATE §2"],
      [C(text="After 60 days of non-payment the supply is restricted to a minimum flow, never cut off completely.",
         citation="LATE §2", key=["minimum flow"])],
      "No - after 60 days the supply is restricted to a minimum flow but never cut off completely.",
      ["consequence of non-payment"], ["water supply"]),
    S("V10", "SEMANTIC", "I think my water meter counts too much. What can I do?", ["METER §1"],
      [C(text="A disputed meter reading must be reported within 30 days of the bill with Form UT-310.",
         citation="METER §1", key=["UT-310"])],
      "Report the disputed reading within 30 days of the bill with Form UT-310.", ["disputing a meter reading"],
      ["water meter"]),
    S("V11", "SEMANTIC", "Do I get my money back if the meter turns out to be broken?", ["METER §2"],
      [C(text="The meter test fee is refunded if the meter is found to be faulty.", citation="METER §2",
         key=["refunded"])], "Yes - the 25 euro test fee is refunded if the meter is faulty.",
      ["refund of the meter test fee"], ["meter test"]),
    S("V12", "SEMANTIC", "Can I get rid of an old sofa and a wardrobe without paying?", ["BULKY §2"],
      [C(text="The first two items are collected free of charge.", citation="BULKY §2", key=["first two items"])],
      "Yes - the first two bulky items are collected free of charge.", ["cost of removing two bulky items"],
      ["bulky items"]),
    S("V13", "SEMANTIC", "Is there help with the water bill for people living on benefits?", ["RELIEF-HARDSHIP §1"],
      [C(text="Households receiving income support get a 40 percent reduction on the monthly base charge.",
         citation="RELIEF-HARDSHIP §1", key=["40 percent"])],
      "Yes - households on income support get a 40 percent reduction on the base charge.",
      ["bill support for benefit recipients"], ["income support"]),
    # ---------------------------------------------------------------- EXACT_TERM
    S("V14", "EXACT_TERM", "What is Form UT-310 for?", ["METER §1"],
      [C(text="Form UT-310 is used to report a disputed meter reading.", citation="METER §1", key=["meter"])],
      "Form UT-310 is used to report a disputed meter reading.", ["purpose of Form UT-310"], ["Form UT-310"]),
    S("V15", "EXACT_TERM", "Who uses Form UT-150?", ["RELIEF-PENSIONER §2"],
      [C(text="Pensioners apply with Form UT-150 and proof of age.", citation="RELIEF-PENSIONER §2",
         key=["pensioner"])], "Pensioners use Form UT-150 with proof of age.", ["users of Form UT-150"],
      ["Form UT-150"]),
    S("V16", "EXACT_TERM", "When is the UCC open?", ["UCC §2"], [UCC_HOURS],
      "The UCC is open Monday to Friday from 8:00 to 16:00.", ["UCC opening hours"], ["UCC"]),
    S("V17", "EXACT_TERM", "What does the UCC handle?", ["UCC §1"],
      [C(text="The UCC handles all relief applications and meter disputes.", citation="UCC §1",
         key=["relief applications"])], "The UCC handles all relief applications and meter disputes.",
      ["responsibilities of the UCC"], ["UCC"]),
    S("V18", "EXACT_TERM", "What must be submitted together with Form UT-120?", ["RELIEF-HARDSHIP §2"],
      [C(text="Households apply with Form UT-120 and a current benefit letter.", citation="RELIEF-HARDSHIP §2",
         key=["benefit letter"])], "A current benefit letter.", ["documents for Form UT-120"], ["Form UT-120"]),
    # ---------------------------------------------------------------- MULTI_INTENT
    S("V19", "MULTI_INTENT", "How much does a meter test cost and when is the fee refunded?", ["METER §2"],
      [METER_TEST, C(text="The fee is refunded if the meter is found to be faulty.", citation="METER §2",
                     key=["faulty"])],
      "A meter test costs 25 euros; the fee is refunded if the meter is faulty.",
      ["price of a meter test", "refund condition"], ["meter test"]),
    S("V20", "MULTI_INTENT", "How much is a new connection and which form do I need?", ["CONNECT §1"],
      [CONNECTION, C(text="Applications use Form UT-200.", citation="CONNECT §1", key=["UT-200"])],
      "A new connection costs 450 euros and uses Form UT-200.", ["connection price", "connection form"],
      ["new connection"]),
    S("V21", "MULTI_INTENT", "What does a payment reminder cost and when is the supply restricted?",
      ["LATE §1", "LATE §2"],
      [REMINDER, C(text="After 60 days of non-payment the water supply is restricted.", citation="LATE §2",
                   key=["60 days"])],
      "A reminder costs 5 euros; after 60 days of non-payment the supply is restricted.",
      ["reminder fee", "when supply is restricted"], ["payment reminder", "water supply"]),
    S("V22", "MULTI_INTENT", "How many bulky items are free and what does each extra item cost?", ["BULKY §2"],
      [C(text="The first two items are collected free of charge.", citation="BULKY §2", key=["two"]),
       C(text="Each additional item costs 8 euros.", citation="BULKY §2", key=["8 euros"])],
      "The first two items are free; each additional item costs 8 euros.",
      ["number of free items", "price of extra items"], ["bulky items"]),
    # ---------------------------------------------------------------- MULTI_CONSTRAINT
    S("V23", "MULTI_CONSTRAINT", "How long does a decision take for hardship applications?", ["RELIEF-HARDSHIP §2"],
      [HARDSHIP_TIME], "Within 15 working days.", ["decision time"], [], ["hardship"],
      forbidden=[r"\b20 working days"]),
    S("V24", "MULTI_CONSTRAINT", "How long does a decision take for pensioners?", ["RELIEF-PENSIONER §2"],
      [PENSIONER_TIME], "Within 20 working days.", ["decision time"], [], ["pensioners"],
      forbidden=[r"\b15 working days"]),
    S("V25", "MULTI_CONSTRAINT", "Which form do households on income support use?", ["RELIEF-HARDSHIP §2"],
      [C(text="Households apply with Form UT-120.", citation="RELIEF-HARDSHIP §2", key=["UT-120"])],
      "Form UT-120.", ["relief form"], [], ["income support"]),
    S("V26", "MULTI_CONSTRAINT", "What do pensioners pay for a meter replacement?", ["RELIEF-PENSIONER §1"],
      [C(text="Residents aged 67 or older pay no fee for a meter replacement.", citation="RELIEF-PENSIONER §1",
         key=["no fee"])], "Nothing - residents aged 67 or older pay no fee for a meter replacement.",
      ["meter replacement fee"], ["meter replacement"], ["pensioners"]),
    S("V27", "MULTI_CONSTRAINT", "What reduction on the base charge do households on income support get?",
      ["RELIEF-HARDSHIP §1"],
      [C(text="Households receiving income support get a 40 percent reduction.", citation="RELIEF-HARDSHIP §1",
         key=["40 percent"])], "A 40 percent reduction on the monthly base charge.", ["base charge reduction"],
      ["base charge"], ["income support"]),
    # ---------------------------------------------------------------- TEMPORAL
    S("V28", "TEMPORAL", "What was the monthly base charge in 2025?", ["TARIFF-2025 §1"],
      [C(text="In 2025 the monthly base charge was 12.00 euros.", citation="TARIFF-2025 §1", key=["12.00"])],
      "12.00 euros.", ["base charge in 2025"], ["base charge"], ["2025"]),
    S("V29", "TEMPORAL", "How often were bills issued before April 2026?", ["TARIFF-2025 §2"],
      [C(text="Bills were issued every two months.", citation="TARIFF-2025 §2", key=["two months"])],
      "Every two months.", ["billing frequency before April 2026"], ["bills"], ["before April 2026"]),
    S("V30", "TEMPORAL", "How often are bills issued now?", ["TARIFF-2026 §2"],
      [C(text="Bills are issued every month.", citation="TARIFF-2026 §2", key=["every month"])], "Every month.",
      ["current billing frequency"], ["bills"], ["now"], forbidden=[r"bills are issued every two months"]),
    S("V31", "TEMPORAL", "What will the monthly base charge be in May 2026?", ["TARIFF-2026 §1"], [BASE],
      "13.50 euros.", ["base charge in May 2026"], ["base charge"], ["May 2026"], forbidden=STALE_BASE),
    S("V32", "TEMPORAL", "What happens to collections on public holidays in 2026?", ["HOLIDAY-COLLECTION §1"],
      [C(text="Collections on a public holiday move to the next working day.", citation="HOLIDAY-COLLECTION §1",
         key=["next working day"])], "They move to the next working day.", ["holiday collections in 2026"],
      ["collections"], ["2026"]),
    # ---------------------------------------------------------------- MULTI_HOP
    S("V33", "MULTI_HOP", "When are bins collected in Oakridge?", ["DISTRICTS §1", "COLLECTION §1"],
      [C(text="Oakridge is in Service Area North, where bins are collected on Mondays.", citation="COLLECTION §1",
         key=["Monday"])], "Oakridge is in Service Area North: bins are collected on Mondays.",
      ["collection day in Oakridge"], ["Oakridge"]),
    S("V34", "MULTI_HOP", "Is the office that handles meter disputes open on Saturdays?", ["UCC §1", "UCC §2"],
      [C(text="The UCC, which handles meter disputes, is closed at weekends.", citation="UCC §2", key=["weekend"])],
      "No - the UCC, which handles meter disputes, is closed at weekends.", ["Saturday opening of that office"],
      ["meter disputes office"]),
    S("V35", "MULTI_HOP", "Which day are bins collected in Kestrel Point?", ["DISTRICTS §1", "COLLECTION §1"],
      [C(text="Kestrel Point is in Service Area East, where bins are collected on Fridays.", citation="COLLECTION §1",
         key=["Friday"])], "On Fridays (Service Area East).", ["collection day in Kestrel Point"], ["Kestrel Point"]),
    S("V36", "MULTI_HOP", "When can I visit the office that handles relief applications?", ["UCC §1", "UCC §2"],
      [UCC_HOURS], "The UCC is open Monday to Friday from 8:00 to 16:00.", ["opening hours of that office"],
      ["relief office"]),
    S("V37", "MULTI_HOP", "When are bins collected in Fernhill?", ["DISTRICTS §1", "COLLECTION §1"],
      [C(text="Fernhill is in Service Area South, where bins are collected on Wednesdays.", citation="COLLECTION §1",
         key=["Wednesday"])], "On Wednesdays (Service Area South).", ["collection day in Fernhill"], ["Fernhill"]),
    # ---------------------------------------------------------------- AMBIGUOUS
    S("V38", "AMBIGUOUS", "How much does it cost?",
      ["TARIFF-2026 §1", "METER §2", "CONNECT §1", "BULKY §2", "LATE §1"], [], "", ["a price (unspecified)"],
      semantics="any"),
    S("V39", "AMBIGUOUS", "How long does it take?", ["RELIEF-HARDSHIP §2", "RELIEF-PENSIONER §2", "CONNECT §1"], [],
      "", ["a duration (unspecified)"], semantics="any"),
    S("V40", "AMBIGUOUS", "Which form do I need?",
      ["RELIEF-HARDSHIP §2", "RELIEF-PENSIONER §2", "METER §1", "CONNECT §1"], [], "", ["a form (unspecified)"],
      semantics="any"),
    S("V41", "AMBIGUOUS", "When is it closed?", ["UCC §2"], [], "", ["a closing time (unspecified)"],
      semantics="any"),
    # ---------------------------------------------------------------- CONTRADICTORY
    S("V42", "CONTRADICTORY", "When is the water supply on Harbour Road interrupted?", ["OUTAGE-A §1", "OUTAGE-B §1"],
      [], "The notices disagree: 9:00-13:00 (Notice A) vs 10:00-15:00 (Notice B) on 12 November 2026.",
      ["Harbour Road interruption time"], ["Harbour Road"], state="CONTRADICTORY", conflict=["9:00", "10:00"]),
    S("V43", "CONTRADICTORY", "Until what time will Harbour Road have no water on 12 November?",
      ["OUTAGE-A §1", "OUTAGE-B §1"], [], "The notices disagree: until 13:00 (Notice A) or until 15:00 (Notice B).",
      ["end of the Harbour Road interruption"], ["Harbour Road"], state="CONTRADICTORY", conflict=["13:00", "15:00"]),
    S("V44", "CONTRADICTORY", "How much is the monthly base charge now?", ["TARIFF-2026 §1"], [BASE],
      "13.50 euros (the 2026 tariff supersedes the 2025 tariff).", ["current base charge"], ["base charge"],
      forbidden=STALE_BASE),
    S("V45", "CONTRADICTORY", "How much does water usage cost per cubic metre?", ["TARIFF-2026 §1"],
      [C(text="Water usage costs 2.05 euros per cubic metre.", citation="TARIFF-2026 §1", key=["2.05"])],
      "2.05 euros per cubic metre.", ["usage price"], ["water usage"], forbidden=[r"usage costs 1\.80"]),
    # ---------------------------------------------------------------- INSUFFICIENT_EVIDENCE
    S("V46", "INSUFFICIENT_EVIDENCE", "Can I pay my water bill with a credit card?", [], [],
      "The sources do not say.", ["payment methods"], ["credit card"], state="INSUFFICIENT"),
    S("V47", "INSUFFICIENT_EVIDENCE", "How much does a sewer inspection cost?", [], [], "The sources do not say.",
      ["price of a sewer inspection"], ["sewer inspection"], state="INSUFFICIENT",
      forbidden=[r"sewer[^.]*\d+ euros"]),
    S("V48", "INSUFFICIENT_EVIDENCE", "What discount do pensioners get on the monthly base charge?", [], [],
      "The sources do not say (pensioners only pay no fee for a meter replacement).",
      ["pensioner base charge discount"], ["base charge"], ["pensioners"], state="INSUFFICIENT",
      forbidden=[r"pensioner[^.]*\d+ percent", r"\d+ percent[^.]*pensioner"]),
    S("V49", "INSUFFICIENT_EVIDENCE", "Is the UCC open on public holidays?", [], [], "The sources do not say.",
      ["UCC holiday opening"], ["UCC"], state="INSUFFICIENT"),
]

SESSIONS = [
    # CONTEXTUAL_FOLLOWUP
    SESSION("VS01", [turn("SIMPLE", "How much is the monthly base charge?", ["TARIFF-2026 §1"], [BASE], "13.50 euros.",
                          ["monthly base charge"], ["base charge"], forbidden=STALE_BASE),
                     turn("CONTEXTUAL_FOLLOWUP", "And for households on income support?", ["RELIEF-HARDSHIP §1"],
                          [C(text="Households receiving income support get a 40 percent reduction.",
                             citation="RELIEF-HARDSHIP §1", key=["40 percent"])],
                          "They get a 40 percent reduction on the base charge.", ["base charge for income support"],
                          ["base charge"], ["income support"])]),
    SESSION("VS02", [turn("MULTI_CONSTRAINT", "How do pensioners apply for relief?", ["RELIEF-PENSIONER §2"],
                          [C(text="Pensioners apply with Form UT-150 and proof of age.", citation="RELIEF-PENSIONER §2",
                             key=["UT-150"])], "With Form UT-150 and proof of age.", ["how pensioners apply"],
                          ["relief"], ["pensioners"]),
                     turn("CONTEXTUAL_FOLLOWUP", "What about households on income support?", ["RELIEF-HARDSHIP §2"],
                          [C(text="Households apply with Form UT-120 and a current benefit letter.",
                             citation="RELIEF-HARDSHIP §2", key=["UT-120"])],
                          "With Form UT-120 and a current benefit letter.", ["how income support households apply"],
                          ["relief"], ["income support"])]),
    SESSION("VS03", [turn("SIMPLE", "What does a meter test cost?", ["METER §2"], [METER_TEST], "25 euros.",
                          ["meter test price"], ["meter test"]),
                     turn("CONTEXTUAL_FOLLOWUP", "Do I get that back?", ["METER §2"],
                          [C(text="The fee is refunded if the meter is found to be faulty.", citation="METER §2",
                             key=["refunded"])], "Yes, if the meter is found to be faulty.",
                          ["refund of the meter test fee"], ["meter test"])]),
    SESSION("VS04", [turn("SIMPLE", "When are bins collected in Service Area North?", ["COLLECTION §1"],
                          [C(text="In Service Area North household bins are collected on Mondays.",
                             citation="COLLECTION §1", key=["Monday"])], "On Mondays.", ["collection day North"],
                          ["Service Area North"]),
                     turn("CONTEXTUAL_FOLLOWUP", "And in the south?", ["COLLECTION §1"],
                          [C(text="In Service Area South household bins are collected on Wednesdays.",
                             citation="COLLECTION §1", key=["Wednesday"])], "On Wednesdays.",
                          ["collection day South"], ["Service Area South"])]),
    SESSION("VS05", [turn("SIMPLE", "How much does a new water connection cost?", ["CONNECT §1"], [CONNECTION],
                          "450 euros.", ["connection price"], ["new connection"]),
                     turn("CONTEXTUAL_FOLLOWUP", "How long does it take?", ["CONNECT §1"],
                          [C(text="New connections take 6 weeks to complete.", citation="CONNECT §1", key=["6 weeks"])],
                          "6 weeks.", ["connection duration"], ["new connection"])]),
    # ENTITY_CORRECTION
    SESSION("VS06", [turn("MULTI_HOP", "When are bins collected in Oakridge?", ["DISTRICTS §1", "COLLECTION §1"],
                          [C(text="Bins in Oakridge (Service Area North) are collected on Mondays.",
                             citation="COLLECTION §1", key=["Monday"])], "On Mondays.", ["collection day Oakridge"],
                          ["Oakridge"]),
                     turn("ENTITY_CORRECTION", "Sorry, I meant Fernhill, not Oakridge.", ["DISTRICTS §1",
                                                                                         "COLLECTION §1"],
                          [C(text="Bins in Fernhill (Service Area South) are collected on Wednesdays.",
                             citation="COLLECTION §1", key=["Wednesday"])], "On Wednesdays.",
                          ["collection day Fernhill"], ["Fernhill"])]),
    SESSION("VS07", [turn("MULTI_CONSTRAINT", "How long does a decision take for pensioners?", ["RELIEF-PENSIONER §2"],
                          [PENSIONER_TIME], "Within 20 working days.", ["decision time"], [], ["pensioners"]),
                     turn("ENTITY_CORRECTION", "Sorry, I meant households on income support, not pensioners.",
                          ["RELIEF-HARDSHIP §2"], [HARDSHIP_TIME], "Within 15 working days.", ["decision time"], [],
                          ["income support"], forbidden=[r"\b20 working days"])]),
    SESSION("VS08", [turn("SIMPLE", "What does a meter test cost?", ["METER §2"], [METER_TEST], "25 euros.",
                          ["meter test price"], ["meter test"]),
                     turn("ENTITY_CORRECTION", "Sorry, I meant a new water connection, not a meter test.",
                          ["CONNECT §1"], [CONNECTION], "450 euros.", ["connection price"], ["new connection"])]),
    SESSION("VS09", [turn("MULTI_CONSTRAINT", "Which form do pensioners use?", ["RELIEF-PENSIONER §2"],
                          [C(text="Pensioners apply with Form UT-150.", citation="RELIEF-PENSIONER §2",
                             key=["UT-150"])], "Form UT-150.", ["pensioner form"], [], ["pensioners"]),
                     turn("ENTITY_CORRECTION", "I meant the form for households on income support instead.",
                          ["RELIEF-HARDSHIP §2"],
                          [C(text="Households apply with Form UT-120.", citation="RELIEF-HARDSHIP §2", key=["UT-120"])],
                          "Form UT-120.", ["income support form"], [], ["income support"])]),
    SESSION("VS10", [turn("MULTI_HOP", "When are bins collected in Kestrel Point?", ["DISTRICTS §1", "COLLECTION §1"],
                          [C(text="Bins in Kestrel Point (Service Area East) are collected on Fridays.",
                             citation="COLLECTION §1", key=["Friday"])], "On Fridays.", ["collection day Kestrel"],
                          ["Kestrel Point"]),
                     turn("ENTITY_CORRECTION", "I meant garden waste, not household bins.", ["COLLECTION §2"],
                          [C(text="Garden waste is collected every second week from April to October.",
                             citation="COLLECTION §2", key=["second week"])], "Every second week, April to October.",
                          ["garden waste collection"], ["garden waste"])]),
    # REPEATED_QUERY
    SESSION("VS11", [turn("SIMPLE", "How much is a payment reminder?", ["LATE §1"], [REMINDER], "5 euros.",
                          ["reminder fee"], ["payment reminder"]),
                     turn("REPEATED_QUERY", "Sorry, how much was the reminder fee again?", ["LATE §1"], [REMINDER],
                          "5 euros.", ["reminder fee"], ["payment reminder"])]),
    SESSION("VS12", [turn("EXACT_TERM", "When is the UCC open?", ["UCC §2"], [UCC_HOURS], "Monday to Friday, 8:00-16:00.",
                          ["UCC hours"], ["UCC"]),
                     turn("REPEATED_QUERY", "What were the UCC opening hours again?", ["UCC §2"], [UCC_HOURS],
                          "Monday to Friday, 8:00-16:00.", ["UCC hours"], ["UCC"])]),
    SESSION("VS13", [turn("SIMPLE", "How long does a new connection take?", ["CONNECT §1"],
                          [C(text="New connections take 6 weeks.", citation="CONNECT §1", key=["6 weeks"])], "6 weeks.",
                          ["connection duration"], ["new connection"]),
                     turn("REPEATED_QUERY", "So how many weeks does a new connection take?", ["CONNECT §1"],
                          [C(text="New connections take 6 weeks.", citation="CONNECT §1", key=["6 weeks"])], "6 weeks.",
                          ["connection duration"], ["new connection"])]),
    SESSION("VS14", [turn("EXACT_TERM", "Which form is used for meter disputes?", ["METER §1"],
                          [C(text="Meter disputes use Form UT-310.", citation="METER §1", key=["UT-310"])],
                          "Form UT-310.", ["meter dispute form"], ["meter dispute"]),
                     turn("REPEATED_QUERY", "Which form was it for disputing a meter reading?", ["METER §1"],
                          [C(text="Meter disputes use Form UT-310.", citation="METER §1", key=["UT-310"])],
                          "Form UT-310.", ["meter dispute form"], ["meter dispute"])]),
]

STREAMING = [
    S("VSC1", "STREAMING_CORRECTION", "How long does a decision take for pensioners?", ["RELIEF-PENSIONER §2"],
      [PENSIONER_TIME], "Within 20 working days.", ["decision time"], [], ["pensioners"],
      forbidden=[r"\b15 working days"],
      stream=["How long does a decision take", "for income support", (1, "for pensioners")]),
    S("VSC2", "STREAMING_CORRECTION", "What does a meter test cost?", ["METER §2"], [METER_TEST], "25 euros.",
      ["meter test price"], ["meter test"], forbidden=[r"connection costs 450"],
      stream=["What does a", "new connection cost", (1, "meter test cost")]),
    S("VSC3", "STREAMING_CORRECTION", "When are bins collected in Fernhill?", ["DISTRICTS §1", "COLLECTION §1"],
      [C(text="Bins in Fernhill are collected on Wednesdays.", citation="COLLECTION §1", key=["Wednesday"])],
      "On Wednesdays.", ["collection day Fernhill"], ["Fernhill"],
      stream=["When are bins collected", "in Oakridge", (1, "in Fernhill")]),
    S("VSC4", "STREAMING_CORRECTION", "How far ahead must bulky waste be booked?", ["BULKY §1"],
      [C(text="Bulky waste pickups are booked at least 5 days in advance.", citation="BULKY §1", key=["5 days"])],
      "At least 5 days in advance.", ["bulky waste lead time"], ["bulky waste"],
      stream=["How far ahead must", "garden waste", (1, "bulky waste"), "be booked?"]),
]


def main() -> None:
    gap = lexical_gap_fn()
    rows = []
    for d in TEST + STREAMING + [t for s in SESSIONS for t in s]:
        s = EvalSample(**d)
        diff, feats = difficulty(s, gap(d))
        rows.append(s.model_copy(update={"difficulty": diff, "difficulty_features": feats}))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "test.jsonl"
    p.write_text(dumps(rows))
    man = manifest({"test": p}, {"construction": "experiments/datasets/build_dataset_v2.py",
                                 "labels": "implementer-written from the fixture corpus text; no independent annotation",
                                 "test_corpus": "tests/fixtures/corpus_eval_utility (new in Phase 11; written and frozen "
                                                "before any Phase 11 system change; never used in development)",
                                 "frozen": "2026-10-04, before Phase 11 fixes", "reportable": False})
    man["version"] = "streamrag_eval_v2"
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
