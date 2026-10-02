import pytest

from streamrag.controller.lexicon import Lexicon
from streamrag.controller.query_builder import QueryBuilder

from conftest import REPO

QB = QueryBuilder(Lexicon.load(REPO / "configs" / "controller_lexicon.yaml"))


@pytest.mark.parametrize("transcript,expected", [
    ("so um I wanted to ask how high the wicks should be trimmed", "how high the wicks should be trimmed"),
    ("uh hi quick question about the tool shed what do workers do", "the tool shed what do workers do"),
    ("can you tell me the fog signal interval during a storm", "the fog signal interval during a storm"),
    ("I need information about", ""),
    ("um okay", ""),
])
def test_filler_and_preamble_removed(transcript, expected):
    assert QB.build(transcript).text == expected


def test_negation_numbers_and_constraints_preserved():
    q = QB.build("are ladders allowed to stay in the orchard overnight or not")
    assert q.text.endswith("or not") and "overnight" in q.text
    q = QB.build("can a picker fill more than 40 crates without a supervisor")
    assert "40" in q.text and "without" in q.text and "supervisor" in q.text


def test_query_is_traceable_to_transcript():
    t = "so um I wanted to ask how high the wicks should be trimmed in the"
    q = QB.build(t)
    assert [t[s:e] for s, e in q.spans] == q.text.split()
    assert "um" in q.removed and "the" in q.removed            # trailing dangling "in the" stripped
