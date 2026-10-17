import pytest

from rag_abstain.metrics.correctness import (
    correctness_metrics,
    exact_match,
    f1,
    key_term_hit,
    normalise,
    score_answer,
)
from rag_abstain.models import Question, Record, SystemResponse

# --- SQuAD normalisation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The Cat", "cat"),  # case and article
        ("a cat, an owl and the dog.", "cat owl and dog"),  # all three articles, punctuation
        ("theory", "theory"),  # article only as a whole word
        ("  March   14,\n2023 ", "march 14 2023"),  # whitespace
        ("tf-idf", "tfidf"),  # punctuation is removed, not replaced by a space
        ("It's A.I.", "its ai"),
        ("", ""),
    ],
)
def test_normalise(text: str, expected: str) -> None:
    assert normalise(text) == expected


# --- exact match and F1 ----------------------------------------------------------------------


def test_exact_match_after_normalisation() -> None:
    assert exact_match("The Cocke-Younger-Kasami.", ["cockeyoungerkasami"]) == 1.0
    assert exact_match("Cocke", ["Cocke Younger Kasami"]) == 0.0


def test_exact_match_best_over_references() -> None:
    assert exact_match("Paris", ["London", "paris"]) == 1.0


def test_exact_match_without_references_is_none() -> None:
    assert exact_match("anything", []) is None


def test_f1_partial_overlap() -> None:
    # [cat, sat] vs [cat, sat, down]: precision 1, recall 2/3 -> F1 0.8
    assert f1("The cat sat", ["a cat sat down"]) == pytest.approx(0.8)


def test_f1_counts_repeated_tokens_once_each() -> None:
    # [x, x, y] vs [x, y]: common 2, precision 2/3, recall 1 -> 0.8
    assert f1("x x y", ["x y"]) == pytest.approx(0.8)


def test_f1_best_over_references() -> None:
    assert f1("cat sat", ["dog", "cat sat"]) == 1.0


def test_f1_no_overlap_is_zero() -> None:
    assert f1("dog", ["cat"]) == 0.0


def test_f1_empty_after_normalisation() -> None:
    assert f1("The.", ["a"]) == 1.0
    assert f1("The.", ["cat"]) == 0.0


def test_f1_without_references_is_none() -> None:
    assert f1("anything", []) is None


# --- key-term hit ----------------------------------------------------------------------------


def test_key_term_hit_every_group_needed() -> None:
    groups = [["Cocke"], ["Younger"], ["Kasami"]]
    assert key_term_hit("Named after Cocke, Younger and Kasami.", groups) == 1.0
    assert key_term_hit("Named after Cocke and Kasami.", groups) == 0.0


def test_key_term_hit_any_alternative_in_a_group() -> None:
    groups = [["March 14", "14 March"], ["2023"]]
    assert key_term_hit("Released on 14 March 2023.", groups) == 1.0


def test_key_term_hit_matches_whole_tokens() -> None:
    assert key_term_hit("a fragment of text", [["rag"]]) == 0.0
    assert key_term_hit("RAG systems", [["rag"]]) == 1.0


def test_key_term_hit_normalises_both_sides() -> None:
    assert key_term_hit("TF-IDF weighting", [["tf-idf"]]) == 1.0
    assert key_term_hit("TF-IDF weighting", [["TFIDF"]]) == 1.0


def test_key_term_hit_without_groups_is_none() -> None:
    assert key_term_hit("anything", []) is None


def test_key_term_alternative_that_normalises_to_nothing_never_matches() -> None:
    assert key_term_hit("the cat", [["the"]]) == 0.0


# --- per-record scores and aggregation -------------------------------------------------------


def test_score_answer() -> None:
    q = Question(
        qid="A",
        question="q",
        expected="answer",
        references=["Cocke Younger Kasami"],
        key_terms=[["Cocke"], ["Kasami"]],
    )
    scores = score_answer(q, "Cocke Younger Kasami")
    assert scores == {"exact_match": 1.0, "f1": 1.0, "key_term_hit": 1.0}
    bare = Question(qid="B", question="q", expected="answer")
    assert score_answer(bare, "x") == {"exact_match": None, "f1": None, "key_term_hit": None}


def rec(
    qid: str,
    outcome: str,
    scores: dict[str, float | None],
    expected: str = "answer",
    category: str | None = "answerable",
    repeat: int = 1,
) -> Record:
    return Record(
        qid=qid,
        repeat=repeat,
        question="q",
        expected=expected,
        category=category,
        response=SystemResponse(),
        outcome=outcome,
        scores=scores,
    )


def s(em: float | None, f: float | None, kt: float | None) -> dict[str, float | None]:
    return {"exact_match": em, "f1": f, "key_term_hit": kt}


RECORDS = [
    rec("A1", "answered", s(1.0, 1.0, 1.0)),
    rec("A2", "answered", s(0.0, 0.5, 0.0)),
    rec("A3", "hedged", s(1.0, 1.0, 1.0)),  # counts 0 in all_answerable, excluded otherwise
    rec("A4", "declined", s(0.0, 0.2, 0.0)),  # counts 0 in all_answerable, excluded otherwise
    rec("A5", "answered", s(None, None, 1.0)),  # key terms only
    rec("A6", "error", s(1.0, 1.0, 1.0)),  # excluded everywhere
    rec("D1", "answered", s(None, None, None), expected="decline", category="ooc"),
]


def test_correctness_views_known_values() -> None:
    m = correctness_metrics(RECORDS)
    only = m["answered_only"]
    assert only["n"] == 3
    assert only["exact_match"] == pytest.approx(0.5)  # A1, A2
    assert only["f1"] == pytest.approx(0.75)
    assert only["key_term_hit"] == pytest.approx(2 / 3)  # A1, A2, A5
    assert only["scored"] == {"exact_match": 2, "f1": 2, "key_term_hit": 3}

    everything = m["all_answerable"]
    assert everything["n"] == 5
    assert everything["exact_match"] == pytest.approx(1 / 4)  # A1..A4
    assert everything["f1"] == pytest.approx(1.5 / 4)
    assert everything["key_term_hit"] == pytest.approx(2 / 5)
    assert m["notes"] == {}


def test_correctness_category_without_answerable_questions_is_null() -> None:
    ooc = correctness_metrics(RECORDS)["by_category"]["ooc"]
    assert ooc["answered_only"]["n"] == 0
    for view in ("answered_only", "all_answerable"):
        for name in ("exact_match", "f1", "key_term_hit"):
            assert ooc[view][name] is None
            assert f"{view}.{name}" in ooc["notes"]


def test_correctness_all_error_run() -> None:
    m = correctness_metrics([rec("A", "error", s(1.0, 1.0, 1.0))])
    assert m["all_answerable"]["n"] == 0
    assert m["all_answerable"]["f1"] is None
    assert "all_answerable.f1" in m["notes"]


def test_correctness_only_repeat_one_is_used() -> None:
    later = [rec("A1", "declined", s(0.0, 0.0, 0.0), repeat=2)]
    assert correctness_metrics(RECORDS + later) == correctness_metrics(RECORDS)
