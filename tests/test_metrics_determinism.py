import pytest

from rag_abstain.metrics.determinism import determinism_metrics
from rag_abstain.models import Record, SystemResponse


def rec(qid: str, repeat: int, answer: str, outcome: str, category: str | None = None) -> Record:
    return Record(
        qid=qid,
        repeat=repeat,
        question=qid,
        expected="answer",
        category=category,
        response=SystemResponse(answer=answer, error="boom" if outcome == "error" else None),
        outcome=outcome,  # type: ignore[arg-type]
    )


def test_identical_text_differs_same_outcome_and_flip() -> None:
    records = [
        rec("A", 1, "Same.", "answered", "c1"),
        rec("A", 2, "Same.", "answered", "c1"),
        rec("B", 1, "Brasilia.", "answered", "c1"),
        rec("B", 2, "It is Brasilia.", "answered", "c1"),
        rec("C", 2, "Some answer.", "answered", "c2"),
        rec("C", 1, "The documents do not contain this information.", "declined", "c2"),
    ]
    result = determinism_metrics(records)
    assert result["n_compared"] == 3
    assert result["identical_answer_rate"] == pytest.approx(1 / 3)
    assert result["outcome_agreement_rate"] == pytest.approx(2 / 3)
    assert result["outcome_flips"] == [{"qid": "C", "outcomes": ["declined", "answered"]}]
    assert result["by_category"]["c1"]["outcome_agreement_rate"] == 1.0
    assert result["by_category"]["c2"]["identical_answer_rate"] == 0.0


def test_text_is_compared_byte_for_byte() -> None:
    records = [rec("A", 1, "Paris.", "answered"), rec("A", 2, "Paris. ", "answered")]
    assert determinism_metrics(records)["identical_answer_rate"] == 0.0


def test_three_repeats_must_all_agree() -> None:
    records = [rec("A", r, "x", o) for r, o in ((1, "answered"), (2, "answered"), (3, "hedged"))]
    result = determinism_metrics(records)
    assert result["outcome_flips"] == [{"qid": "A", "outcomes": ["answered", "answered", "hedged"]}]


def test_questions_with_an_error_repeat_are_listed_not_compared() -> None:
    records = [
        rec("A", 1, "x", "answered"),
        rec("A", 2, "", "error"),
        rec("B", 1, "y", "answered"),
        rec("B", 2, "y", "answered"),
    ]
    result = determinism_metrics(records)
    assert result["error_qids"] == ["A"]
    assert result["n_compared"] == 1 and result["identical_answer_rate"] == 1.0


def test_single_repeat_is_null_with_a_note() -> None:
    result = determinism_metrics([rec("A", 1, "x", "answered")])
    assert result["n_compared"] == 0
    assert result["identical_answer_rate"] is None
    assert result["outcome_agreement_rate"] is None
    assert "identical_answer_rate" in result["notes"]
    assert result["outcome_flips"] == []
