import pytest

from rag_abstain.metrics.abstention import abstention_metrics
from rag_abstain.models import Record, SystemResponse

RATES = (
    "unsupported_answer_rate",
    "false_decline_rate",
    "hedge_rate",
    "decline_precision",
    "decline_recall",
    "decline_f1",
    "answer_recall",
    "balanced_accuracy",
)


def rec(
    qid: str,
    expected: str,
    outcome: str,
    stage: str | None = None,
    category: str | None = None,
    repeat: int = 1,
) -> Record:
    return Record(
        qid=qid,
        repeat=repeat,
        question=f"question {qid}",
        expected=expected,
        category=category,
        response=SystemResponse(error="boom" if outcome == "error" else None),
        outcome=outcome,
        decline_stage=stage,
    )


# 4 non-error decline-expected: 2 declined (tp), 1 answered + 1 hedged (unsupported); 1 error.
# 4 answer-expected: 2 answered, 1 hedged, 1 false decline at the gate.
MIXED = [
    rec("D1", "decline", "declined", "gate", "out_of_corpus"),
    rec("D2", "decline", "declined", "model", "out_of_corpus"),
    rec("D3", "decline", "answered", None, "out_of_corpus"),
    rec("D4", "decline", "hedged", None, "out_of_corpus"),
    rec("D5", "decline", "error", None, "out_of_corpus"),
    rec("A1", "answer", "answered", None, "answerable"),
    rec("A2", "answer", "answered", None, "answerable"),
    rec("A3", "answer", "declined", "gate", "answerable"),
    rec("A4", "answer", "hedged", None, "answerable"),
]


def test_known_values_overall() -> None:
    m = abstention_metrics(MIXED)
    assert m["n"] == 8
    assert m["n_expected_answer"] == 4 and m["n_expected_decline"] == 4
    assert m["errors"] == 1 and m["error_qids"] == ["D5"]
    assert m["outcomes"] == {"answered": 3, "declined": 3, "hedged": 2}
    assert m["unsupported_answer_rate"] == pytest.approx(2 / 4)
    assert m["false_decline_rate"] == pytest.approx(1 / 4)
    assert m["hedge_rate"] == pytest.approx(2 / 8)
    assert m["decline_precision"] == pytest.approx(2 / 3)
    assert m["decline_recall"] == pytest.approx(2 / 4)
    assert m["decline_f1"] == pytest.approx(4 / 7)
    assert m["answer_recall"] == pytest.approx(3 / 4)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 0.75) / 2)
    assert m["declines_by_stage"] == {"gate": 2, "model": 1}
    assert m["notes"] == {}


def test_hedge_counts_as_unsupported_not_as_decline() -> None:
    m = abstention_metrics([rec("D", "decline", "hedged"), rec("A", "answer", "answered")])
    assert m["unsupported_answer_rate"] == 1.0
    assert m["decline_recall"] == 0.0


def test_by_category_with_one_expected_class_gives_null_with_notes() -> None:
    cats = abstention_metrics(MIXED)["by_category"]
    assert sorted(cats) == ["answerable", "out_of_corpus"]
    answerable = cats["answerable"]
    assert answerable["false_decline_rate"] == pytest.approx(1 / 4)
    assert answerable["unsupported_answer_rate"] is None
    assert answerable["decline_recall"] is None
    assert answerable["balanced_accuracy"] is None
    for key in ("unsupported_answer_rate", "decline_recall", "balanced_accuracy"):
        assert key in answerable["notes"]
    ooc = cats["out_of_corpus"]
    assert ooc["unsupported_answer_rate"] == pytest.approx(2 / 4)
    assert ooc["false_decline_rate"] is None
    assert ooc["errors"] == 1


def test_records_without_category_are_grouped() -> None:
    m = abstention_metrics([rec("A", "answer", "answered")])
    assert list(m["by_category"]) == ["uncategorised"]


def test_category_with_only_errors() -> None:
    records = [*MIXED, rec("E1", "answer", "error", category="broken")]
    broken = abstention_metrics(records)["by_category"]["broken"]
    assert broken["n"] == 0 and broken["errors"] == 1
    for key in RATES:
        assert broken[key] is None
        assert key in broken["notes"]


def test_all_error_run() -> None:
    m = abstention_metrics([rec("A", "answer", "error"), rec("D", "decline", "error")])
    assert m["n"] == 0
    assert m["errors"] == 2
    assert all(m[key] is None for key in RATES)
    assert set(RATES) <= set(m["notes"])


def test_empty_input() -> None:
    m = abstention_metrics([])
    assert m["n"] == 0 and m["errors"] == 0 and m["by_category"] == {}
    assert all(m[key] is None for key in RATES)


def test_no_declines_made_precision_is_null_but_f1_is_zero() -> None:
    m = abstention_metrics([rec("D", "decline", "answered"), rec("A", "answer", "answered")])
    assert m["decline_precision"] is None
    assert "decline_precision" in m["notes"]
    assert m["decline_f1"] == 0.0
    assert m["decline_recall"] == 0.0


def test_perfect_system() -> None:
    m = abstention_metrics(
        [rec("D", "decline", "declined", "gate"), rec("A", "answer", "answered")]
    )
    assert m["unsupported_answer_rate"] == 0.0
    assert m["false_decline_rate"] == 0.0
    assert m["decline_precision"] == m["decline_recall"] == m["decline_f1"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_only_repeat_one_is_used() -> None:
    later = [rec(r.qid, r.expected, "answered", None, r.category, repeat=2) for r in MIXED]
    assert abstention_metrics(MIXED + later) == abstention_metrics(MIXED)
