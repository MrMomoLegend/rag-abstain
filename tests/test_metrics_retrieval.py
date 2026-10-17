import pytest

from rag_abstain.metrics.retrieval import (
    SCORE_NAMES,
    basename,
    hit_at_k,
    mrr,
    recall_at_k,
    retrieval_metrics,
    score_retrieval,
    unique_sources,
)
from rag_abstain.models import Record, SystemResponse


def test_basename_handles_both_slashes_and_case() -> None:
    assert basename("docs/CYK Algorithm.txt") == "cyk algorithm.txt"
    assert basename("C:\\corpus\\GPT-4.txt") == "gpt-4.txt"
    assert basename("plain.txt") == "plain.txt"


def test_duplicates_collapse_in_rank_order() -> None:
    sources = ["a/X.txt", "X.txt", "b/y.txt", "x.TXT", "z.txt"]
    assert unique_sources(sources) == ["x.txt", "y.txt", "z.txt"]
    # y.txt is rank 2 once duplicates of x are removed, not rank 3.
    assert mrr(sources, ["Y.txt"]) == 0.5
    assert hit_at_k(sources, ["y.txt"], 1) == 0.0
    assert hit_at_k(sources, ["y.txt"], 2) == 1.0


def test_k_larger_than_the_list() -> None:
    assert hit_at_k(["a.txt", "b.txt"], ["b.txt"], 10) == 1.0
    assert recall_at_k(["a.txt", "b.txt"], ["b.txt", "c.txt"], 10) == 0.5


def test_recall_counts_distinct_gold_sources() -> None:
    sources = ["a.txt", "b.txt", "c.txt"]
    assert recall_at_k(sources, ["a.txt", "dir/A.txt", "c.txt"], 1) == 0.5
    assert recall_at_k(sources, ["a.txt", "c.txt"], 3) == 1.0


def test_no_match_and_no_gold() -> None:
    assert mrr(["a.txt"], ["b.txt"]) == 0.0
    assert hit_at_k(["a.txt"], [], 1) is None
    assert recall_at_k(["a.txt"], [""], 1) is None
    assert mrr(["a.txt"], []) is None


def test_score_retrieval_skips_unscorable_records() -> None:
    assert score_retrieval([], ["a.txt"]) == {}
    assert score_retrieval(["a.txt"], []) == {}
    scores = score_retrieval(["b.txt", "a.txt"], ["a.txt"])
    assert set(scores) == set(SCORE_NAMES)
    assert scores["hit_at_1"] == 0.0 and scores["hit_at_3"] == 1.0 and scores["mrr"] == 0.5


def rec(qid: str, sources: list[str], gold: list[str], **kwargs: object) -> Record:
    outcome = kwargs.pop("outcome", "answered")
    return Record(
        qid=qid,
        question=qid,
        expected="answer",
        response=SystemResponse(sources=sources),
        outcome=outcome,  # type: ignore[arg-type]
        scores=score_retrieval(sources, gold) if outcome != "error" else {},
        **kwargs,  # type: ignore[arg-type]
    )


def test_aggregate_over_repeat_one_by_category() -> None:
    records = [
        rec("A", ["a.txt"], ["a.txt"], category="c1"),
        rec(
            "B",
            ["x.txt", "b.txt"],
            ["b.txt"],
            category="c1",
            decline_stage="gate",
            outcome="declined",
        ),
        rec("C", ["x.txt"], ["c.txt"], category="c2"),
        rec("D", [], ["d.txt"], category="c2"),  # no sources returned: unscored
        rec("E", ["e.txt"], [], category="c2"),  # no gold: unscored
        rec("F", ["f.txt"], ["f.txt"], outcome="error"),
        rec("A", ["x.txt"], ["a.txt"], category="c1", repeat=2),  # ignored
    ]
    result = retrieval_metrics(records)
    assert result["n_scored"] == 3 and result["n_unscored"] == 2
    assert result["hit_at_1"] == pytest.approx(1 / 3)
    assert result["hit_at_3"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert result["by_category"]["c1"]["mrr"] == pytest.approx(0.75)
    assert result["by_category"]["c2"]["n_scored"] == 1
    assert result["by_category"]["uncategorised"]["mrr"] is None


def test_nothing_scorable_is_null_with_a_note() -> None:
    result = retrieval_metrics([rec("A", [], ["a.txt"])])
    assert all(result[name] is None for name in SCORE_NAMES)
    assert set(result["notes"]) == set(SCORE_NAMES)
    assert retrieval_metrics([])["mrr"] is None
