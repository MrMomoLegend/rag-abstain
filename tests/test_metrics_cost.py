import pytest

from rag_abstain.metrics.cost import cost_metrics, percentile
from rag_abstain.models import Record, SystemResponse


def rec(
    outcome: str,
    latency: float | None,
    stage: str | None = None,
    tokens: tuple[int | None, int | None] = (None, None),
    repeat: int = 1,
    category: str | None = None,
) -> Record:
    return Record(
        qid="Q",
        repeat=repeat,
        question="q",
        expected="answer",
        category=category,
        response=SystemResponse(
            latency_s=latency,
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            error="boom" if outcome == "error" else None,
        ),
        outcome=outcome,  # type: ignore[arg-type]
        decline_stage=stage,  # type: ignore[arg-type]
    )


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 95) == 19.0
    assert percentile(values, 100) == 20.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 95) is None


def test_latency_split_by_stage_and_tokens_over_all_repeats() -> None:
    records = [
        rec("declined", 0.01, "gate", category="out"),
        rec("declined", 0.03, "gate", repeat=2, category="out"),
        rec("declined", 1.0, "model", (900, 10)),
        rec("answered", 2.0, None, (900, 30)),
        rec("hedged", 4.0, None, (800, 50), repeat=2),
        rec("error", 60.0),
    ]
    result = cost_metrics(records)
    latency = result["latency"]
    assert latency["all"] == {"n": 5, "median_s": 1.0, "p95_s": 4.0}
    assert latency["gate"] == {"n": 2, "median_s": pytest.approx(0.02), "p95_s": 0.03}
    assert latency["model"]["n"] == 1
    assert latency["no_decline"] == {"n": 2, "median_s": 3.0, "p95_s": 4.0}
    assert result["errors"] == 1 and result["n_records"] == 6
    assert result["prompt_tokens_total"] == 2600
    assert result["completion_tokens_total"] == 90
    assert result["notes"]["prompt_tokens_total"] == "3 record(s) did not report it"
    assert result["by_category"]["out"]["latency"]["gate"]["n"] == 2
    assert result["by_category"]["out"]["prompt_tokens_total"] is None


def test_all_error_run_is_null_not_zero() -> None:
    result = cost_metrics([rec("error", 1.0), rec("error", None)])
    assert result["errors"] == 2
    assert result["latency"]["all"] == {"n": 0, "median_s": None, "p95_s": None}
    assert result["prompt_tokens_total"] is None
    assert "latency.all" in result["notes"]
    assert result["notes"]["prompt_tokens_total"] == "no record reported this count"


def test_empty_run() -> None:
    result = cost_metrics([])
    assert result["n_records"] == 0 and result["by_category"] == {}
