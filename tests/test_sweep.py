import builtins
from pathlib import Path
from typing import Any

import pytest

from rag_abstain.models import Record, SystemResponse
from rag_abstain.sweep import (
    IN_SAMPLE_NOTE,
    SweepError,
    gate,
    margin,
    parse_grid,
    plot_sweep,
    render_markdown,
    sweep_records,
)


def rec(qid: str, expected: str, nearest: float | None, outcome: str = "answered") -> Record:
    return Record(
        qid=qid,
        question=qid,
        expected=expected,  # type: ignore[arg-type]
        response=SystemResponse(
            answer="x",
            distances=[] if nearest is None else [nearest, nearest + 0.1],
            error="boom" if outcome == "error" else None,
        ),
        outcome=outcome,  # type: ignore[arg-type]
        decline_stage="model" if outcome == "declined" else None,
    )


# Answerable nearest distances 0.2-0.4, unanswerable 0.5-0.7: margin 0.1, separable.
# Every question was answered with the gate off, so the gate alone decides.
SEPARABLE = [
    rec("A1", "answer", 0.2),
    rec("A2", "answer", 0.3),
    rec("A3", "answer", 0.4),
    rec("D1", "decline", 0.5),
    rec("D2", "decline", 0.6),
    rec("D3", "decline", 0.7),
]


def test_parse_grid_is_inclusive_and_rounded() -> None:
    assert parse_grid("0.3:0.7:0.1") == [0.3, 0.4, 0.5, 0.6, 0.7]
    grid = parse_grid("0.20:1.20:0.05")
    assert len(grid) == 21 and grid[0] == 0.2 and grid[-1] == 1.2 and grid[3] == 0.35


@pytest.mark.parametrize("spec", ["0.2:1.2", "a:b:c", "0.5:0.2:0.1", "0.2:1.2:0"])
def test_bad_grids(spec: str) -> None:
    with pytest.raises(SweepError):
        parse_grid(spec)


def test_known_margin() -> None:
    assert margin(SEPARABLE) == {
        "max_answer": 0.4,
        "min_decline": 0.5,
        "margin": 0.1,
        "separable": True,
    }
    overlapping = margin([rec("A", "answer", 0.6), rec("D", "decline", 0.5)])
    assert overlapping["separable"] is False and overlapping["margin"] == -0.1
    assert margin([rec("A", "answer", 0.6)])["separable"] is None


def test_gate_is_strictly_greater_than() -> None:
    gated = gate(SEPARABLE, 0.4)
    assert [r.outcome for r in gated] == ["answered"] * 3 + ["declined"] * 3
    assert all(r.decline_stage == "gate" for r in gated[3:])


def test_known_best_threshold() -> None:
    result = sweep_records(SEPARABLE, parse_grid("0.3:0.7:0.1"), "run")
    rows = {row["threshold"]: row for row in result["rows"]}
    assert rows[0.3]["false_decline_rate"] == pytest.approx(1 / 3)
    assert rows[0.4] == {
        "threshold": 0.4,
        "unsupported_answer_rate": 0.0,
        "false_decline_rate": 0.0,
        "balanced_accuracy": 1.0,
        "gate_declines": 3,
    }
    assert rows[0.5]["unsupported_answer_rate"] == pytest.approx(1 / 3)
    assert rows[0.7]["gate_declines"] == 0
    assert result["best"]["threshold"] == 0.4
    assert result["in_sample"] is True and result["note"] == IN_SAMPLE_NOTE


def test_recorded_model_declines_and_errors() -> None:
    records = [
        rec("A", "answer", 0.2),
        rec("D", "decline", 0.3, outcome="declined"),  # declined by the model below the gate
        rec("E", "decline", 0.9, outcome="error"),
    ]
    result = sweep_records(records, [0.5], "run")
    assert result["n"] == 2 and result["errors_excluded"] == 1
    assert result["rows"][0]["unsupported_answer_rate"] == 0.0
    assert result["rows"][0]["gate_declines"] == 0


def test_gate_on_run_is_refused() -> None:
    gated = SEPARABLE[0].model_copy(update={"outcome": "declined", "decline_stage": "gate"})
    with pytest.raises(SweepError, match="gate off"):
        sweep_records([gated, *SEPARABLE[1:]], [0.5], "run")


def test_missing_distances_are_refused() -> None:
    with pytest.raises(SweepError, match="no distances"):
        sweep_records([rec("A", "answer", None)], [0.5], "run")
    with pytest.raises(SweepError, match="no non-error"):
        sweep_records([rec("A", "answer", 0.2, outcome="error")], [0.5], "run")


def test_markdown_is_labelled_in_sample() -> None:
    text = render_markdown(sweep_records(SEPARABLE, parse_grid("0.3:0.7:0.1"), "my-run"))
    assert text.startswith("# Threshold sweep (in-sample): my-run")
    assert IN_SAMPLE_NOTE in text
    assert "Separable: any threshold t with 0.400 <= t < 0.500" in text
    assert "1.000 at t = 0.40 (in-sample)" in text
    assert "| 0.40 | 0.000 | 0.000 | 1.000 | 3 |" in text


def test_plot_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "sweep.png"
    plot_sweep(sweep_records(SEPARABLE, parse_grid("0.3:0.7:0.1"), "run"), path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_without_matplotlib_says_how_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def no_matplotlib(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("matplotlib"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    result = sweep_records(SEPARABLE, [0.5], "run")
    with pytest.raises(SweepError, match=r"rag-abstain\[plot\]"):
        plot_sweep(result, tmp_path / "x.png")
