from pathlib import Path

import pytest

from rag_abstain.adapters import load_system
from rag_abstain.compare import compare_runs, render_markdown, truncate
from rag_abstain.runner import run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DATASET = EXAMPLES / "toy_questions.csv"


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("runs")
    gate_on = run(load_system(EXAMPLES / "toy_system.yaml"), DATASET, out_dir=out).run_dir
    gate_off = run(load_system(EXAMPLES / "toy_system_gate_off.yaml"), DATASET, out_dir=out).run_dir
    return gate_on, gate_off


def test_changed_outcomes_include_stage_changes(runs: tuple[Path, Path]) -> None:
    result = compare_runs(*runs)
    changed = {c["qid"]: c for c in result["changed"]}
    assert set(changed) == {"T08", "T09", "T10"}
    assert changed["T08"]["a"]["outcome"] == "declined"
    assert changed["T08"]["b"]["answer"] == "The capital of Brazil is Brasilia."
    assert (changed["T09"]["a"]["decline_stage"], changed["T09"]["b"]["decline_stage"]) == (
        "gate",
        "model",
    )
    assert result["only_in_a"] == [] and result["only_in_b"] == [] and result["notes"] == []


def test_metric_deltas_are_b_minus_a(runs: tuple[Path, Path]) -> None:
    metrics = {m["metric"]: m for m in compare_runs(*runs)["metrics"]}
    unsupported = metrics["unsupported answer rate"]
    assert (unsupported["a"], unsupported["b"]) == (0.25, 0.5)
    assert unsupported["delta"] == pytest.approx(0.25)
    assert metrics["false decline rate"]["delta"] == pytest.approx(-1 / 6)
    assert metrics["identical answer rate"]["delta"] is None  # one repeat: n/a on both sides
    assert metrics["prompt tokens"]["delta"] == 2700


def test_markdown(runs: tuple[Path, Path]) -> None:
    text = render_markdown(compare_runs(*runs))
    assert "| unsupported answer rate | 0.250 | 0.500 | +0.250 |" in text
    assert "| identical answer rate | n/a | n/a | n/a |" in text
    assert "## Changed outcomes (repeat 1): 3" in text
    assert "| T09 | decline | declined (gate) | declined (model) |" in text


def test_different_datasets_and_missing_questions(runs: tuple[Path, Path], tmp_path: Path) -> None:
    short = tmp_path / "short.csv"
    short.write_text(
        "\n".join(DATASET.read_text(encoding="utf-8").splitlines()[:4]) + "\n", encoding="utf-8"
    )
    small = run(load_system(EXAMPLES / "toy_system.yaml"), short, out_dir=tmp_path).run_dir
    result = compare_runs(runs[0], small)
    assert result["notes"] == [
        "the runs used different datasets; deltas compare different question sets"
    ]
    assert result["only_in_a"] == [f"T{i:02d}" for i in range(4, 11)]
    assert "Only in A: T04" in render_markdown(result)


def test_truncate_keeps_200_characters() -> None:
    assert truncate("x" * 200) == "x" * 200
    cut = truncate("x" * 500)
    assert len(cut) == 200 and cut.endswith("...")
