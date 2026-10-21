import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_abstain import __version__
from rag_abstain.cli import EXIT_OK, EXIT_RUN_ERRORS, EXIT_USAGE, app, run_cli
from rag_abstain.report import REPORT_FILE, SUMMARY_FILE

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DATASET = str(EXAMPLES / "toy_questions.csv")
GATE_ON = str(EXAMPLES / "toy_system.yaml")
GATE_OFF = str(EXAMPLES / "toy_system_gate_off.yaml")

runner = CliRunner()


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def only_run_dir(out: Path) -> Path:
    (run_dir,) = out.iterdir()
    return run_dir


def test_run_report_compare_sweep_end_to_end(tmp_path: Path) -> None:
    on_out, off_out = tmp_path / "on", tmp_path / "off"

    result = runner.invoke(
        app,
        ["run", "--system", GATE_ON, "--dataset", DATASET, "--out", str(on_out), "--repeats", "2"],
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "Records: 20 new (0 retried), 0 skipped; errors: 0" in result.stdout
    assert "Unsupported answer rate: 0.250  False decline rate: 0.167" in result.stdout
    gate_on = only_run_dir(on_out)
    assert {p.name for p in gate_on.iterdir()} == {
        "manifest.json",
        "records.jsonl",
        SUMMARY_FILE,
        REPORT_FILE,
    }

    result = runner.invoke(
        app, ["run", "--system", GATE_OFF, "--dataset", DATASET, "--out", str(off_out)]
    )
    assert result.exit_code == EXIT_OK, result.output
    gate_off = only_run_dir(off_out)

    result = runner.invoke(app, ["report", str(gate_on)])
    assert result.exit_code == EXIT_OK, result.output
    assert result.stdout.startswith(f"# Run report: {gate_on.name}")
    assert "Outcome flips: none." in result.stdout

    result = runner.invoke(app, ["report", str(gate_on), "--format", "json"])
    assert json.loads(result.stdout)["determinism"]["identical_answer_rate"] == 1.0

    result = runner.invoke(app, ["compare", str(gate_on), str(gate_off)])
    assert result.exit_code == EXIT_OK, result.output
    assert "## Changed outcomes (repeat 1): 3" in result.stdout

    result = runner.invoke(app, ["compare", str(gate_on), str(gate_off), "--format", "json"])
    assert {c["qid"] for c in json.loads(result.stdout)["changed"]} == {"T08", "T09", "T10"}

    plot = tmp_path / "sweep.png"
    args = ["sweep", str(gate_off), "--grid", "0.30:0.60:0.05"]
    if _has_matplotlib():
        args += ["--plot", str(plot)]
    result = runner.invoke(app, args)
    assert result.exit_code == EXIT_OK, result.output
    assert result.stdout.startswith(f"# Threshold sweep (in-sample): {gate_off.name}")
    assert "| 0.55 | 0.250 | 0.167 | 0.792 | 3 |" in result.stdout  # same as the 0.53 gate
    assert plot.exists() == _has_matplotlib()

    result = runner.invoke(app, ["sweep", str(gate_on)])
    assert result.exit_code == EXIT_USAGE
    assert "the sweep needs a run made with the gate off" in result.output


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def test_run_with_error_records_exits_2_and_resume_retries(tmp_path: Path) -> None:
    config = tmp_path / "raising.yaml"
    config.write_text(
        "name: raising\nadapter: python\ntarget: rag_abstain_cli_raiser:ask\npythonpath: .\n",
        encoding="utf-8",
    )
    (tmp_path / "rag_abstain_cli_raiser.py").write_text(
        "def ask(question):\n    raise RuntimeError('quota exceeded')\n", encoding="utf-8"
    )
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["run", "--system", str(config), "--dataset", DATASET, "--out", str(out), "--limit", "2"],
    )
    assert result.exit_code == EXIT_RUN_ERRORS
    assert "errors: 2" in result.stdout
    run_dir = only_run_dir(out)

    result = runner.invoke(
        app,
        [
            "run",
            "--system",
            str(config),
            "--dataset",
            DATASET,
            "--resume",
            str(run_dir),
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == EXIT_RUN_ERRORS
    assert "Records: 2 new (2 retried), 0 skipped" in result.stdout


def test_validate_dataset(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-dataset", DATASET])
    assert result.exit_code == EXIT_OK
    assert "OK, 10 questions, 0 warning(s)" in result.stdout

    bad = tmp_path / "bad.csv"
    bad.write_text("qid,question,expected\nQ1,Hi?,maybe\n", encoding="utf-8")
    result = runner.invoke(app, ["validate-dataset", str(bad)])
    assert result.exit_code == EXIT_USAGE
    assert "1 error(s)" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--dataset", DATASET],  # missing --system
        ["run", "--system", GATE_ON, "--dataset", DATASET, "--workers", "0"],
        ["validate-dataset", "no/such/dataset.csv"],
        ["report", "no/such/run/dir"],
        ["sweep", "."],  # a directory, but not a run
        ["no-such-command"],
    ],
)
def test_usage_errors_exit_1(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(argv) == EXIT_USAGE
    assert capsys.readouterr().err


def test_bad_system_and_dataset_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / "missing.csv")
    assert run_cli(["run", "--system", GATE_ON, "--dataset", missing, "--out", str(tmp_path)]) == 1
    assert run_cli(["run", "--system", str(tmp_path / "nope.yaml"), "--dataset", DATASET]) == 1
    assert "cannot read system config" in capsys.readouterr().err


def test_run_cli_success_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(["version"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == __version__
