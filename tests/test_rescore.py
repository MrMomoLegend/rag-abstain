from pathlib import Path

import pytest

from rag_abstain.adapters import System, load_system
from rag_abstain.adapters.base import Secrets
from rag_abstain.cli import EXIT_OK, EXIT_USAGE, run_cli
from rag_abstain.compare import compare_runs
from rag_abstain.models import Question, SystemResponse
from rag_abstain.report import REPORT_FILE, SUMMARY_FILE
from rag_abstain.rescore import rescore
from rag_abstain.runner import RECORDS_FILE, RunError, read_manifest, read_records, run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DATASET = EXAMPLES / "toy_questions.csv"


def clarifying_system() -> System:
    def adapter(question: Question) -> SystemResponse:
        if question.qid == "U01":
            return SystemResponse(answer="Could you clarify which model you mean?")
        return SystemResponse(answer="It opened in 1932.")

    return System(
        name="clarifier",
        adapter=adapter,  # type: ignore[arg-type]
        config={"name": "clarifier", "adapter": "python", "refusal_text": "No idea."},
        refusal_text="No idea.",
        secrets=Secrets(),
        path=Path("clarifier.yaml"),
    )


def test_rescore_with_rule_fyp_reproduces_the_run(tmp_path: Path) -> None:
    toy = load_system(EXAMPLES / "toy_system.yaml")
    result = run(toy, DATASET, repeats=2, out_dir=tmp_path)

    out, summary = rescore(result.run_dir, "rule-fyp")

    assert out == result.run_dir / "rescore_rule-fyp"
    assert (out / SUMMARY_FILE).exists() and (out / REPORT_FILE).exists()
    before = [(r.qid, r.repeat, r.outcome, r.decline_stage, r.scores) for r in result.records]
    after = [(r.qid, r.repeat, r.outcome, r.decline_stage, r.scores) for r in read_records(out)]
    assert after == before
    manifest = read_manifest(out)
    assert manifest.rescored_from == result.run_dir.name
    assert manifest.run_id == f"{result.run_dir.name}+rule-fyp"
    assert manifest.classifier == read_manifest(result.run_dir).classifier
    assert summary["records"] == 20
    assert compare_runs(result.run_dir, out)["changed"] == []


def test_rescore_with_rule_v2_changes_only_clarifications(tmp_path: Path) -> None:
    dataset = tmp_path / "q.csv"
    dataset.write_text(
        "qid,expected,question\nU01,decline,How big is it?\nA01,answer,When did it open?\n",
        encoding="utf-8",
    )
    result = run(clarifying_system(), dataset, out_dir=tmp_path / "runs")
    original = (result.run_dir / RECORDS_FILE).read_text(encoding="utf-8")

    out, summary = rescore(result.run_dir, "rule-v2")

    outcomes = {r.qid: (r.outcome, r.decline_stage) for r in read_records(out)}
    assert outcomes == {"U01": ("declined", "model"), "A01": ("answered", None)}
    assert {r.qid: r.outcome for r in result.records} == {"U01": "answered", "A01": "answered"}
    assert summary["classifier"] == "rule-v2"
    assert summary["abstention"]["unsupported_answer_rate"] == 0.0
    assert (result.run_dir / RECORDS_FILE).read_text(encoding="utf-8") == original
    changed = compare_runs(result.run_dir, out)["changed"]
    assert [c["qid"] for c in changed] == ["U01"]


def test_rescore_refuses_a_rescore_and_an_unknown_classifier(tmp_path: Path) -> None:
    result = run(load_system(EXAMPLES / "toy_system.yaml"), DATASET, out_dir=tmp_path)
    out, _ = rescore(result.run_dir, "rule-v2")
    with pytest.raises(RunError, match="already a rescore"):
        rescore(out, "rule-fyp")
    with pytest.raises(ValueError, match="unknown classifier"):
        rescore(result.run_dir, "judge")


def test_rescore_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = run(load_system(EXAMPLES / "toy_system.yaml"), DATASET, out_dir=tmp_path)
    assert run_cli(["rescore", str(result.run_dir), "--classifier", "rule-v2"]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "Rescored:" in printed and "Unsupported answer rate:" in printed
    assert (result.run_dir / "rescore_rule-v2" / SUMMARY_FILE).exists()
    assert run_cli(["rescore", str(result.run_dir), "--classifier", "judge"]) == EXIT_USAGE
