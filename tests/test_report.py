import json
from pathlib import Path

import httpx
import pytest

from rag_abstain.adapters import System, load_system
from rag_abstain.adapters.base import Secrets
from rag_abstain.models import Question, SystemResponse
from rag_abstain.report import (
    BARE_MODEL_CAVEAT,
    REPORT_FILE,
    SUMMARY_FILE,
    caveats,
    cell,
    fmt,
    render_markdown,
    summarise,
    write_reports,
)
from rag_abstain.runner import run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DATASET = EXAMPLES / "toy_questions.csv"


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    system = load_system(EXAMPLES / "toy_system.yaml")
    return run(system, DATASET, repeats=2, out_dir=tmp_path_factory.mktemp("runs")).run_dir


def test_summary_has_every_section(toy_run: Path) -> None:
    summary = summarise(toy_run)
    assert summary["system"] == {"name": "toy-gate-0.53", "adapter": "python"}
    assert summary["records"] == 20 and summary["errors"] == [] and summary["caveats"] == []
    assert summary["abstention"]["unsupported_answer_rate"] == pytest.approx(0.25)
    assert summary["abstention"]["false_decline_rate"] == pytest.approx(1 / 6)
    assert summary["correctness"]["answered_only"]["n"] == 4
    assert summary["retrieval"]["mrr"] == 1.0
    assert summary["determinism"]["outcome_agreement_rate"] == 1.0
    assert summary["cost"]["latency"]["gate"]["n"] == 6


def test_write_reports_writes_both_files(toy_run: Path) -> None:
    summary = write_reports(toy_run)
    assert json.loads((toy_run / SUMMARY_FILE).read_text(encoding="utf-8")) == summary
    text = (toy_run / REPORT_FILE).read_text(encoding="utf-8")
    assert text == render_markdown(summary)


def test_markdown_tables_and_wording(toy_run: Path) -> None:
    text = render_markdown(summarise(toy_run))
    for heading in ("Abstention", "Correctness", "Retrieval", "Determinism", "Cost", "Errors"):
        assert f"## {heading}" in text
    assert "| **all** | 10 | 0 | 5 | 4 | 1 | 0.250 | 0.167 |" in text
    assert "| out_of_corpus | 2 |" in text
    assert "Declines by stage: gate 3, model 1." in text
    assert "Outcome flips: none." in text
    assert "hallucination" not in text.lower()
    assert "\u2014" not in text  # no em dashes


def test_error_records_are_listed(tmp_path: Path) -> None:
    def broken(question: Question) -> SystemResponse:
        if question.qid == "T03":
            raise RuntimeError("index | missing\nsecond line")
        return SystemResponse(answer="Fine.")

    system = System(
        "broken", broken, {"name": "broken", "adapter": "python"}, None, Secrets(), Path("x.yaml")
    )  # type: ignore[arg-type]
    run_dir = run(system, DATASET, out_dir=tmp_path).run_dir
    summary = summarise(run_dir)
    assert summary["errors"] == [
        {"qid": "T03", "repeat": 1, "error": "RuntimeError: index | missing\nsecond line"}
    ]
    assert "| T03 | 1 | RuntimeError: index \\| missing second line |" in render_markdown(summary)


def test_bare_question_caveat_only_for_bare_openai_chat() -> None:
    assert caveats({"adapter": "openai_chat"}) == [BARE_MODEL_CAVEAT]
    assert caveats({"adapter": "openai_chat", "prompt_template": " {question} "}) == [
        BARE_MODEL_CAVEAT
    ]
    assert (
        caveats({"adapter": "openai_chat", "prompt_template": "Use only facts. {question}"}) == []
    )
    assert caveats({"adapter": "python"}) == []


def test_openai_chat_run_prints_the_caveat(tmp_path: Path) -> None:
    config = tmp_path / "bare.yaml"
    config.write_text(
        "name: bare\nadapter: openai_chat\nbase_url: http://llm.test/v1\nmodel: m\n",
        encoding="utf-8",
    )
    reply = {"choices": [{"message": {"content": "Brasilia."}}]}
    system = load_system(
        config, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply))
    )
    summary = write_reports(run(system, DATASET, out_dir=tmp_path).run_dir)
    assert summary["caveats"] == [BARE_MODEL_CAVEAT]
    assert f"> {BARE_MODEL_CAVEAT}" in render_markdown(summary)
    assert "Not scored: no record has both gold sources and returned sources." in (
        render_markdown(summary)
    )


def test_fmt_and_cell() -> None:
    assert fmt(None) == "n/a" and fmt(0.5) == "0.500" and fmt(3) == "3"
    assert cell("a|b\n  c") == "a\\|b c"
