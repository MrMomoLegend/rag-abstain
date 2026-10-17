"""The example toy system, question set and config, run through the phase A pieces."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from rag_abstain.classify import RuleClassifier, resolve_outcome
from rag_abstain.dataset import load_dataset, validate_dataset
from rag_abstain.metrics.abstention import abstention_metrics
from rag_abstain.metrics.correctness import score_answer
from rag_abstain.models import Question, Record, SystemResponse

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def toy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("toy_system", EXAMPLES / "toy_system.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("toy_system", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def questions() -> list[Question]:
    return load_dataset(EXAMPLES / "toy_questions.csv")


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return yaml.safe_load((EXAMPLES / "toy_system.yaml").read_text(encoding="utf-8"))


def to_response(result: dict[str, Any]) -> SystemResponse:
    """Minimal stand-in for the `demo` response_map preset (the real mapping is phase B)."""
    stage = {"retrieval_distance": "gate", "model_judgement": "model"}
    return SystemResponse(
        answer=result["answer"],
        declined=result["refused"],
        decline_stage=stage.get(result["refusal_reason"]),
        sources=result["sources"],
        contexts=result["retrieved_chunks"],
        distances=result["distances"],
        latency_s=result["total_s"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        raw=result,
    )


def run(
    toy: ModuleType, questions: list[Question], config: dict[str, Any], use_threshold: bool
) -> list[Record]:
    classifier = RuleClassifier(refusal_text=config["refusal_text"])
    records = []
    for q in questions:
        response = to_response(toy.answer_question(q.question, use_threshold=use_threshold))
        outcome, stage = resolve_outcome(response, classifier)
        records.append(
            Record(
                qid=q.qid,
                question=q.question,
                expected=q.expected,
                category=q.category,
                response=response,
                outcome=outcome,
                decline_stage=stage,
                scores=score_answer(q, response.answer),
            )
        )
    return records


def test_toy_questions_csv_is_clean() -> None:
    report = validate_dataset(EXAMPLES / "toy_questions.csv")
    assert report.errors == [] and report.warnings == []
    assert 8 <= len(report.questions) <= 10


def test_toy_system_yaml(config: dict[str, Any], toy: ModuleType) -> None:
    assert config["adapter"] == "python"
    assert config["response_map"] == "demo"
    assert config["params"] == {"use_threshold": True}
    module_name, func_name = config["target"].split(":")
    assert (EXAMPLES / config["pythonpath"] / f"{module_name}.py").is_file()
    assert getattr(toy, func_name) is toy.answer_question
    assert config["refusal_text"] == toy.REFUSAL


def test_toy_is_deterministic(toy: ModuleType, questions: list[Question]) -> None:
    for q in questions:
        assert toy.answer_question(q.question) == toy.answer_question(q.question)


def test_toy_matches_questions_case_and_space_insensitively(toy: ModuleType) -> None:
    a = toy.answer_question("When was GPT-4 released?")
    b = toy.answer_question("  when was   gpt-4 RELEASED? ")
    assert a["answer"] == b["answer"]


def test_unknown_question_is_declined_at_the_gate(toy: ModuleType) -> None:
    result = toy.answer_question("What is the boiling point of tungsten?")
    assert result["refusal_reason"] == "retrieval_distance"
    assert result["distances"][0] > toy.DISTANCE_THRESHOLD
    assert result["prompt_tokens"] is None


def test_unknown_question_with_gate_off_reaches_the_model(toy: ModuleType) -> None:
    result = toy.answer_question("What is the boiling point of tungsten?", use_threshold=False)
    assert result["refusal_reason"] == "model_judgement"
    assert result["answer"] == toy.REFUSAL
    assert result["threshold"] is None


def test_returned_lists_are_fresh_copies(toy: ModuleType) -> None:
    toy.answer_question("When was GPT-4 released?")["sources"].append("junk")
    assert "junk" not in toy.answer_question("When was GPT-4 released?")["sources"]


GATE_ON = {
    "T01": ("answered", None),
    "T02": ("answered", None),
    "T03": ("answered", None),
    "T04": ("hedged", None),
    "T05": ("answered", None),
    "T06": ("declined", "model"),
    "T07": ("answered", None),
    "T08": ("declined", "gate"),
    "T09": ("declined", "gate"),
    "T10": ("declined", "gate"),
}
GATE_OFF = {
    **GATE_ON,
    "T08": ("answered", None),
    "T09": ("declined", "model"),
    "T10": ("answered", None),
}


@pytest.mark.parametrize("use_threshold,expected", [(True, GATE_ON), (False, GATE_OFF)])
def test_outcomes_cover_every_case(
    toy: ModuleType,
    questions: list[Question],
    config: dict[str, Any],
    use_threshold: bool,
    expected: dict[str, tuple[str, str | None]],
) -> None:
    records = run(toy, questions, config, use_threshold)
    assert {r.qid: (r.outcome, r.decline_stage) for r in records} == expected


def test_hedge_is_flagged_refused_by_toy_but_resolved_as_hedged(
    toy: ModuleType, questions: list[Question], config: dict[str, Any]
) -> None:
    t04 = next(r for r in run(toy, questions, config, True) if r.qid == "T04")
    assert t04.response.declined is True and t04.response.decline_stage == "model"
    assert t04.outcome == "hedged"


def test_abstention_metrics_on_toy_run(
    toy: ModuleType, questions: list[Question], config: dict[str, Any]
) -> None:
    on = abstention_metrics(run(toy, questions, config, True))
    assert on["unsupported_answer_rate"] == pytest.approx(1 / 4)  # T07
    assert on["false_decline_rate"] == pytest.approx(1 / 6)  # T10
    assert on["declines_by_stage"] == {"gate": 3, "model": 1}
    off = abstention_metrics(run(toy, questions, config, False))
    assert off["unsupported_answer_rate"] == pytest.approx(2 / 4)  # T07, T08
    assert off["false_decline_rate"] == 0.0


def test_key_terms_score_the_wrong_answer_zero(
    toy: ModuleType, questions: list[Question], config: dict[str, Any]
) -> None:
    hits = {r.qid: r.scores["key_term_hit"] for r in run(toy, questions, config, False)}
    assert hits["T05"] == 0.0
    assert all(hits[qid] == 1.0 for qid in ("T01", "T02", "T03", "T10"))
