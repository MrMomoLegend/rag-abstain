from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag_abstain.models import Question, Record, RunManifest, SystemResponse


def test_question_minimal_fields_and_defaults() -> None:
    q = Question(qid="A01", question="What is X?", expected="answer")
    assert q.category is None
    assert q.references == []
    assert q.key_terms == []
    assert q.gold_sources == []


def test_question_expected_is_case_and_space_insensitive() -> None:
    assert Question(qid="A", question="q", expected=" Decline ").expected == "decline"


@pytest.mark.parametrize("expected", ["refuse", "", "yes"])
def test_question_rejects_invalid_expected(expected: str) -> None:
    with pytest.raises(ValidationError):
        Question(qid="A", question="q", expected=expected)


@pytest.mark.parametrize("field", ["qid", "question"])
def test_question_rejects_blank_required_text(field: str) -> None:
    data = {"qid": "A", "question": "q", "expected": "answer", field: "   "}
    with pytest.raises(ValidationError):
        Question(**data)


def test_question_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Question(qid="A", question="q", expected="answer", colour="red")


def test_question_key_terms_accepts_bare_strings_and_drops_blanks() -> None:
    q = Question(qid="A", question="q", expected="answer", key_terms=["Cocke", ["a", " "], []])
    assert q.key_terms == [["Cocke"], ["a"]]


def test_question_blank_category_becomes_none() -> None:
    assert Question(qid="A", question="q", expected="answer", category="").category is None


def test_system_response_defaults() -> None:
    r = SystemResponse()
    assert r.answer == ""
    assert r.declined is None
    assert r.decline_stage is None
    assert r.error is None


def test_system_response_rejects_unknown_stage() -> None:
    with pytest.raises(ValidationError):
        SystemResponse(decline_stage="retriever")


def test_record_round_trips_through_json() -> None:
    rec = Record(
        qid="A01",
        repeat=2,
        question="q",
        expected="decline",
        response=SystemResponse(answer="no", decline_stage="gate", distances=[0.7]),
        outcome="declined",
        decline_stage="gate",
        scores={"f1": None},
    )
    assert Record.model_validate_json(rec.model_dump_json()) == rec


def test_record_repeat_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        Record(
            qid="A",
            repeat=0,
            question="q",
            expected="answer",
            response=SystemResponse(),
            outcome="answered",
        )


def test_run_manifest_serialises() -> None:
    m = RunManifest(
        run_id="2026-10-19T1530_toy",
        created_at=datetime(2026, 10, 19, 15, 30, tzinfo=UTC),
        harness_version="0.1.0",
        system={"name": "toy"},
        dataset_path="examples/toy_questions.csv",
        dataset_sha256="0" * 64,
        repeats=1,
        classifier={"name": "rules"},
        python_version="3.13.5",
    )
    assert RunManifest.model_validate_json(m.model_dump_json()) == m
