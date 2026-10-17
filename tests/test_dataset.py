import hashlib
import json
from pathlib import Path

import pytest

from rag_abstain.dataset import (
    DatasetError,
    load_dataset,
    sha256_file,
    split_key_terms,
    split_list,
    validate_dataset,
)

DESIGN_EXAMPLE = (
    "qid,category,expected,question,references,key_terms,gold_sources\n"
    "A03,answerable,answer,Which researchers is the CYK algorithm named after?,"
    "Cocke Younger Kasami,Cocke;Younger;Kasami,CYK algorithm.txt\n"
    "B01,out_of_corpus,decline,What is the capital of Brazil?,,,\n"
)


def write(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


def test_split_list() -> None:
    assert split_list("a | b||c ") == ["a", "b", "c"]
    assert split_list("") == []


def test_split_key_terms() -> None:
    assert split_key_terms("Cocke;Younger|Y.;Kasami") == [["Cocke"], ["Younger", "Y."], ["Kasami"]]
    assert split_key_terms(" ; ") == []


def test_load_csv_design_example(tmp_path: Path) -> None:
    questions = load_dataset(write(tmp_path, "q.csv", DESIGN_EXAMPLE))
    a03, b01 = questions
    assert a03.qid == "A03"
    assert a03.expected == "answer"
    assert a03.references == ["Cocke Younger Kasami"]
    assert a03.key_terms == [["Cocke"], ["Younger"], ["Kasami"]]
    assert a03.gold_sources == ["CYK algorithm.txt"]
    assert b01.expected == "decline"
    assert b01.references == [] and b01.key_terms == [] and b01.gold_sources == []
    assert b01.category == "out_of_corpus"


def test_load_csv_with_byte_order_mark(tmp_path: Path) -> None:
    path = write(tmp_path, "q.csv", DESIGN_EXAMPLE, encoding="utf-8-sig")
    assert [q.qid for q in load_dataset(path)] == ["A03", "B01"]


def test_load_csv_required_columns_only(tmp_path: Path) -> None:
    path = write(tmp_path, "q.csv", "qid,question,expected\nX1,Why?,decline\n")
    (q,) = load_dataset(path)
    assert q.category is None


def test_load_jsonl(tmp_path: Path) -> None:
    lines = [
        {
            "qid": "A03",
            "question": "Who?",
            "expected": "answer",
            "references": ["Cocke Younger Kasami"],
            "key_terms": [["Cocke"], ["Younger", "Y."]],
            "gold_sources": ["CYK algorithm.txt"],
        },
        {"qid": "B01", "question": "Capital of Brazil?", "expected": "decline"},
    ]
    text = "\n".join(json.dumps(line) for line in lines) + "\n\n"
    a03, b01 = load_dataset(write(tmp_path, "q.jsonl", text))
    assert a03.key_terms == [["Cocke"], ["Younger", "Y."]]
    assert b01.expected == "decline"


def test_csv_and_jsonl_give_the_same_questions(tmp_path: Path) -> None:
    from_csv = load_dataset(write(tmp_path, "q.csv", DESIGN_EXAMPLE))
    text = "\n".join(q.model_dump_json() for q in from_csv)
    assert load_dataset(write(tmp_path, "q.jsonl", text)) == from_csv


def test_duplicate_qid_is_an_error(tmp_path: Path) -> None:
    text = "qid,question,expected\nA,one,decline\nA,two,decline\n"
    report = validate_dataset(write(tmp_path, "q.csv", text))
    assert not report.ok
    assert any("duplicate qid 'A'" in e for e in report.errors)


def test_invalid_expected_is_an_error(tmp_path: Path) -> None:
    report = validate_dataset(write(tmp_path, "q.csv", "qid,question,expected\nA,q,refuse\n"))
    assert len(report.errors) == 1
    assert "qid A" in report.errors[0] and "expected" in report.errors[0]


def test_empty_question_is_an_error(tmp_path: Path) -> None:
    report = validate_dataset(write(tmp_path, "q.csv", "qid,question,expected\nA,  ,decline\n"))
    assert len(report.errors) == 1
    assert "question" in report.errors[0]


def test_missing_required_column_is_an_error(tmp_path: Path) -> None:
    report = validate_dataset(write(tmp_path, "q.csv", "qid,question\nA,q\n"))
    assert report.errors == ["missing required column(s): expected"]


def test_answer_without_references_or_key_terms_is_a_warning(tmp_path: Path) -> None:
    report = validate_dataset(write(tmp_path, "q.csv", "qid,question,expected\nA,q,answer\n"))
    assert report.ok
    assert len(report.questions) == 1
    assert len(report.warnings) == 1 and "qid A" in report.warnings[0]


def test_unknown_column_is_a_warning(tmp_path: Path) -> None:
    text = "qid,question,expected,difficulty\nA,q,decline,hard\n"
    report = validate_dataset(write(tmp_path, "q.csv", text))
    assert report.ok
    assert report.warnings == ["unknown column 'difficulty' ignored"]


def test_jsonl_bad_line_is_an_error_and_others_still_load(tmp_path: Path) -> None:
    text = '{"qid": "A", "question": "q", "expected": "decline"}\n{not json\n[1, 2]\n'
    report = validate_dataset(write(tmp_path, "q.jsonl", text))
    assert len(report.questions) == 1
    assert report.errors[0].startswith("line 2: invalid JSON")
    assert report.errors[1] == "line 3: expected a JSON object"


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    report = validate_dataset(write(tmp_path, "q.csv", "qid,question,expected\n"))
    assert report.errors == ["no questions found"]


def test_unsupported_suffix_is_an_error(tmp_path: Path) -> None:
    report = validate_dataset(write(tmp_path, "q.txt", "x"))
    assert not report.ok and "unsupported file type" in report.errors[0]


def test_load_dataset_raises_with_all_errors(tmp_path: Path) -> None:
    text = "qid,question,expected\nA,,answer\nB,q,maybe\n"
    with pytest.raises(DatasetError) as info:
        load_dataset(write(tmp_path, "q.csv", text))
    assert len(info.value.errors) == 2


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = write(tmp_path, "q.csv", DESIGN_EXAMPLE)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
