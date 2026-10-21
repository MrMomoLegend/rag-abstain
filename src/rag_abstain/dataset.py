"""Load and validate question sets in CSV or JSONL (DESIGN section 4)."""

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_abstain.models import Question

REQUIRED_FIELDS = ("qid", "question", "expected")
KNOWN_FIELDS = frozenset(Question.model_fields)
CSV_SUFFIXES = frozenset({".csv"})
JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})


class DatasetError(Exception):
    """Raised when a question set has validation errors. `errors` lists every problem found."""

    def __init__(self, path: Path, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        super().__init__(f"{path}: {len(errors)} error(s)\n" + "\n".join(errors))


@dataclass
class DatasetReport:
    """Result of validating a question set: the valid questions plus errors and warnings."""

    path: Path
    questions: list[Question] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return not self.errors


def split_list(cell: str) -> list[str]:
    """Decode a CSV list cell: items separated by `|`, blanks dropped."""
    return [item.strip() for item in cell.split("|") if item.strip()]


def split_key_terms(cell: str) -> list[list[str]]:
    """Decode a CSV key_terms cell: groups separated by `;`, alternatives by `|`."""
    groups = (split_list(group) for group in cell.split(";"))
    return [group for group in groups if group]


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, report: DatasetReport) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [name.strip() for name in (reader.fieldnames or [])]
        reader.fieldnames = header
        missing = [name for name in REQUIRED_FIELDS if name not in header]
        if missing:
            report.errors.append(f"missing required column(s): {', '.join(missing)}")
            return rows
        for name in header:
            if name not in KNOWN_FIELDS:
                report.warnings.append(f"unknown column '{name}' ignored")
        for raw in reader:
            if None in raw:
                report.errors.append(f"line {reader.line_num}: more cells than header columns")
                continue
            if not any((value or "").strip() for value in raw.values()):
                continue
            row: dict[str, Any] = {}
            for name, value in raw.items():
                value = value or ""
                if name in ("references", "gold_sources"):
                    row[name] = split_list(value)
                elif name == "key_terms":
                    row[name] = split_key_terms(value)
                elif name in KNOWN_FIELDS:
                    row[name] = value
            rows.append((reader.line_num, row))
    return rows


def _read_jsonl(path: Path, report: DatasetReport) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    unknown: set[str] = set()
    with path.open(encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(obj, dict):
                report.errors.append(f"line {line_no}: expected a JSON object")
                continue
            unknown.update(key for key in obj if key not in KNOWN_FIELDS)
            rows.append((line_no, {k: v for k, v in obj.items() if k in KNOWN_FIELDS}))
    for name in sorted(unknown):
        report.warnings.append(f"unknown field '{name}' ignored")
    return rows


def _format_error(line_no: int, row: dict[str, Any], exc: ValidationError) -> list[str]:
    qid = str(row.get("qid") or "").strip()
    where = f"line {line_no}" + (f" (qid {qid})" if qid else "")
    messages = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "row"
        messages.append(f"{where}: {loc}: {err['msg']}")
    return messages


def validate_dataset(path: str | Path) -> DatasetReport:
    """Read and check a question set.

    Errors: unreadable file or format, missing required columns, invalid rows (empty qid or
    question, `expected` not `answer`/`decline`), duplicate qids, no questions at all.
    Warnings: `expected = answer` rows with neither `references` nor `key_terms`; unknown fields.
    """
    path = Path(path)
    report = DatasetReport(path=path)
    suffix = path.suffix.lower()
    try:
        if suffix in CSV_SUFFIXES:
            rows = _read_csv(path, report)
        elif suffix in JSONL_SUFFIXES:
            rows = _read_jsonl(path, report)
        else:
            report.errors.append(f"unsupported file type '{path.suffix}' (use .csv or .jsonl)")
            return report
    except OSError as exc:
        report.errors.append(f"cannot read file: {exc.strerror or exc}")
        return report
    except UnicodeDecodeError:
        report.errors.append("cannot read file: not valid UTF-8")
        return report

    seen: dict[str, int] = {}
    for line_no, row in rows:
        try:
            question = Question(**row)
        except ValidationError as exc:
            report.errors.extend(_format_error(line_no, row, exc))
            continue
        if question.qid in seen:
            first = seen[question.qid]
            report.errors.append(
                f"line {line_no}: duplicate qid '{question.qid}' (first on line {first})"
            )
            continue
        seen[question.qid] = line_no
        if question.expected == "answer" and not (question.references or question.key_terms):
            report.warnings.append(
                f"line {line_no} (qid {question.qid}): expected=answer but no references "
                "or key_terms, so correctness cannot be scored"
            )
        report.questions.append(question)

    if not rows and not report.errors:
        report.errors.append("no questions found")
    return report


def load_dataset(path: str | Path) -> list[Question]:
    """Load a question set, raising DatasetError if validation finds any error."""
    report = validate_dataset(path)
    if not report.ok:
        raise DatasetError(report.path, report.errors)
    return report.questions
