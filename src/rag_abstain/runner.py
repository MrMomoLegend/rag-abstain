"""The run loop (DESIGN section 9): ask every question R times and write records as they come.

A run directory holds `manifest.json` and an append-only `records.jsonl`, one line per
(qid, repeat) attempt, flushed to disk after every record, so a crash loses at most the call in
flight. Resuming a run skips pairs that already have a result and retries pairs whose latest
record is an error (a quota hit or an outage); the retry is appended, the old error line stays
as history, and the latest record for a pair is the one that counts. Any exception raised by
the system becomes an `error` record. Everything written is scrubbed of secret values read from
the environment.
"""

import json
import os
import platform
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from rag_abstain import __version__
from rag_abstain.adapters import System
from rag_abstain.classify import Classifier, RuleClassifier, resolve_outcome
from rag_abstain.dataset import load_dataset, sha256_file
from rag_abstain.metrics.correctness import score_answer
from rag_abstain.metrics.retrieval import score_retrieval
from rag_abstain.models import Question, Record, RunManifest, SystemResponse

MANIFEST_FILE = "manifest.json"
RECORDS_FILE = "records.jsonl"


class RunError(Exception):
    """Raised when a run cannot start, e.g. resuming with a different system or dataset."""


@dataclass
class RunResult:
    """What a run produced.

    `records` is the latest record for every (qid, repeat) in the run directory, old and new.
    `new` counts calls made by this invocation, of which `retried` were retries of pairs whose
    latest record was an error; `skipped` counts pairs that already had a non-error result.
    """

    run_dir: Path
    manifest: RunManifest
    records: list[Record]
    new: int
    skipped: int
    retried: int = 0

    @property
    def errors(self) -> int:
        """Number of error records in the run directory."""
        return sum(r.outcome == "error" for r in self.records)


def run_id_for(name: str, created_at: datetime) -> str:
    """Directory name for a run: `2026-10-19T1530_<name>`, safe on every filesystem."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "system"
    return f"{created_at:%Y-%m-%dT%H%M}_{slug}"


def read_records(run_dir: str | Path) -> list[Record]:
    """Read records.jsonl, keeping the latest record for each (qid, repeat).

    A pair has more than one line when an error was retried on resume; the later line wins and
    takes the earlier one's place in the order. An unterminated last line (a write cut off by a
    crash) is ignored.
    """
    path = Path(run_dir) / RECORDS_FILE
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    complete = lines[:-1]  # the last piece is "" if the file ends in a newline, else partial
    latest: dict[tuple[str, int], Record] = {}
    for line in complete:
        if line.strip():
            record = Record.model_validate_json(line)
            latest[(record.qid, record.repeat)] = record
    return list(latest.values())


def read_manifest(run_dir: str | Path) -> RunManifest:
    """Read manifest.json from a run directory."""
    path = Path(run_dir) / MANIFEST_FILE
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunError(f"{run_dir}: no readable {MANIFEST_FILE} ({exc})") from exc


def _drop_partial_line(path: Path) -> None:
    """Cut an unterminated last line off records.jsonl so appends start on a fresh line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with path.open("r+b") as fh:
            fh.truncate(data.rfind(b"\n") + 1)


def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so a system's raw payload can always be written."""
    return json.loads(json.dumps(value, default=repr))


def evaluate(
    adapter: Callable[[Question], SystemResponse],
    question: Question,
    repeat: int,
    classifier: Classifier,
) -> Record:
    """Ask one question once and build its record. Never raises for a failing system.

    The harness measures latency and uses it when the system does not report its own.
    """
    started = time.perf_counter()
    try:
        response = adapter(question)
        if not isinstance(response, SystemResponse):
            raise TypeError(f"adapter returned {type(response).__name__}, not SystemResponse")
    except Exception as exc:
        response = SystemResponse(error=f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started

    update: dict[str, Any] = {"raw": _json_safe(response.raw)}
    if response.latency_s is None:
        update["latency_s"] = elapsed
    response = response.model_copy(update=update)

    outcome, stage = resolve_outcome(response, classifier)
    scores: dict[str, float | None] = {}
    if outcome != "error":
        if question.expected == "answer":
            scores.update(score_answer(question, response.answer))
        scores.update(score_retrieval(response.sources, question.gold_sources))
    return Record(
        qid=question.qid,
        repeat=repeat,
        question=question.question,
        expected=question.expected,
        category=question.category,
        response=response,
        outcome=outcome,
        decline_stage=stage,
        scores=scores,
    )


class _Writer:
    """Appends scrubbed records to records.jsonl, flushing each one to disk."""

    def __init__(self, fh: IO[str], scrub: Callable[[str], str]) -> None:
        self.fh = fh
        self.scrub = scrub

    def write(self, record: Record) -> Record:
        line = self.scrub(record.model_dump_json())
        self.fh.write(line + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        return Record.model_validate_json(line)


def _new_run_dir(out_dir: Path, name: str, created_at: datetime) -> Path:
    base = run_id_for(name, created_at)
    candidate, n = out_dir / base, 1
    while candidate.exists():
        n += 1
        candidate = out_dir / f"{base}-{n}"
    candidate.mkdir(parents=True)
    return candidate


def _check_resume(
    manifest: RunManifest,
    system: System,
    dataset_sha: str,
    classifier: dict[str, Any],
    repeats: int,
) -> None:
    problems = []
    if manifest.system != system.config:
        problems.append("the system config differs from the one in the manifest")
    if manifest.dataset_sha256 != dataset_sha:
        problems.append("the dataset's SHA-256 differs from the one in the manifest")
    if manifest.classifier != classifier:
        problems.append("the classifier config differs from the one in the manifest")
    if repeats < manifest.repeats:
        problems.append(f"repeats {repeats} is below the run's {manifest.repeats}")
    if problems:
        raise RunError("cannot resume: " + "; ".join(problems))


def run(
    system: System,
    dataset_path: str | Path,
    *,
    repeats: int | None = None,
    workers: int = 1,
    limit: int | None = None,
    out_dir: str | Path = "runs",
    resume: str | Path | None = None,
    classifier: Classifier | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Run `system` over the question set and write the run directory.

    Asks the first `limit` questions (all if None) `repeats` times, repeat 1 for every question
    before repeat 2. With `resume`, appends to that run directory, skipping (qid, repeat) pairs
    that already have a non-error result and retrying pairs whose latest record is an error;
    `repeats` then defaults to the run's and may only grow, and the system, dataset and
    classifier must match the manifest. `workers` > 1 asks questions in parallel
    threads. The default classifier is RuleClassifier with the system's refusal_text.
    The caller owns `system` and closes it.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if repeats is not None and repeats < 1:
        raise ValueError("repeats must be at least 1")

    questions = load_dataset(dataset_path)[:limit]
    dataset_sha = sha256_file(dataset_path)
    classifier = classifier or RuleClassifier(refusal_text=system.refusal_text)
    classifier_config = classifier.config()

    if resume is not None:
        run_dir = Path(resume)
        previous = read_manifest(run_dir)
        repeats = repeats or previous.repeats
        _check_resume(previous, system, dataset_sha, classifier_config, repeats)
        manifest = previous.model_copy(update={"repeats": repeats, "limit": limit})
        _drop_partial_line(run_dir / RECORDS_FILE)
    else:
        repeats = repeats or 1
        created_at = now or datetime.now().astimezone()
        run_dir = _new_run_dir(Path(out_dir), system.name, created_at)
        manifest = RunManifest(
            run_id=run_dir.name,
            created_at=created_at,
            harness_version=__version__,
            system=system.config,
            dataset_path=str(dataset_path),
            dataset_sha256=dataset_sha,
            repeats=repeats,
            limit=limit,
            classifier=classifier_config,
            python_version=platform.python_version(),
        )
    (run_dir / MANIFEST_FILE).write_text(
        system.secrets.scrub(manifest.model_dump_json(indent=2)) + "\n", encoding="utf-8"
    )

    latest = {(r.qid, r.repeat): r for r in read_records(run_dir)}
    todo = [
        (q, rep)
        for rep in range(1, repeats + 1)
        for q in questions
        if (q.qid, rep) not in latest or latest[(q.qid, rep)].outcome == "error"
    ]
    retried = sum((q.qid, rep) in latest for q, rep in todo)
    skipped = len(questions) * repeats - len(todo)

    def keep(record: Record) -> None:
        latest[(record.qid, record.repeat)] = record

    with (run_dir / RECORDS_FILE).open("a", encoding="utf-8", newline="\n") as fh:
        writer = _Writer(fh, system.secrets.scrub)
        if workers == 1:
            for q, rep in todo:
                keep(writer.write(evaluate(system.adapter, q, rep, classifier)))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(evaluate, system.adapter, q, rep, classifier) for q, rep in todo
                ]
                try:
                    for future in as_completed(futures):
                        keep(writer.write(future.result()))
                except BaseException:
                    pool.shutdown(cancel_futures=True)
                    raise

    return RunResult(
        run_dir, manifest, list(latest.values()), new=len(todo), skipped=skipped, retried=retried
    )
