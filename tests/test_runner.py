import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from rag_abstain import __version__
from rag_abstain.adapters import System, load_system
from rag_abstain.adapters.base import Secrets
from rag_abstain.dataset import sha256_file
from rag_abstain.models import Question, SystemResponse
from rag_abstain.runner import (
    MANIFEST_FILE,
    RECORDS_FILE,
    RunError,
    read_manifest,
    read_records,
    run,
    run_id_for,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DATASET = EXAMPLES / "toy_questions.csv"
NOW = datetime(2026, 10, 19, 15, 30, tzinfo=timezone(timedelta(hours=8)))


@pytest.fixture
def toy() -> System:
    return load_system(EXAMPLES / "toy_system.yaml")


class Counting:
    """Wraps an adapter and records which questions it was asked."""

    def __init__(self, inner: Callable[[Question], SystemResponse]) -> None:
        self.inner = inner
        self.asked: list[str] = []

    def __call__(self, question: Question) -> SystemResponse:
        self.asked.append(question.qid)
        return self.inner(question)

    def close(self) -> None:
        """Nothing to release."""


def fake_system(adapter: Callable[[Question], SystemResponse], name: str = "fake") -> System:
    return System(
        name=name,
        adapter=adapter,  # type: ignore[arg-type]
        config={"name": name, "adapter": "python"},
        refusal_text=None,
        secrets=Secrets(),
        path=Path("fake.yaml"),
    )


def lines(run_dir: Path) -> list[dict]:
    text = (run_dir / RECORDS_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_toy_run_writes_manifest_and_records(toy: System, tmp_path: Path) -> None:
    result = run(toy, DATASET, out_dir=tmp_path, now=NOW)

    assert result.run_dir == tmp_path / "2026-10-19T1530_toy-gate-0.53"
    assert (result.new, result.skipped, result.errors) == (10, 0, 0)

    manifest = read_manifest(result.run_dir)
    assert manifest.run_id == "2026-10-19T1530_toy-gate-0.53"
    assert manifest.created_at == NOW
    assert manifest.harness_version == __version__
    assert manifest.system["target"] == "toy_system:answer_question"
    assert manifest.dataset_sha256 == sha256_file(DATASET)
    assert manifest.repeats == 1 and manifest.limit is None
    assert manifest.classifier["name"] == "rules"
    refusal = "the provided documents do not contain this information."
    assert refusal in manifest.classifier["markers"]

    by_qid = {r.qid: r for r in read_records(result.run_dir)}
    assert (by_qid["T04"].outcome, by_qid["T08"].outcome) == ("hedged", "declined")
    assert by_qid["T08"].decline_stage == "gate"
    assert by_qid["T06"].decline_stage == "model"
    assert by_qid["T01"].scores["exact_match"] == 0.0
    assert by_qid["T01"].scores["key_term_hit"] == 1.0
    assert by_qid["T01"].scores["mrr"] == 1.0
    assert "exact_match" not in by_qid["T08"].scores  # expected=decline: no correctness
    assert by_qid["T01"].response.latency_s == 0.81  # the system's own timing is kept
    assert result.records == list(read_records(result.run_dir))


def test_repeats_run_repeat_one_first(toy: System, tmp_path: Path) -> None:
    result = run(toy, DATASET, repeats=2, out_dir=tmp_path, now=NOW)
    pairs = [(line["qid"], line["repeat"]) for line in lines(result.run_dir)]
    assert len(pairs) == 20 and len(set(pairs)) == 20
    assert [rep for _, rep in pairs] == [1] * 10 + [2] * 10


def test_limit_takes_the_first_questions(toy: System, tmp_path: Path) -> None:
    result = run(toy, DATASET, limit=3, out_dir=tmp_path, now=NOW)
    assert [line["qid"] for line in lines(result.run_dir)] == ["T01", "T02", "T03"]
    assert read_manifest(result.run_dir).limit == 3


def test_resume_skips_done_pairs(toy: System, tmp_path: Path) -> None:
    first = run(toy, DATASET, limit=3, out_dir=tmp_path, now=NOW)
    counting = Counting(toy.adapter)
    toy.adapter = counting  # type: ignore[assignment]

    again = run(toy, DATASET, limit=3, resume=first.run_dir)
    assert counting.asked == [] and (again.new, again.skipped) == (0, 3)

    rest = run(toy, DATASET, resume=first.run_dir)
    assert counting.asked == [f"T{i:02d}" for i in range(4, 11)]
    assert (rest.new, rest.skipped) == (7, 3)
    assert len(lines(first.run_dir)) == 10
    assert read_manifest(first.run_dir).limit is None


def test_resume_can_add_a_repeat(toy: System, tmp_path: Path) -> None:
    first = run(toy, DATASET, out_dir=tmp_path, now=NOW)
    more = run(toy, DATASET, repeats=2, resume=first.run_dir)
    assert (more.new, more.skipped) == (10, 10)
    assert read_manifest(first.run_dir).repeats == 2
    assert run(toy, DATASET, resume=first.run_dir).new == 0  # repeats default to the run's


def test_resume_repairs_a_cut_off_last_line(toy: System, tmp_path: Path) -> None:
    first = run(toy, DATASET, limit=2, out_dir=tmp_path, now=NOW)
    with (first.run_dir / RECORDS_FILE).open("a", encoding="utf-8") as fh:
        fh.write('{"qid": "T03", "repe')  # a crash in the middle of a write
    assert len(read_records(first.run_dir)) == 2

    result = run(toy, DATASET, limit=3, resume=first.run_dir)
    assert result.new == 1
    assert [line["qid"] for line in lines(first.run_dir)] == ["T01", "T02", "T03"]


def test_resume_refuses_a_different_dataset_or_system(toy: System, tmp_path: Path) -> None:
    first = run(toy, DATASET, repeats=2, out_dir=tmp_path, now=NOW)
    other = tmp_path / "other.csv"
    other.write_text(DATASET.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RunError, match="dataset"):
        run(toy, other, resume=first.run_dir)
    with pytest.raises(RunError, match="system config"):
        run(fake_system(toy.adapter), DATASET, resume=first.run_dir)
    with pytest.raises(RunError, match="repeats 1 is below"):
        run(toy, DATASET, repeats=1, resume=first.run_dir)
    with pytest.raises(RunError, match="manifest"):
        run(toy, DATASET, resume=tmp_path / "nowhere")


def test_raising_system_becomes_error_records(tmp_path: Path) -> None:
    def flaky(question: Question) -> SystemResponse:
        if question.qid in ("T02", "T05"):
            raise RuntimeError(f"vector store down for {question.qid}")
        return SystemResponse(answer="An answer.")

    result = run(fake_system(flaky), DATASET, out_dir=tmp_path, now=NOW)
    assert result.new == 10 and result.errors == 2
    errors = {r.qid: r for r in result.records if r.outcome == "error"}
    assert errors["T02"].response.error == "RuntimeError: vector store down for T02"
    assert errors["T02"].scores == {} and errors["T02"].response.latency_s is not None


def test_wrong_return_type_becomes_an_error_record(tmp_path: Path) -> None:
    result = run(fake_system(lambda q: "not a response"), DATASET, limit=1, out_dir=tmp_path)  # type: ignore[arg-type,return-value]
    assert result.records[0].response.error == (
        "TypeError: adapter returned str, not SystemResponse"
    )


def test_unserialisable_raw_payload_is_kept_as_repr(tmp_path: Path) -> None:
    marker = object()
    system = fake_system(lambda q: SystemResponse(answer="x", raw={"obj": marker}))
    result = run(system, DATASET, limit=1, out_dir=tmp_path)
    assert result.records[0].response.raw == {"obj": repr(marker)}


def test_every_record_is_on_disk_before_a_crash(tmp_path: Path) -> None:
    def interrupted(question: Question) -> SystemResponse:
        if question.qid == "T04":
            raise KeyboardInterrupt  # not an Exception: the run stops, as on Ctrl+C
        return SystemResponse(answer="An answer.")

    with pytest.raises(KeyboardInterrupt):
        run(fake_system(interrupted), DATASET, out_dir=tmp_path, now=NOW)
    run_dir = tmp_path / run_id_for("fake", NOW)
    assert [line["qid"] for line in lines(run_dir)] == ["T01", "T02", "T03"]


def test_workers_give_the_same_records(toy: System, tmp_path: Path) -> None:
    serial = run(toy, DATASET, repeats=2, out_dir=tmp_path / "a", now=NOW)
    parallel = run(toy, DATASET, repeats=2, workers=4, out_dir=tmp_path / "b", now=NOW)

    def key(records: list) -> dict:
        return {(r.qid, r.repeat): (r.outcome, r.response.answer) for r in records}

    assert len(parallel.records) == 20
    assert key(parallel.records) == key(serial.records)


def test_run_ids_do_not_collide(toy: System, tmp_path: Path) -> None:
    a = run(toy, DATASET, limit=1, out_dir=tmp_path, now=NOW)
    b = run(toy, DATASET, limit=1, out_dir=tmp_path, now=NOW)
    assert b.run_dir.name == a.run_dir.name + "-2"
    assert run_id_for("my system / v2", NOW) == "2026-10-19T1530_my-system-v2"


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"limit": 0}, {"repeats": 0}])
def test_bad_arguments(toy: System, tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        run(toy, DATASET, out_dir=tmp_path, **kwargs)


SECRET = "sk-live-do-not-leak-4242"


def test_env_secrets_never_reach_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_ABSTAIN_RUN_KEY", SECRET)
    config = tmp_path / "remote.yaml"
    config.write_text(
        "name: remote\nadapter: http\nurl: https://rag.example.test/ask\n"
        'headers: {Authorization: "Bearer ${RAG_ABSTAIN_RUN_KEY}",\n'
        '          X-Key: "${RAG_ABSTAIN_RUN_KEY}"}\n'
        "retries: 0\n",
        encoding="utf-8",
    )

    def echo_the_key(request: httpx.Request) -> httpx.Response:
        # A badly behaved server that echoes the credential in answers and errors.
        auth = request.headers["authorization"]
        if b"GPT-4" in request.content:
            return httpx.Response(401, text=f"bad credential {auth}")
        return httpx.Response(200, json={"answer": f"Your key is {auth}", "raw_key": SECRET})

    system = load_system(config, transport=httpx.MockTransport(echo_the_key))
    result = run(system, DATASET, out_dir=tmp_path / "runs", now=NOW)

    manifest_text = (result.run_dir / MANIFEST_FILE).read_text(encoding="utf-8")
    records_text = (result.run_dir / RECORDS_FILE).read_text(encoding="utf-8")
    assert SECRET not in manifest_text
    assert SECRET not in records_text
    assert "${RAG_ABSTAIN_RUN_KEY}" in manifest_text
    assert "<redacted>" in records_text
    assert result.errors > 0  # the 401s were recorded, scrubbed
    assert all(SECRET not in r.model_dump_json() for r in result.records)
