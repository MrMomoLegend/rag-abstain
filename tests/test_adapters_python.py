from pathlib import Path

import pytest

from rag_abstain.adapters import SystemConfigError, load_system
from rag_abstain.adapters.mapping import resolve_map
from rag_abstain.adapters.python_callable import PythonCallableAdapter, import_target
from rag_abstain.models import Question, SystemResponse

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def q(text: str) -> Question:
    return Question(qid="Q", question=text, expected="answer")


def test_toy_system_through_load_system() -> None:
    system = load_system(EXAMPLES / "toy_system.yaml")
    assert system.name == "toy-gate-0.53"
    assert isinstance(system.adapter, PythonCallableAdapter)
    assert system.refusal_text == "The provided documents do not contain this information."

    gate = system.adapter(q("What is the capital of Brazil?"))
    assert gate.decline_stage == "gate" and gate.declined is True
    assert gate.distances == [0.78, 0.80, 0.83]
    assert gate.contexts[0].startswith("(toy chunk 1")
    assert gate.raw["refusal_reason"] == "retrieval_distance"

    answered = system.adapter(q("When was GPT-4 released?"))
    assert answered.answer == "GPT-4 was released on March 14, 2023."
    assert answered.decline_stage is None
    assert answered.latency_s == 0.81
    assert answered.prompt_tokens == 900


def test_params_are_passed_as_keywords(tmp_path: Path) -> None:
    (tmp_path / "gate_off.yaml").write_text(
        f"name: toy-off\nadapter: python\ntarget: toy_system:answer_question\n"
        f"pythonpath: {EXAMPLES.as_posix()}\nparams: {{use_threshold: false}}\n"
        "response_map: demo\n",
        encoding="utf-8",
    )
    system = load_system(tmp_path / "gate_off.yaml")
    assert system.adapter(q("What is the capital of Brazil?")).answer.startswith("The capital")


def test_relative_pythonpath_resolves_against_the_yaml(tmp_path: Path) -> None:
    pkg = tmp_path / "code"
    pkg.mkdir()
    (pkg / "rag_abstain_test_echo.py").write_text(
        "def echo(question, prefix=''):\n    return prefix + question\n", encoding="utf-8"
    )
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "echo.yaml").write_text(
        "name: echo\nadapter: python\ntarget: rag_abstain_test_echo:echo\n"
        "pythonpath: ../code\nparams: {prefix: '> '}\n",
        encoding="utf-8",
    )
    system = load_system(conf / "echo.yaml")
    assert system.adapter(q("hi")) == SystemResponse(answer="> hi")


def test_exceptions_propagate_to_the_runner() -> None:
    def broken(question: str) -> str:
        raise RuntimeError("index missing")

    adapter = PythonCallableAdapter(broken)
    with pytest.raises(RuntimeError, match="index missing"):
        adapter(q("x"))


def test_dict_with_explicit_map() -> None:
    adapter = PythonCallableAdapter(
        lambda question: {"reply": {"text": question.upper()}},
        response_map=resolve_map({"answer": "reply.text"}),
    )
    assert adapter(q("hi")).answer == "HI"


@pytest.mark.parametrize(
    "target,message",
    [
        ("no_colon", "module:function"),
        ("rag_abstain_no_such_module:f", "cannot import"),
        ("rag_abstain.models:NOT_THERE", "not callable"),
    ],
)
def test_bad_targets(target: str, message: str) -> None:
    with pytest.raises(SystemConfigError, match=message):
        import_target(target)
