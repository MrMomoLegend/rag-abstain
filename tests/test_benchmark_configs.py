"""The milestone 2 benchmark inputs (datasets/fyp30.csv, systems/*.yaml) load and are wired right.

The demo configs import `rag.pipeline` from rag-assistant-demo, which is not available here or in
CI, so a stub module stands in for it. Nothing calls the network.
"""

import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag_abstain.adapters import load_system
from rag_abstain.adapters.openai_chat import OpenAIChatAdapter
from rag_abstain.adapters.python_callable import PythonCallableAdapter
from rag_abstain.dataset import load_dataset
from rag_abstain.models import Question

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ROOT / "systems"
REFUSAL = "The provided documents do not contain this information."
FAKE_KEY = "fake-key-for-config-test"


@pytest.fixture
def demo_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a stub `rag.pipeline` that records each call and answers like the demo's gate."""
    calls: list[dict[str, Any]] = []

    def answer_question(question: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"question": question, **kwargs})
        return {
            "answer": REFUSAL,
            "refused": True,
            "refusal_reason": "retrieval_distance",
            "sources": ["GPT-4.txt"],
            "retrieved_chunks": ["chunk"],
            "distances": [0.61],
            "total_s": 0.01,
            "prompt_tokens": None,
            "completion_tokens": None,
        }

    package = types.ModuleType("rag")
    pipeline = types.ModuleType("rag.pipeline")
    pipeline.answer_question = answer_question  # type: ignore[attr-defined]
    package.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rag", package)
    monkeypatch.setitem(sys.modules, "rag.pipeline", pipeline)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return calls


def test_fyp30_dataset() -> None:
    questions = load_dataset(ROOT / "datasets" / "fyp30.csv")
    assert len(questions) == 30
    by_category: dict[str | None, set[str]] = {}
    for question in questions:
        by_category.setdefault(question.category, set()).add(question.expected)
    assert by_category == {
        "answerable": {"answer"},
        "out_of_corpus": {"decline"},
        "related_unanswerable": {"decline"},
    }
    answerable = [q for q in questions if q.expected == "answer"]
    assert len(answerable) == 15
    for question in answerable:
        assert question.key_terms and question.gold_sources
        assert all(source.endswith(".txt") for source in question.gold_sources)
    assert all(not q.gold_sources for q in questions if q.expected == "decline")


@pytest.mark.parametrize(
    "file,name,use_threshold",
    [("demo_gate_on.yaml", "demo-gate-0.53", True), ("demo_gate_off.yaml", "demo-gate-off", False)],
)
def test_demo_configs(
    demo_calls: list[dict[str, Any]], file: str, name: str, use_threshold: bool
) -> None:
    system = load_system(SYSTEMS / file)
    assert system.name == name
    assert system.refusal_text == REFUSAL
    assert isinstance(system.adapter, PythonCallableAdapter)
    assert str((ROOT.parent / "rag-assistant-demo").resolve()) in sys.path

    response = system.adapter(Question(qid="Q", question="When?", expected="answer"))
    assert demo_calls[0]["question"] == "When?"
    assert demo_calls[0]["use_rag"] is True
    assert demo_calls[0]["use_threshold"] is use_threshold
    assert response.decline_stage == "gate" and response.declined is True
    assert response.distances == [0.61] and response.latency_s == 0.01


def test_llama_alone_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("loading a config must not send a request")

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    system = load_system(SYSTEMS / "llama_alone.yaml", transport=httpx.MockTransport(no_network))
    try:
        assert system.name == "llama31-8b-alone"
        assert isinstance(system.adapter, OpenAIChatAdapter)
        assert system.adapter.url == "https://llm.example.test/v1/chat/completions"
        assert system.adapter.prompt_template == "{question}"
        assert system.adapter.body == {
            "model": "@cf/meta/llama-3.1-8b-instruct-fp8",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 1024,
        }
        assert FAKE_KEY in system.secrets.values
        assert FAKE_KEY not in repr(system.config)
    finally:
        system.close()
