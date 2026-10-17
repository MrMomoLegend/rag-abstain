import json
from pathlib import Path

import httpx
import pytest

from rag_abstain.adapters import SystemConfigError, load_system
from rag_abstain.models import Question

SECRET = "sk-chat-fedcba9876543210"

CONFIG = """\
name: llama-alone
adapter: openai_chat
base_url_env: RAG_ABSTAIN_TEST_BASE
api_key_env: RAG_ABSTAIN_TEST_CHAT_KEY
model: "@cf/meta/llama-3.1-8b-instruct-fp8"
prompt_template: "Q: {question}"
temperature: 0
seed: 42
max_tokens: 64
retry_backoff_s: 0
"""

COMPLETION = {
    "choices": [{"message": {"role": "assistant", "content": "Brasilia."}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ABSTAIN_TEST_BASE", "https://llm.example.test/v1/")
    monkeypatch.setenv("RAG_ABSTAIN_TEST_CHAT_KEY", SECRET)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "chat.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def q() -> Question:
    return Question(qid="B01", question="What is the capital of Brazil?", expected="decline")


def reply(status: int, **kwargs: object) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status, **kwargs))  # type: ignore[arg-type]


def test_sends_a_chat_completion_and_reads_the_reply(config: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPLETION)

    system = load_system(config, transport=httpx.MockTransport(handler))
    response = system.adapter(q())

    assert response.answer == "Brasilia."
    assert (response.prompt_tokens, response.completion_tokens) == (12, 3)
    assert response.sources == [] and response.decline_stage is None
    request = seen[0]
    assert str(request.url) == "https://llm.example.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(request.content) == {
        "model": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "temperature": 0,
        "seed": 42,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Q: What is the capital of Brazil?"}],
    }


def test_non_2xx_is_an_error(config: Path) -> None:
    system = load_system(config, transport=reply(401, json={"error": "bad key"}))
    response = system.adapter(q())
    assert response.error is not None and response.error.startswith("HTTP 401")


def test_invalid_json_is_an_error(config: Path) -> None:
    system = load_system(config, transport=reply(200, text="not json"))
    response = system.adapter(q())
    assert response.error is not None and response.error.startswith("invalid JSON")


def test_reply_without_content_is_an_error(config: Path) -> None:
    system = load_system(config, transport=reply(200, json={"choices": []}))
    assert system.adapter(q()).error == "reply has no choices[0].message.content"


def test_missing_usage_leaves_tokens_unknown(config: Path) -> None:
    body = {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": True}}
    response = load_system(config, transport=reply(200, json=body)).adapter(q())
    assert response.prompt_tokens is None and response.completion_tokens is None


def test_literal_base_url_and_no_key(tmp_path: Path) -> None:
    path = tmp_path / "local.yaml"
    path.write_text(
        "name: local\nadapter: openai_chat\nbase_url: http://localhost:8080/v1\nmodel: m\n",
        encoding="utf-8",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPLETION)

    load_system(path, transport=httpx.MockTransport(handler)).adapter(q())
    assert str(seen[0].url) == "http://localhost:8080/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert set(json.loads(seen[0].content)) == {"model", "messages"}


def test_needs_exactly_one_base_url(tmp_path: Path) -> None:
    path = tmp_path / "both.yaml"
    path.write_text(
        "name: x\nadapter: openai_chat\nbase_url: http://a\nbase_url_env: B\nmodel: m\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemConfigError, match="exactly one of base_url_env or base_url"):
        load_system(path)


def test_missing_key_fails_at_load(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_ABSTAIN_TEST_CHAT_KEY")
    with pytest.raises(SystemConfigError, match="RAG_ABSTAIN_TEST_CHAT_KEY"):
        load_system(config)
