import json
from pathlib import Path

import httpx
import pytest

from rag_abstain.adapters import SystemConfigError, load_system
from rag_abstain.adapters.http import render_request
from rag_abstain.models import Question

SECRET = "sk-test-0123456789abcdef"

CONFIG = """\
name: remote
adapter: http
url: https://rag.example.test/ask
request: {{query: "{{question}}", top_k: 3}}
headers:
  Authorization: "Bearer ${{RAG_ABSTAIN_TEST_KEY}}"
response_map: {response_map}
retries: {retries}
retry_backoff_s: 0
"""


def q(text: str = "What is TF-IDF?") -> Question:
    return Question(qid="Q", question=text, expected="answer")


def write(tmp_path: Path, response_map: str = "demo", retries: int = 1) -> Path:
    path = tmp_path / "remote.yaml"
    path.write_text(CONFIG.format(response_map=response_map, retries=retries), encoding="utf-8")
    return path


class Server:
    """Records requests and replies with queued (status, body) pairs; the last one repeats."""

    def __init__(self, *replies: tuple[int, object]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ABSTAIN_TEST_KEY", SECRET)


def test_posts_template_with_env_header_and_maps_reply(tmp_path: Path) -> None:
    server = Server((200, {"answer": "Term frequency.", "sources": ["Tf-idf.txt"]}))
    system = load_system(write(tmp_path), transport=httpx.MockTransport(server))
    response = system.adapter(q('Say "hi"\nnow'))

    assert response.answer == "Term frequency." and response.sources == ["Tf-idf.txt"]
    sent = server.requests[0]
    assert sent.headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(sent.content) == {"query": 'Say "hi"\nnow', "top_k": 3}
    assert SECRET in system.secrets.values
    assert system.config["headers"]["Authorization"] == "Bearer ${RAG_ABSTAIN_TEST_KEY}"


def test_non_2xx_is_an_error_not_a_crash(tmp_path: Path) -> None:
    server = Server((403, "forbidden"))
    system = load_system(write(tmp_path), transport=httpx.MockTransport(server))
    response = system.adapter(q())
    assert response.error == "HTTP 403: forbidden"
    assert len(server.requests) == 1  # 4xx other than 429 is not retried


def test_invalid_json_is_an_error(tmp_path: Path) -> None:
    system = load_system(
        write(tmp_path), transport=httpx.MockTransport(Server((200, "<html>oops</html>")))
    )
    response = system.adapter(q())
    assert response.error is not None and response.error.startswith("invalid JSON")


def test_json_that_is_not_an_object_is_an_error(tmp_path: Path) -> None:
    system = load_system(write(tmp_path), transport=httpx.MockTransport(Server((200, [1, 2]))))
    assert system.adapter(q()).error == "expected a JSON object, got list"


def test_reply_that_does_not_fit_the_map_is_an_error(tmp_path: Path) -> None:
    server = Server((200, {"answer": "x", "refusal_reason": "budget"}))
    system = load_system(write(tmp_path), transport=httpx.MockTransport(server))
    response = system.adapter(q())
    assert response.error is not None and "decline_stage" in response.error
    assert response.raw == {"answer": "x", "refusal_reason": "budget"}


def test_5xx_is_retried_then_succeeds(tmp_path: Path) -> None:
    server = Server((503, "busy"), (200, {"answer": "ok"}))
    system = load_system(write(tmp_path), transport=httpx.MockTransport(server))
    assert system.adapter(q()).answer == "ok"
    assert len(server.requests) == 2


def test_retries_run_out(tmp_path: Path) -> None:
    server = Server((429, "slow down"))
    system = load_system(write(tmp_path, retries=2), transport=httpx.MockTransport(server))
    assert system.adapter(q()).error == "HTTP 429: slow down"
    assert len(server.requests) == 3


def test_transport_error_is_retried_and_reported(tmp_path: Path) -> None:
    calls = []

    def refuse(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    system = load_system(write(tmp_path), transport=httpx.MockTransport(refuse))
    assert system.adapter(q()).error == "ConnectError: connection refused"
    assert len(calls) == 2


def test_missing_env_var_fails_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_ABSTAIN_TEST_KEY")
    with pytest.raises(SystemConfigError, match="RAG_ABSTAIN_TEST_KEY"):
        load_system(write(tmp_path))


def test_render_request_escapes_the_question() -> None:
    assert render_request('{"q": "{question}"}', 'a "b" \\ c') == {"q": 'a "b" \\ c'}
    assert render_request({"q": "Q: {question}", "n": 1}, "x") == {"q": "Q: x", "n": 1}


@pytest.mark.parametrize(
    "request_line,message",
    [("request: {query: fixed}", "must contain"), ("request: '{question}'", "not valid JSON")],
)
def test_bad_request_templates(tmp_path: Path, request_line: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"name: bad\nadapter: http\nurl: https://x.test\n{request_line}\n", encoding="utf-8"
    )
    with pytest.raises(SystemConfigError, match=message):
        load_system(path)
