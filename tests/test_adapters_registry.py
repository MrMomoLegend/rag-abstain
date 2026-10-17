from pathlib import Path

import pytest

from rag_abstain.adapters import ADAPTERS, SystemConfigError, load_system
from rag_abstain.adapters.base import Secrets, redact_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "system.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_registry_has_the_three_adapters() -> None:
    assert set(ADAPTERS) == {"python", "http", "openai_chat"}


@pytest.mark.parametrize(
    "text,message",
    [
        ("- just\n- a list\n", "must be a YAML mapping"),
        ("name: x\nadapter: grpc\n", "adapter must be one of"),
        ("name: x\n", "adapter must be one of"),
        ("name: x\nadapter: python\n", "target"),
        ("name: x\nadapter: python\ntarget: m:f\ntypo_key: 1\n", "typo_key"),
        ("name: x\nadapter: http\nurl: https://x.test\nresponse_map: nope\n", "preset"),
        ("name: [unclosed\n", "cannot read"),
    ],
)
def test_invalid_configs(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(SystemConfigError, match=message):
        load_system(write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemConfigError, match="cannot read"):
        load_system(tmp_path / "absent.yaml")


def test_redact_config_keeps_references_and_drops_literals() -> None:
    config = {
        "api_key_env": "LLM_API_KEY",
        "headers": {
            "Authorization": "Bearer ${KEY}",
            "X-Api-Key": "${KEY}",
            "X-Other-Token": "literal-secret",
            "Cookie": "session=abc",
            "Accept": "application/json",
        },
        "response_map": {"prompt_tokens": "usage.prompt_tokens", "answer": "answer"},
        "params": {"api_key": "sk-literal", "max_tokens": 5, "keys": ["a"]},
    }
    redacted = redact_config(config)
    assert redacted["api_key_env"] == "LLM_API_KEY"
    assert redacted["headers"] == {
        "Authorization": "Bearer ${KEY}",
        "X-Api-Key": "${KEY}",
        "X-Other-Token": "<redacted>",
        "Cookie": "<redacted>",
        "Accept": "application/json",
    }
    assert redacted["response_map"] == config["response_map"]
    assert redacted["params"] == {"api_key": "<redacted>", "max_tokens": 5, "keys": ["a"]}
    assert config["params"]["api_key"] == "sk-literal"  # the input is not modified


def test_secrets_expand_and_scrub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ABSTAIN_A", 'p"w')
    monkeypatch.setenv("RAG_ABSTAIN_B", "token-b")
    secrets = Secrets()
    assert secrets.expand("x ${RAG_ABSTAIN_A} ${RAG_ABSTAIN_B}", "test") == 'x p"w token-b'
    assert secrets.scrub('raw p"w, json p\\"w, token-b') == (
        "raw <redacted>, json <redacted>, <redacted>"
    )
