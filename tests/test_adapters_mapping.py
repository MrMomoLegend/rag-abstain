import pytest

from rag_abstain.adapters.mapping import (
    MAPPABLE_FIELDS,
    PRESETS,
    FieldMap,
    MappingError,
    apply_map,
    get_path,
    resolve_map,
    to_response,
)
from rag_abstain.models import SystemResponse

DEMO_GATE = {
    "answer": "The provided documents do not contain this information.",
    "refused": True,
    "refusal_reason": "retrieval_distance",
    "sources": ["a.txt", "b.txt"],
    "retrieved_chunks": ["chunk a", "chunk b"],
    "distances": [0.81, 0.84],
    "total_s": 0.01,
    "prompt_tokens": None,
    "completion_tokens": None,
}


def test_demo_preset_is_exactly_design_section_5_2() -> None:
    assert PRESETS["demo"] == {
        "answer": "answer",
        "sources": "sources",
        "distances": "distances",
        "contexts": "retrieved_chunks",
        "latency_s": "total_s",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "decline_stage": {
            "path": "refusal_reason",
            "values": {"retrieval_distance": "gate", "model_judgement": "model"},
        },
        "declined": "refused",
    }


def test_demo_preset_maps_a_gate_decline() -> None:
    response = apply_map(DEMO_GATE, resolve_map("demo"))
    assert response.decline_stage == "gate"
    assert response.declined is True
    assert response.contexts == ["chunk a", "chunk b"]
    assert response.latency_s == 0.01
    assert response.prompt_tokens is None
    assert response.raw == DEMO_GATE


def test_demo_preset_maps_a_model_decline_and_an_answer() -> None:
    model = {**DEMO_GATE, "refusal_reason": "model_judgement", "prompt_tokens": 900}
    assert apply_map(model, resolve_map("demo")).decline_stage == "model"
    answered = {**DEMO_GATE, "refused": False, "refusal_reason": None, "answer": "Brasilia."}
    response = apply_map(answered, resolve_map("demo"))
    assert response.decline_stage is None and response.declined is False


def test_none_is_identity() -> None:
    mapping = resolve_map(None)
    assert set(mapping) == set(MAPPABLE_FIELDS)
    assert "raw" not in mapping
    response = apply_map({"answer": "x", "sources": ["s"], "unrelated": 1}, mapping)
    assert response.answer == "x" and response.sources == ["s"]


def test_explicit_map_with_shorthand_and_values() -> None:
    mapping = resolve_map({"answer": "out.text", "decline_stage": {"path": "why", "values": {}}})
    assert mapping["answer"] == FieldMap(path="out.text")
    assert mapping["decline_stage"].values == {}


@pytest.mark.parametrize(
    "spec,message",
    [
        ("nope", "unknown response_map preset"),
        ({"answr": "answer"}, "unknown field 'answr'"),
        ({"raw": "raw"}, "unknown field 'raw'"),
        ({"answer": {"pth": "x"}}, "response_map.answer"),
        ({"answer": 3}, "response_map.answer"),
        ({}, "non-empty"),
    ],
)
def test_bad_maps_are_rejected(spec: object, message: str) -> None:
    with pytest.raises(MappingError, match=message):
        resolve_map(spec)  # type: ignore[arg-type]


def test_get_path_follows_dicts_and_list_indices() -> None:
    payload = {"choices": [{"message": {"content": "hi"}}], "n": None}
    assert get_path(payload, "choices.0.message.content") == "hi"
    assert get_path(payload, "choices.-1.message.content") == "hi"
    assert get_path(payload, "choices.1.message") is None
    assert get_path(payload, "choices.x") is None
    assert get_path(payload, "missing.deeper") is None
    assert get_path(payload, "n.deeper") is None


def test_unmapped_value_passes_through_and_bad_value_raises() -> None:
    mapping = resolve_map({"decline_stage": {"path": "why", "values": {"dist": "gate"}}})
    assert apply_map({"why": "model"}, mapping).decline_stage == "model"
    with pytest.raises(MappingError, match="decline_stage"):
        apply_map({"why": "budget"}, mapping)


def test_to_response_accepts_the_three_return_types() -> None:
    mapping = resolve_map(None)
    ready = SystemResponse(answer="ready")
    assert to_response(ready, mapping) is ready
    assert to_response("plain", mapping) == SystemResponse(answer="plain")
    assert to_response({"answer": "d"}, mapping).answer == "d"
    with pytest.raises(MappingError, match="returned list"):
        to_response(["no"], mapping)
