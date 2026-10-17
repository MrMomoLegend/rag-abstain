"""Resolve `response_map` (DESIGN section 5.2): turn a system's dict into a SystemResponse.

A map goes from SystemResponse field names to dotted paths in the returned payload. A path step
is a dict key, or an integer index into a list (`choices.0.message.content`). An entry is either
a path string or `{path, values}`, where `values` translates what was found (for example
`retrieval_distance` -> `gate`); a value missing from `values` is passed through unchanged.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from rag_abstain.models import SystemResponse

# Fields a map may fill. `raw` is always the whole payload, so it is not mappable.
MAPPABLE_FIELDS: tuple[str, ...] = tuple(f for f in SystemResponse.model_fields if f != "raw")

PRESETS: dict[str, dict[str, Any]] = {
    # Matches the dict returned by rag-assistant-demo's rag.pipeline.answer_question.
    "demo": {
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
    },
}

_MISSING = object()


class MappingError(Exception):
    """Raised for an invalid response_map, or a payload that does not fit the SystemResponse."""


class FieldMap(BaseModel):
    """Where one SystemResponse field comes from, with an optional value translation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    values: dict[str, Any] | None = None


ResponseMap = dict[str, FieldMap]


def identity_map() -> ResponseMap:
    """The map used when none is given: each field is read from the key of the same name."""
    return {name: FieldMap(path=name) for name in MAPPABLE_FIELDS}


def resolve_map(spec: str | dict[str, Any] | None) -> ResponseMap:
    """Turn a preset name, an explicit mapping or None (identity) into a validated ResponseMap."""
    if spec is None:
        return identity_map()
    if isinstance(spec, str):
        if spec not in PRESETS:
            known = ", ".join(sorted(PRESETS))
            raise MappingError(f"unknown response_map preset '{spec}' (known: {known})")
        spec = PRESETS[spec]
    if not isinstance(spec, dict) or not spec:
        raise MappingError("response_map must be a preset name or a non-empty mapping")

    resolved: ResponseMap = {}
    for name, entry in spec.items():
        if name not in MAPPABLE_FIELDS:
            raise MappingError(
                f"response_map: unknown field '{name}' (allowed: {', '.join(MAPPABLE_FIELDS)})"
            )
        try:
            resolved[name] = FieldMap(path=entry) if isinstance(entry, str) else FieldMap(**entry)
        except (TypeError, ValidationError) as exc:
            raise MappingError(
                f"response_map.{name}: expected a path or {{path, values}} ({exc})"
            ) from exc
    return resolved


def get_path(payload: Any, path: str) -> Any:
    """Follow a dotted path through dicts and lists. Returns None if any step is missing."""
    node = payload
    for step in path.split("."):
        if isinstance(node, dict):
            node = node.get(step, _MISSING)
        elif isinstance(node, list) and step.lstrip("-").isdigit():
            index = int(step)
            node = node[index] if -len(node) <= index < len(node) else _MISSING
        else:
            return None
        if node is _MISSING:
            return None
    return node


def apply_map(payload: dict[str, Any], mapping: ResponseMap) -> SystemResponse:
    """Build a SystemResponse from `payload`. Paths that resolve to None leave the default.

    Raises MappingError if a found value does not fit its field (e.g. a stage that is neither
    `gate` nor `model` after translation).
    """
    fields: dict[str, Any] = {}
    for name, field_map in mapping.items():
        value = get_path(payload, field_map.path)
        if value is None:
            continue
        if field_map.values is not None and isinstance(value, str | int | float | bool):
            value = field_map.values.get(str(value), value)
        fields[name] = value
    try:
        return SystemResponse(**fields, raw=payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise MappingError(f"response does not fit response_map: {problems}") from exc


def to_response(value: Any, mapping: ResponseMap) -> SystemResponse:
    """Convert whatever a system returned: SystemResponse as is, str as the answer, dict mapped."""
    if isinstance(value, SystemResponse):
        return value
    if isinstance(value, str):
        return SystemResponse(answer=value)
    if isinstance(value, dict):
        return apply_map(value, mapping)
    raise MappingError(
        f"system returned {type(value).__name__}; expected SystemResponse, str or dict"
    )
