"""Core data models (DESIGN sections 4, 5.1 and 9)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Expected = Literal["answer", "decline"]
Outcome = Literal["answered", "declined", "hedged", "error"]
DeclineStage = Literal["gate", "model"]


class Question(BaseModel):
    """One labelled question from a question set."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected: Expected
    category: str | None = None
    references: list[str] = []
    key_terms: list[list[str]] = []
    gold_sources: list[str] = []
    notes: str | None = None

    @field_validator("expected", mode="before")
    @classmethod
    def _normalise_expected(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("category", "notes", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("references", "gold_sources", mode="after")
    @classmethod
    def _drop_blank_items(cls, value: list[str]) -> list[str]:
        return [item for item in value if item]

    @field_validator("key_terms", mode="before")
    @classmethod
    def _groups_from_strings(cls, value: Any) -> Any:
        # A bare string inside the list is a group with a single alternative.
        if isinstance(value, list):
            return [[item] if isinstance(item, str) else item for item in value]
        return value

    @field_validator("key_terms", mode="after")
    @classmethod
    def _drop_blank_groups(cls, value: list[list[str]]) -> list[list[str]]:
        groups = [[alt.strip() for alt in group if alt.strip()] for group in value]
        return [group for group in groups if group]


class SystemResponse(BaseModel):
    """What a system returned for one question, in the common shape every adapter produces."""

    answer: str = ""
    declined: bool | None = None
    decline_stage: DeclineStage | None = None
    sources: list[str] = []
    contexts: list[str] = []
    distances: list[float] = []
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = {}
    error: str | None = None


class Record(BaseModel):
    """One (qid, repeat) result: the question, the response, the resolved outcome and scores."""

    qid: str
    repeat: int = Field(default=1, ge=1)
    question: str
    expected: Expected
    category: str | None = None
    response: SystemResponse
    outcome: Outcome
    decline_stage: DeclineStage | None = None
    scores: dict[str, float | None] = {}


class RunManifest(BaseModel):
    """Run metadata written to manifest.json. `system` must already have secrets removed."""

    run_id: str
    created_at: datetime
    harness_version: str
    system: dict[str, Any]
    dataset_path: str
    dataset_sha256: str
    repeats: int = Field(ge=1)
    limit: int | None = Field(default=None, ge=1)
    classifier: dict[str, Any]
    python_version: str
