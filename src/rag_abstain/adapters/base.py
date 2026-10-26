"""The adapter protocol and what every system config shares (DESIGN section 5)."""

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag_abstain.models import Question, SystemResponse

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class SystemConfigError(Exception):
    """Raised when a system YAML is invalid or needs an environment variable that is not set."""


class Adapter(Protocol):
    """Sends one question to a system and returns its response in the common shape.

    Exceptions may escape `__call__`; the runner turns them into `error` records.
    """

    def __call__(self, question: Question) -> SystemResponse:
        """Ask the system one question."""
        ...

    def close(self) -> None:
        """Release any resources (HTTP connections). Safe to call more than once."""
        ...


class SystemConfig(BaseModel):
    """Keys every system YAML has. Each adapter extends this with its own keys."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    adapter: str
    refusal_text: str | None = None
    extra_markers: list[str] = []  # added to the rule classifier's refusal markers


class Secrets:
    """Values read from the environment, so the runner can scrub them from anything it writes."""

    def __init__(self) -> None:
        self.values: set[str] = set()

    def env(self, name: str, purpose: str) -> str:
        """Read environment variable `name`, remember its value as secret, and return it."""
        value = os.environ.get(name)
        if not value:
            raise SystemConfigError(f"environment variable {name} ({purpose}) is not set")
        self.values.add(value)
        return value

    def expand(self, text: str, purpose: str) -> str:
        """Replace every `${NAME}` in `text` with that environment variable's value."""
        return _ENV_REF.sub(lambda m: self.env(m.group(1), purpose), text)

    def scrub(self, text: str) -> str:
        """Replace every known secret in `text` with `<redacted>`, raw and JSON-escaped forms."""
        forms = {form for value in self.values for form in (value, json.dumps(value)[1:-1])}
        for form in sorted(forms, key=len, reverse=True):
            text = text.replace(form, "<redacted>")
        return text


# Matched against the last part of a key name, so `x-api-key` and `access_token` match but
# `prompt_tokens` and `max_tokens` do not.
_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|apikey|key|token|secret|password|passwd|authorization|auth|cookie)$",
    re.IGNORECASE,
)


def redact_config(config: Any, key: str = "") -> Any:
    """Copy a system config for the manifest with literal secrets replaced by `<redacted>`.

    `${NAME}` references and `*_env` keys hold variable names, not values, so they are kept.
    Any other string under a key that looks secret (api_key, token, Authorization, ...) is
    replaced, in case someone typed a key straight into the YAML.
    """
    if isinstance(config, dict):
        return {k: redact_config(v, str(k)) for k, v in config.items()}
    if isinstance(config, list):
        return [redact_config(v, key) for v in config]
    if (
        isinstance(config, str)
        and _SECRET_KEY.search(key)
        and not key.lower().endswith("_env")
        and _ENV_REF.sub("", config).strip(" ") not in ("", "Bearer", "Basic", "Token")
    ):
        return "<redacted>"
    return config


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve a path from a system YAML relative to the YAML file's directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()
