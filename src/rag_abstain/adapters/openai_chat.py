"""The `openai_chat` adapter (DESIGN section 5.4): a bare model, the no-retrieval baseline.

Calls any OpenAI-compatible `/chat/completions` endpoint with httpx (no SDK). Returns no
sources or distances, and never a decline stage: every decline it makes is the model's own.
"""

from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, model_validator

from rag_abstain.adapters.base import Secrets, SystemConfig
from rag_abstain.adapters.http import QUESTION_SLOT, post_json
from rag_abstain.adapters.mapping import get_path
from rag_abstain.models import Question, SystemResponse


class OpenAIChatConfig(SystemConfig):
    """System YAML keys for the openai_chat adapter. Give `base_url_env` or `base_url`."""

    adapter: Literal["openai_chat"]
    base_url_env: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    model: str = Field(min_length=1)
    prompt_template: str = QUESTION_SLOT
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    retries: int = Field(default=1, ge=0)
    retry_backoff_s: float = Field(default=3.0, ge=0)

    @model_validator(mode="after")
    def _one_base_url(self) -> "OpenAIChatConfig":
        if (self.base_url_env is None) == (self.base_url is None):
            raise ValueError("give exactly one of base_url_env or base_url")
        return self


class OpenAIChatAdapter:
    """Sends `prompt_template` (with the question filled in) as a single user message."""

    def __init__(
        self,
        url: str,
        body: dict[str, Any],
        prompt_template: str,
        client: httpx.Client,
        retries: int = 1,
        retry_backoff_s: float = 3.0,
    ) -> None:
        """Wrap a configured client. `body` holds every request field except `messages`."""
        self.url = url
        self.body = body
        self.prompt_template = prompt_template
        self.client = client
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        base_dir: Path,
        secrets: Secrets,
        transport: httpx.BaseTransport | None = None,
    ) -> "OpenAIChatAdapter":
        """Build from a system YAML, reading the base URL and API key from the environment.

        `transport` replaces the network (tests pass an httpx.MockTransport).
        """
        cfg = OpenAIChatConfig(**config)
        if cfg.base_url_env is not None:
            base_url = secrets.env(cfg.base_url_env, "base URL")
        else:
            base_url = str(cfg.base_url)
        headers = {}
        if cfg.api_key_env is not None:
            headers["Authorization"] = f"Bearer {secrets.env(cfg.api_key_env, 'API key')}"
        body: dict[str, Any] = {"model": cfg.model}
        for key in ("temperature", "seed", "max_tokens"):
            value = getattr(cfg, key)
            if value is not None:
                body[key] = value
        client = httpx.Client(headers=headers, timeout=cfg.timeout_s, transport=transport)
        return cls(
            base_url.rstrip("/") + "/chat/completions",
            body,
            cfg.prompt_template,
            client,
            cfg.retries,
            cfg.retry_backoff_s,
        )

    def __call__(self, question: Question) -> SystemResponse:
        """Ask the model; failures and replies without message content become error responses."""
        prompt = self.prompt_template.replace(QUESTION_SLOT, question.question)
        body = {**self.body, "messages": [{"role": "user", "content": prompt}]}
        payload, error = post_json(self.client, self.url, body, self.retries, self.retry_backoff_s)
        if payload is None:
            return SystemResponse(error=error)
        content = get_path(payload, "choices.0.message.content")
        if not isinstance(content, str):
            return SystemResponse(error="reply has no choices[0].message.content", raw=payload)
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return SystemResponse(
            answer=content,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            raw=payload,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self.client.close()


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
