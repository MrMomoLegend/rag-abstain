"""The `http` adapter (DESIGN section 5.3): POST JSON to an endpoint, map the JSON reply.

Transport failures, non-2xx statuses and bodies that are not a JSON object become a response
with `error` set, never an exception. Transport failures, 429 and 5xx are retried.
"""

import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field

from rag_abstain.adapters.base import Secrets, SystemConfig, SystemConfigError
from rag_abstain.adapters.mapping import MappingError, ResponseMap, apply_map, resolve_map
from rag_abstain.models import Question, SystemResponse

QUESTION_SLOT = "{question}"
ERROR_BODY_CHARS = 200


class HttpConfig(SystemConfig):
    """System YAML keys for the http adapter."""

    adapter: Literal["http"]
    url: str = Field(min_length=1)
    request: dict[str, Any] | str = {"question": QUESTION_SLOT}
    headers: dict[str, str] = {}
    response_map: str | dict[str, Any] | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    retries: int = Field(default=1, ge=0)
    retry_backoff_s: float = Field(default=3.0, ge=0)


def render_request(template: dict[str, Any] | str, question: str) -> Any:
    """Fill `{question}` into a JSON template, JSON-escaped, and parse the result."""
    text = template if isinstance(template, str) else json.dumps(template)
    return json.loads(text.replace(QUESTION_SLOT, json.dumps(question)[1:-1]))


def post_json(
    client: httpx.Client,
    url: str,
    body: Any,
    retries: int,
    backoff_s: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """POST `body` as JSON. Returns (payload, None) on success or (None, error message).

    Transport errors, 429 and 5xx are retried `retries` times with `backoff_s` between tries.
    Other non-2xx statuses, invalid JSON and non-object JSON fail at once.
    """
    error = "no attempt made"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff_s)
        try:
            reply = client.post(url, json=body)
        except httpx.TransportError as exc:
            error = f"{type(exc).__name__}: {exc}"
            continue
        if not reply.is_success:
            error = f"HTTP {reply.status_code}: {reply.text[:ERROR_BODY_CHARS]}"
            if reply.status_code == 429 or reply.status_code >= 500:
                continue
            return None, error
        try:
            payload = reply.json()
        except ValueError:
            return None, f"invalid JSON in response: {reply.text[:ERROR_BODY_CHARS]!r}"
        if not isinstance(payload, dict):
            return None, f"expected a JSON object, got {type(payload).__name__}"
        return payload, None
    return None, error


class HttpAdapter:
    """POSTs each question to `url` and maps the JSON reply through `response_map`."""

    def __init__(
        self,
        url: str,
        request: dict[str, Any] | str,
        response_map: ResponseMap,
        client: httpx.Client,
        retries: int = 1,
        retry_backoff_s: float = 3.0,
    ) -> None:
        """Wrap a configured httpx client (headers and timeout already set)."""
        self.url = url
        self.request = request
        self.response_map = response_map
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
    ) -> "HttpAdapter":
        """Build from a system YAML. `${NAME}` in header values is read from the environment.

        `transport` replaces the network (tests pass an httpx.MockTransport).
        """
        cfg = HttpConfig(**config)
        template_text = cfg.request if isinstance(cfg.request, str) else json.dumps(cfg.request)
        if QUESTION_SLOT not in template_text:
            raise SystemConfigError(f"request template must contain {QUESTION_SLOT}")
        try:
            render_request(cfg.request, "probe")
        except ValueError as exc:
            raise SystemConfigError(f"request template is not valid JSON: {exc}") from exc
        response_map = resolve_map(cfg.response_map)
        headers = {k: secrets.expand(v, f"header {k}") for k, v in cfg.headers.items()}
        client = httpx.Client(headers=headers, timeout=cfg.timeout_s, transport=transport)
        return cls(
            cfg.url,
            cfg.request,
            response_map,
            client,
            cfg.retries,
            cfg.retry_backoff_s,
        )

    def __call__(self, question: Question) -> SystemResponse:
        """POST the question; any failure is returned as an error response."""
        body = render_request(self.request, question.question)
        payload, error = post_json(self.client, self.url, body, self.retries, self.retry_backoff_s)
        if payload is None:
            return SystemResponse(error=error)
        try:
            return apply_map(payload, self.response_map)
        except MappingError as exc:
            return SystemResponse(error=str(exc), raw=payload)

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self.client.close()
