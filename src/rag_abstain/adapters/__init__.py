"""Adapter registry and `load_system` (DESIGN section 5): from a system YAML to a callable."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import ValidationError

from rag_abstain.adapters.base import Adapter, Secrets, SystemConfigError, redact_config
from rag_abstain.adapters.http import HttpAdapter
from rag_abstain.adapters.mapping import MappingError
from rag_abstain.adapters.openai_chat import OpenAIChatAdapter
from rag_abstain.adapters.python_callable import PythonCallableAdapter

AdapterFactory = Callable[[dict[str, Any], Path, Secrets, httpx.BaseTransport | None], Adapter]

ADAPTERS: dict[str, AdapterFactory] = {
    "python": PythonCallableAdapter.from_config,
    "http": HttpAdapter.from_config,
    "openai_chat": OpenAIChatAdapter.from_config,
}

__all__ = ["ADAPTERS", "Adapter", "System", "SystemConfigError", "load_system"]


@dataclass
class System:
    """A loaded system under test.

    `config` is the YAML as written with literal secrets redacted, safe for the manifest.
    `secrets` holds the values read from the environment, for scrubbing output.
    `extra_markers` are added to the default rule classifier's refusal markers.
    """

    name: str
    adapter: Adapter
    config: dict[str, Any]
    refusal_text: str | None
    secrets: Secrets
    path: Path
    extra_markers: tuple[str, ...] = ()

    def close(self) -> None:
        """Release the adapter's resources."""
        self.adapter.close()


def load_system(path: str | Path, transport: httpx.BaseTransport | None = None) -> System:
    """Read a system YAML and build its adapter.

    Relative paths in the YAML resolve against the YAML file's directory. `transport` is passed
    to HTTP-based adapters in place of the network (tests use httpx.MockTransport).
    Raises SystemConfigError for an unreadable or invalid config or a missing variable.
    """
    path = Path(path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemConfigError(f"{path}: cannot read system config: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemConfigError(f"{path}: system config must be a YAML mapping")

    adapter_name = config.get("adapter")
    if adapter_name not in ADAPTERS:
        known = ", ".join(ADAPTERS)
        raise SystemConfigError(f"{path}: adapter must be one of {known}, got {adapter_name!r}")

    secrets = Secrets()
    try:
        adapter = ADAPTERS[adapter_name](config, path.parent.resolve(), secrets, transport)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}"
            for err in exc.errors()
        )
        raise SystemConfigError(f"{path}: {problems}") from exc
    except (SystemConfigError, MappingError) as exc:
        raise SystemConfigError(f"{path}: {exc}") from exc

    return System(
        name=config["name"],
        adapter=adapter,
        config=redact_config(config),
        refusal_text=config.get("refusal_text"),
        secrets=secrets,
        path=path,
        extra_markers=tuple(config.get("extra_markers") or ()),
    )
