"""The `python` adapter (DESIGN section 5.2): call `module:function` in this process."""

import importlib
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field

from rag_abstain.adapters.base import Secrets, SystemConfig, SystemConfigError, resolve_path
from rag_abstain.adapters.mapping import ResponseMap, resolve_map, to_response
from rag_abstain.models import Question, SystemResponse

_TARGET = r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$"


class PythonConfig(SystemConfig):
    """System YAML keys for the python adapter."""

    adapter: Literal["python"]
    target: str = Field(pattern=_TARGET)
    pythonpath: str | list[str] | None = None
    params: dict[str, Any] = {}
    response_map: str | dict[str, Any] | None = None


class PythonCallableAdapter:
    """Calls `function(question, **params)` and converts the return value to a SystemResponse."""

    def __init__(
        self,
        function: Callable[..., Any],
        params: dict[str, Any] | None = None,
        response_map: ResponseMap | None = None,
    ) -> None:
        """Wrap an already-imported function."""
        self.function = function
        self.params = dict(params or {})
        self.response_map = response_map if response_map is not None else resolve_map(None)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        base_dir: Path,
        secrets: Secrets,
        transport: httpx.BaseTransport | None = None,
    ) -> "PythonCallableAdapter":
        """Build from a system YAML: put `pythonpath` on sys.path, then import `target`.

        `pythonpath` entries are relative to the YAML file's directory. `secrets` and
        `transport` are unused; they are part of the common factory signature.
        """
        cfg = PythonConfig(**config)
        paths = [cfg.pythonpath] if isinstance(cfg.pythonpath, str) else cfg.pythonpath or []
        for entry in reversed(paths):
            resolved = str(resolve_path(base_dir, entry))
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        return cls(import_target(cfg.target), cfg.params, resolve_map(cfg.response_map))

    def __call__(self, question: Question) -> SystemResponse:
        """Call the function with the question text. Exceptions propagate to the runner."""
        return to_response(self.function(question.question, **self.params), self.response_map)

    def close(self) -> None:
        """Nothing to release."""


def import_target(target: str) -> Callable[..., Any]:
    """Import `module:function` and return the function."""
    if not re.match(_TARGET, target):
        raise SystemConfigError(f"target '{target}' must look like module:function")
    module_name, func_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemConfigError(f"cannot import '{module_name}': {exc}") from exc
    function = getattr(module, func_name, None)
    if not callable(function):
        raise SystemConfigError(f"'{func_name}' in module '{module_name}' is not callable")
    return function
