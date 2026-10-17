"""Cost metrics (DESIGN section 7): latency, token totals and errors.

Unlike the rates, cost uses every repeat: it describes what the whole run spent. Latency
excludes error records (a timeout is not a response time) and is split by decline stage,
because gate declines skip the model and are near-instant.
"""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rag_abstain.metrics.abstention import UNCATEGORISED
from rag_abstain.models import Record

LATENCY_GROUPS: tuple[str, ...] = ("all", "gate", "model", "no_decline")


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (`pct` in 0..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency(values: list[float], name: str, notes: dict[str, str]) -> dict[str, Any]:
    if not values:
        notes[f"latency.{name}"] = "no non-error records with a latency in this group"
        return {"n": 0, "median_s": None, "p95_s": None}
    return {
        "n": len(values),
        "median_s": statistics.median(values),
        "p95_s": percentile(values, 95),
    }


def _tokens(records: list[Record], field: str, notes: dict[str, str]) -> int | None:
    values = [getattr(r.response, field) for r in records]
    reported = [v for v in values if v is not None]
    if not reported:
        notes[f"{field}_total"] = "no record reported this count"
        return None
    if len(reported) < len(values):
        notes[f"{field}_total"] = f"{len(values) - len(reported)} record(s) did not report it"
    return sum(reported)


def _compute(records: list[Record]) -> dict[str, Any]:
    notes: dict[str, str] = {}
    timed = [r for r in records if r.outcome != "error" and r.response.latency_s is not None]
    groups = {
        "all": timed,
        "gate": [r for r in timed if r.decline_stage == "gate"],
        "model": [r for r in timed if r.decline_stage == "model"],
        "no_decline": [r for r in timed if r.decline_stage is None],
    }
    return {
        "n_records": len(records),
        "errors": sum(r.outcome == "error" for r in records),
        "latency": {
            name: _latency([float(r.response.latency_s or 0.0) for r in groups[name]], name, notes)
            for name in LATENCY_GROUPS
        },
        "prompt_tokens_total": _tokens(records, "prompt_tokens", notes),
        "completion_tokens_total": _tokens(records, "completion_tokens", notes),
        "notes": notes,
    }


def cost_metrics(records: Iterable[Record]) -> dict[str, Any]:
    """Latency median and p95 by decline stage, token totals and error count, all repeats.

    Token totals are None (with a note) when no record reported tokens; a partial count is
    summed with a note saying how many records were missing it.
    """
    items = list(records)
    result = _compute(items)
    by_category: dict[str, list[Record]] = defaultdict(list)
    for r in items:
        by_category[r.category or UNCATEGORISED].append(r)
    result["by_category"] = {name: _compute(by_category[name]) for name in sorted(by_category)}
    return result
