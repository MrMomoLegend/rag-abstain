"""Metrics (DESIGN section 7). Every rate that would divide by zero is None, with a note."""

from collections.abc import Iterable


def ratio(
    numerator: float, denominator: float, name: str, notes: dict[str, str], reason: str
) -> float | None:
    """Return numerator / denominator, or None and record `reason` under `name` in `notes`."""
    if denominator == 0:
        notes[name] = reason
        return None
    return numerator / denominator


def mean(values: Iterable[float], name: str, notes: dict[str, str], reason: str) -> float | None:
    """Return the mean of `values`, or None and record `reason` if there are none."""
    items = list(values)
    return ratio(sum(items), len(items), name, notes, reason)
