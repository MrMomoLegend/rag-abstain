"""Retrieval metrics (DESIGN section 7): did the system retrieve the gold source documents?

Returned sources are reduced to basenames, lowercased and de-duplicated in rank order before
comparing with `gold_sources`, so "docs/CYK algorithm.txt" twice counts once as rank 1.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from rag_abstain.metrics import mean
from rag_abstain.metrics.abstention import UNCATEGORISED
from rag_abstain.models import Record

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
SCORE_NAMES: tuple[str, ...] = (
    *(f"hit_at_{k}" for k in K_VALUES),
    *(f"recall_at_{k}" for k in K_VALUES),
    "mrr",
)


def basename(source: str) -> str:
    """Last path part of a source, lowercased, with either slash style."""
    return source.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip().lower()


def unique_sources(sources: Iterable[str]) -> list[str]:
    """Basenames in rank order, first occurrence kept, blanks dropped."""
    return list(dict.fromkeys(name for name in map(basename, sources) if name))


def hit_at_k(sources: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    """1.0 if any gold source is in the top k unique sources; None with no gold sources."""
    wanted = {basename(g) for g in gold} - {""}
    if not wanted:
        return None
    return float(any(s in wanted for s in unique_sources(sources)[:k]))


def recall_at_k(sources: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    """Share of distinct gold sources found in the top k unique sources; None with no gold."""
    wanted = {basename(g) for g in gold} - {""}
    if not wanted:
        return None
    return len(wanted & set(unique_sources(sources)[:k])) / len(wanted)


def mrr(sources: Sequence[str], gold: Sequence[str]) -> float | None:
    """1 / rank of the first gold source among unique sources, 0.0 if none; None with no gold."""
    wanted = {basename(g) for g in gold} - {""}
    if not wanted:
        return None
    for rank, source in enumerate(unique_sources(sources), start=1):
        if source in wanted:
            return 1 / rank
    return 0.0


def score_retrieval(sources: Sequence[str], gold: Sequence[str]) -> dict[str, float | None]:
    """Per-record retrieval scores. Empty when there is nothing to score.

    Nothing is scored when the question has no gold sources, or the system returned no
    sources at all (a bare model, or a system that does not expose them): 0 would claim it
    retrieved the wrong documents, when it reported none.
    """
    if not gold or not sources:
        return {}
    scores: dict[str, float | None] = {}
    for k in K_VALUES:
        scores[f"hit_at_{k}"] = hit_at_k(sources, gold, k)
        scores[f"recall_at_{k}"] = recall_at_k(sources, gold, k)
    scores["mrr"] = mrr(sources, gold)
    return scores


def _compute(records: list[Record]) -> dict[str, Any]:
    notes: dict[str, str] = {}
    ok = [r for r in records if r.outcome != "error"]
    scored = [r for r in ok if r.scores.get("mrr") is not None]
    result: dict[str, Any] = {"n_scored": len(scored), "n_unscored": len(ok) - len(scored)}
    for name in SCORE_NAMES:
        result[name] = mean(
            (r.scores[name] for r in scored),
            name,
            notes,
            "no non-error records with both gold_sources and returned sources",
        )
    result["notes"] = notes
    return result


def retrieval_metrics(records: Iterable[Record]) -> dict[str, Any]:
    """Mean retrieval scores over repeat 1, overall and per category.

    Reads the scores the runner stored on each record (see `score_retrieval`). Covers every
    non-error record, declined ones included, since retrieval happens before any decline.
    `n_unscored` counts non-error records without gold sources or without returned sources.
    """
    first = [r for r in records if r.repeat == 1]
    result = _compute(first)
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in first:
        groups[r.category or UNCATEGORISED].append(r)
    result["by_category"] = {name: _compute(groups[name]) for name in sorted(groups)}
    return result
