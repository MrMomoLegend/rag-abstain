"""Answer correctness (DESIGN section 7): SQuAD Exact Match and token F1, and key-term hit."""

import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from rag_abstain.metrics import mean
from rag_abstain.metrics.abstention import UNCATEGORISED
from rag_abstain.models import Question, Record

SCORE_NAMES = ("exact_match", "f1", "key_term_hit")

_PUNCTUATION = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalise(text: str) -> str:
    """SQuAD normalisation: lowercase, strip ASCII punctuation, drop a/an/the, squash spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: Sequence[str]) -> float | None:
    """1.0 if the normalised prediction equals any normalised reference; None with no references."""
    if not references:
        return None
    pred = normalise(prediction)
    return float(any(pred == normalise(ref) for ref in references))


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalise(prediction).split()
    ref_tokens = normalise(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    same = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def f1(prediction: str, references: Sequence[str]) -> float | None:
    """Best SQuAD token F1 over the references; None with no references."""
    if not references:
        return None
    return max(_token_f1(prediction, ref) for ref in references)


def _contains_tokens(haystack: list[str], needle: list[str]) -> bool:
    width = len(needle)
    return width > 0 and any(
        haystack[i : i + width] == needle for i in range(len(haystack) - width + 1)
    )


def key_term_hit(prediction: str, key_terms: Sequence[Sequence[str]]) -> float | None:
    """1.0 if every group has an alternative in the prediction; None with no groups.

    Both sides are normalised as in `normalise`, and an alternative must match whole tokens
    ("tf idf" does not match "tf idfs"), so "rag" is not found inside "fragment".
    """
    if not key_terms:
        return None
    tokens = normalise(prediction).split()
    return float(
        all(
            any(_contains_tokens(tokens, normalise(alt).split()) for alt in group)
            for group in key_terms
        )
    )


def score_answer(question: Question, answer: str) -> dict[str, float | None]:
    """Per-record scores. A score is None only when the question has nothing to score against.

    Score every non-error record of an `expected = answer` question this way, whatever its
    outcome; `correctness_metrics` decides how declined and hedged records count.
    """
    return {
        "exact_match": exact_match(answer, question.references),
        "f1": f1(answer, question.references),
        "key_term_hit": key_term_hit(answer, question.key_terms),
    }


def _views(records: list[Record]) -> dict[str, Any]:
    notes: dict[str, str] = {}
    answerable = [r for r in records if r.expected == "answer" and r.outcome != "error"]
    answered = [r for r in answerable if r.outcome == "answered"]
    views: dict[str, Any] = {
        "answered_only": {"n": len(answered), "scored": {}},
        "all_answerable": {"n": len(answerable), "scored": {}},
    }
    for name in SCORE_NAMES:
        only = [r.scores[name] for r in answered if r.scores.get(name) is not None]
        everything = [
            r.scores[name] if r.outcome == "answered" else 0.0
            for r in answerable
            if r.scores.get(name) is not None
        ]
        for view, values, reason in (
            ("answered_only", only, "no answered expected=answer records with this score"),
            ("all_answerable", everything, "no expected=answer records with this score"),
        ):
            views[view][name] = mean(values, f"{view}.{name}", notes, reason)
            views[view]["scored"][name] = len(values)
    views["notes"] = notes
    return views


def correctness_metrics(records: Iterable[Record]) -> dict[str, Any]:
    """Mean correctness over `expected = answer` records (repeat 1), overall and per category.

    Two views: `answered_only` (outcome answered; hedged and declined excluded) and
    `all_answerable` (hedged and declined score 0). Errors are excluded from both. `scored`
    gives how many records each mean is over.
    """
    first = [r for r in records if r.repeat == 1]
    result = _views(first)
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in first:
        groups[r.category or UNCATEGORISED].append(r)
    result["by_category"] = {name: _views(groups[name]) for name in sorted(groups)}
    return result
