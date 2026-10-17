"""Abstention metrics (DESIGN section 7): does the system decline when, and only when, it should?

Decline is the positive class. Hedged counts as not declined. Error records are excluded from
every rate and counted separately. With repeats, only repeat 1 is used.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rag_abstain.metrics import ratio
from rag_abstain.models import Record

UNCATEGORISED = "uncategorised"


def _compute(records: list[Record]) -> dict[str, Any]:
    notes: dict[str, str] = {}
    errors = [r for r in records if r.outcome == "error"]
    ok = [r for r in records if r.outcome != "error"]
    should_decline = [r for r in ok if r.expected == "decline"]
    should_answer = [r for r in ok if r.expected == "answer"]

    tp = sum(r.outcome == "declined" for r in should_decline)
    fn = len(should_decline) - tp
    fp = sum(r.outcome == "declined" for r in should_answer)
    tn = len(should_answer) - fp

    no_decline_rows = "no non-error questions with expected=decline"
    no_answer_rows = "no non-error questions with expected=answer"

    recall_decline = ratio(tp, len(should_decline), "decline_recall", notes, no_decline_rows)
    recall_answer = ratio(tn, len(should_answer), "answer_recall", notes, no_answer_rows)
    if recall_decline is None or recall_answer is None:
        balanced_accuracy = None
        notes["balanced_accuracy"] = "needs non-error questions of both expected classes"
    else:
        balanced_accuracy = (recall_decline + recall_answer) / 2

    return {
        "n": len(ok),
        "n_expected_answer": len(should_answer),
        "n_expected_decline": len(should_decline),
        "errors": len(errors),
        "error_qids": [r.qid for r in errors],
        "outcomes": {
            outcome: sum(r.outcome == outcome for r in ok)
            for outcome in ("answered", "declined", "hedged")
        },
        "unsupported_answer_rate": ratio(
            fn, len(should_decline), "unsupported_answer_rate", notes, no_decline_rows
        ),
        "false_decline_rate": ratio(
            fp, len(should_answer), "false_decline_rate", notes, no_answer_rows
        ),
        "hedge_rate": ratio(
            sum(r.outcome == "hedged" for r in ok),
            len(ok),
            "hedge_rate",
            notes,
            "no non-error records",
        ),
        "decline_precision": ratio(
            tp, tp + fp, "decline_precision", notes, "the system declined nothing"
        ),
        "decline_recall": recall_decline,
        "decline_f1": ratio(
            2 * tp,
            2 * tp + fp + fn,
            "decline_f1",
            notes,
            "no declines expected and none made",
        ),
        "answer_recall": recall_answer,
        "balanced_accuracy": balanced_accuracy,
        "declines_by_stage": {
            stage: sum(r.outcome == "declined" and r.decline_stage == stage for r in ok)
            for stage in ("gate", "model")
        },
        "notes": notes,
    }


def abstention_metrics(records: Iterable[Record]) -> dict[str, Any]:
    """Compute abstention metrics overall and per category, over repeat 1 only.

    Records without a category are grouped under "uncategorised". `answer_recall` (recall on the
    answer class, 1 - false_decline_rate) is reported because balanced_accuracy is built from it.
    """
    first = [r for r in records if r.repeat == 1]
    result = _compute(first)
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in first:
        groups[r.category or UNCATEGORISED].append(r)
    result["by_category"] = {name: _compute(groups[name]) for name in sorted(groups)}
    return result
