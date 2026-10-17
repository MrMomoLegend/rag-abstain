"""Determinism metrics (DESIGN section 7): does the same question get the same result twice?

Uses every repeat. A question is compared when it has at least two records and none of them
is an error; questions with an error in any repeat are listed in `error_qids` instead.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rag_abstain.metrics import ratio
from rag_abstain.metrics.abstention import UNCATEGORISED
from rag_abstain.models import Record


def _compute(records: list[Record]) -> dict[str, Any]:
    notes: dict[str, str] = {}
    by_qid: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_qid[r.qid].append(r)

    compared: list[list[Record]] = []
    error_qids: list[str] = []
    for qid in sorted(by_qid):
        runs = sorted(by_qid[qid], key=lambda r: r.repeat)
        if len(runs) < 2:
            continue
        if any(r.outcome == "error" for r in runs):
            error_qids.append(qid)
        else:
            compared.append(runs)

    identical = sum(len({r.response.answer for r in runs}) == 1 for runs in compared)
    flips = [
        {"qid": runs[0].qid, "outcomes": [r.outcome for r in runs]}
        for runs in compared
        if len({r.outcome for r in runs}) > 1
    ]
    reason = "no question has two or more non-error repeats"
    return {
        "n_compared": len(compared),
        "error_qids": error_qids,
        "identical_answer_rate": ratio(
            identical, len(compared), "identical_answer_rate", notes, reason
        ),
        "outcome_agreement_rate": ratio(
            len(compared) - len(flips), len(compared), "outcome_agreement_rate", notes, reason
        ),
        "outcome_flips": flips,
        "notes": notes,
    }


def determinism_metrics(records: Iterable[Record]) -> dict[str, Any]:
    """Answer and outcome agreement across repeats, overall and per category.

    `identical_answer_rate` compares answer text byte for byte; `outcome_flips` lists each
    question whose outcome changed, with its outcomes in repeat order. With a single repeat
    every rate is None with a note.
    """
    items = list(records)
    result = _compute(items)
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in items:
        groups[r.category or UNCATEGORISED].append(r)
    result["by_category"] = {name: _compute(groups[name]) for name in sorted(groups)}
    return result
