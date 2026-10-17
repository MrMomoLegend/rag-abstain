"""Compare two runs (DESIGN section 9): metric deltas and every question whose outcome changed.

Questions are joined on qid, using repeat 1 as the rates do. An outcome counts as changed when
the outcome or the decline stage differs, so a gate decline that became a model decline shows.
"""

from pathlib import Path
from typing import Any

from rag_abstain.report import cell, fmt, summarise, table
from rag_abstain.runner import read_records

ANSWER_CHARS = 200

# (label, path into the summary)
METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("unsupported answer rate", ("abstention", "unsupported_answer_rate")),
    ("false decline rate", ("abstention", "false_decline_rate")),
    ("hedge rate", ("abstention", "hedge_rate")),
    ("decline F1", ("abstention", "decline_f1")),
    ("balanced accuracy", ("abstention", "balanced_accuracy")),
    ("exact match (all answerable)", ("correctness", "all_answerable", "exact_match")),
    ("F1 (all answerable)", ("correctness", "all_answerable", "f1")),
    ("key-term hit (all answerable)", ("correctness", "all_answerable", "key_term_hit")),
    ("hit@3", ("retrieval", "hit_at_3")),
    ("MRR", ("retrieval", "mrr")),
    ("identical answer rate", ("determinism", "identical_answer_rate")),
    ("outcome agreement rate", ("determinism", "outcome_agreement_rate")),
    ("median latency (s)", ("cost", "latency", "all", "median_s")),
    ("p95 latency (s)", ("cost", "latency", "all", "p95_s")),
    ("prompt tokens", ("cost", "prompt_tokens_total")),
    ("completion tokens", ("cost", "completion_tokens_total")),
    ("error records", ("cost", "errors")),
)


def truncate(text: str, limit: int = ANSWER_CHARS) -> str:
    """Cut text to at most `limit` characters, ending in "..." when cut."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _lookup(summary: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = summary
    for step in path:
        node = node[step]
    return node


def _delta(a: Any, b: Any) -> float | int | None:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return b - a
    return None


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """Metric deltas (B minus A) and the repeat-1 questions whose outcome changed.

    A delta is None when either side is None. `notes` warns when the runs used different
    datasets, since their rates are then over different questions.
    """
    summary_a, summary_b = summarise(run_a), summarise(run_b)
    notes = []
    if summary_a["dataset"]["sha256"] != summary_b["dataset"]["sha256"]:
        notes.append("the runs used different datasets; deltas compare different question sets")
    for side, summary in (("A", summary_a), ("B", summary_b)):
        notes += [f"run {side}: {c}" for c in summary["caveats"]]

    metrics = []
    for label, path in METRICS:
        a, b = _lookup(summary_a, path), _lookup(summary_b, path)
        metrics.append({"metric": label, "a": a, "b": b, "delta": _delta(a, b)})

    first_a = {r.qid: r for r in read_records(run_a) if r.repeat == 1}
    first_b = {r.qid: r for r in read_records(run_b) if r.repeat == 1}
    changed = []
    for qid in [q for q in first_a if q in first_b]:
        ra, rb = first_a[qid], first_b[qid]
        if (ra.outcome, ra.decline_stage) == (rb.outcome, rb.decline_stage):
            continue
        changed.append(
            {
                "qid": qid,
                "category": ra.category,
                "expected": ra.expected,
                "a": _side(ra.outcome, ra.decline_stage, ra.response.answer, ra.response.error),
                "b": _side(rb.outcome, rb.decline_stage, rb.response.answer, rb.response.error),
            }
        )

    return {
        "run_a": summary_a["run_id"],
        "run_b": summary_b["run_id"],
        "system_a": summary_a["system"]["name"],
        "system_b": summary_b["system"]["name"],
        "notes": notes,
        "metrics": metrics,
        "changed": changed,
        "only_in_a": sorted(set(first_a) - set(first_b)),
        "only_in_b": sorted(set(first_b) - set(first_a)),
    }


def _side(outcome: str, stage: str | None, answer: str, error: str | None) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "decline_stage": stage,
        "answer": truncate(error if outcome == "error" and error else answer),
    }


def _label(side: dict[str, Any]) -> str:
    stage = side["decline_stage"]
    return f"{side['outcome']} ({stage})" if stage else side["outcome"]


def render_markdown(result: dict[str, Any]) -> str:
    """Render a comparison as Markdown."""
    lines = [
        f"# Compare: {result['run_a']} (A) vs {result['run_b']} (B)",
        "",
        f"- A: `{result['system_a']}`",
        f"- B: `{result['system_b']}`",
        "",
    ]
    lines += [f"> Note: {note}" for note in result["notes"]]
    if result["notes"]:
        lines.append("")
    rows = [[m["metric"], fmt(m["a"]), fmt(m["b"]), _signed(m["delta"])] for m in result["metrics"]]
    lines += ["## Metrics", "", table(["metric", "A", "B", "B - A"], rows), ""]

    lines += [f"## Changed outcomes (repeat 1): {len(result['changed'])}", ""]
    if result["changed"]:
        lines.append(
            table(
                ["qid", "expected", "A", "B", "answer A", "answer B"],
                [
                    [
                        c["qid"],
                        c["expected"],
                        _label(c["a"]),
                        _label(c["b"]),
                        cell(c["a"]["answer"]),
                        cell(c["b"]["answer"]),
                    ]
                    for c in result["changed"]
                ],
            )
        )
    else:
        lines.append("None.")
    for side in ("a", "b"):
        only = result[f"only_in_{side}"]
        if only:
            lines += ["", f"Only in {side.upper()}: {', '.join(only)}"]
    lines.append("")
    return "\n".join(lines)


def _signed(delta: float | int | None) -> str:
    if delta is None:
        return "n/a"
    if isinstance(delta, float):
        return f"{delta:+.3f}"
    return f"{delta:+d}"
