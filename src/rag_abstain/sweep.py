"""Threshold sweep (DESIGN section 8): replay a distance gate offline on a gate-off run.

In a gate-off run every question reached the model. For a threshold t, a question is
`declined (gate)` if its nearest distance is above t, otherwise it keeps its recorded outcome.
Results are in-sample: a threshold picked here must be validated on another question set.
"""

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_abstain.metrics.abstention import abstention_metrics
from rag_abstain.models import Record
from rag_abstain.report import fmt, table
from rag_abstain.runner import read_manifest, read_records

DEFAULT_GRID = "0.20:1.20:0.05"
IN_SAMPLE_NOTE = (
    "In-sample: these thresholds are scored on the same questions they would be chosen from. "
    "Validate a chosen threshold on a different question set before reporting it as a result."
)


class SweepError(Exception):
    """Raised for a bad grid, or a run the sweep cannot replay."""


def parse_grid(spec: str) -> list[float]:
    """Parse `start:stop:step` into an inclusive list of thresholds, rounded to 6 places."""
    try:
        start, stop, step = (float(part) for part in spec.split(":"))
    except ValueError as exc:
        raise SweepError(f"grid '{spec}' must be start:stop:step, e.g. {DEFAULT_GRID}") from exc
    if step <= 0 or stop < start:
        raise SweepError(f"grid '{spec}' needs step > 0 and stop >= start")
    count = math.floor((stop - start) / step + 1e-9) + 1
    return [round(start + i * step, 6) for i in range(count)]


def gate(records: Iterable[Record], threshold: float) -> list[Record]:
    """Apply a gate at `threshold`: nearest distance above it becomes a gate decline."""
    gated = []
    for r in records:
        if r.response.distances and r.response.distances[0] > threshold:
            r = r.model_copy(update={"outcome": "declined", "decline_stage": "gate"})
        gated.append(r)
    return gated


def margin(records: Iterable[Record]) -> dict[str, Any]:
    """Largest nearest distance among expected=answer vs smallest among expected=decline.

    If the first is below the second, any threshold t with max_answer <= t < min_decline
    gates every expected=decline question and no expected=answer question.
    """
    items = list(records)
    answer = [r.response.distances[0] for r in items if r.expected == "answer"]
    decline = [r.response.distances[0] for r in items if r.expected == "decline"]
    max_answer = max(answer) if answer else None
    min_decline = min(decline) if decline else None
    if max_answer is None or min_decline is None:
        return {
            "max_answer": max_answer,
            "min_decline": min_decline,
            "margin": None,
            "separable": None,
            "note": "needs questions of both expected classes",
        }
    return {
        "max_answer": max_answer,
        "min_decline": min_decline,
        "margin": round(min_decline - max_answer, 6),
        "separable": max_answer < min_decline,
    }


def sweep_records(records: Iterable[Record], grid: list[float], run_id: str) -> dict[str, Any]:
    """Sweep the grid over repeat-1, non-error records of a gate-off run.

    Raises SweepError if the run already has gate declines (the gate was on) or a non-error
    record has no distances. `best` is the grid point with the highest balanced accuracy
    (the lowest threshold on a tie); it is in-sample.
    """
    first = [r for r in records if r.repeat == 1]
    ok = [r for r in first if r.outcome != "error"]
    gated = sorted(r.qid for r in ok if r.decline_stage == "gate")
    if gated:
        raise SweepError(
            f"{len(gated)} question(s) were declined at the gate ({', '.join(gated[:5])}); "
            "the sweep needs a run made with the gate off"
        )
    missing = sorted(r.qid for r in ok if not r.response.distances)
    if missing:
        raise SweepError(
            f"{len(missing)} record(s) have no distances ({', '.join(missing[:5])}); "
            "the sweep needs a system that exposes nearest-chunk distances"
        )
    if not ok:
        raise SweepError("no non-error records to sweep")

    rows = []
    for t in grid:
        metrics = abstention_metrics(gate(ok, t))
        rows.append(
            {
                "threshold": t,
                "unsupported_answer_rate": metrics["unsupported_answer_rate"],
                "false_decline_rate": metrics["false_decline_rate"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "gate_declines": metrics["declines_by_stage"]["gate"],
            }
        )
    scored = [row for row in rows if row["balanced_accuracy"] is not None]
    best = max(scored, key=lambda row: row["balanced_accuracy"]) if scored else None
    return {
        "run_id": run_id,
        "in_sample": True,
        "note": IN_SAMPLE_NOTE,
        "n": len(ok),
        "errors_excluded": len(first) - len(ok),
        "margin": margin(ok),
        "best": best,
        "rows": rows,
    }


def sweep_run(run_dir: str | Path, grid: list[float]) -> dict[str, Any]:
    """Sweep a run directory (see `sweep_records`)."""
    return sweep_records(read_records(run_dir), grid, read_manifest(run_dir).run_id)


def render_markdown(result: dict[str, Any]) -> str:
    """Render a sweep as Markdown, with the in-sample label first."""
    m = result["margin"]
    lines = [
        f"# Threshold sweep (in-sample): {result['run_id']}",
        "",
        f"> {result['note']}",
        "",
        f"Questions: {result['n']} (errors excluded: {result['errors_excluded']}).",
        "",
        "## Margin",
        "",
        f"- Largest nearest distance, expected=answer: {fmt(m['max_answer'])}",
        f"- Smallest nearest distance, expected=decline: {fmt(m['min_decline'])}",
        f"- Margin: {fmt(m['margin'])}",
    ]
    if m["separable"]:
        lines.append(
            f"- Separable: any threshold t with {fmt(m['max_answer'])} <= t < "
            f"{fmt(m['min_decline'])} gates every expected=decline question and no "
            "expected=answer question."
        )
    elif m["separable"] is False:
        lines.append("- Not separable: no threshold gates exactly the expected=decline questions.")
    best = result["best"]
    if best is not None:
        lines += [
            "",
            f"Highest balanced accuracy on this set: {fmt(best['balanced_accuracy'])} "
            f"at t = {best['threshold']:.2f} (in-sample).",
        ]
    rows = [
        [
            f"{row['threshold']:.2f}",
            fmt(row["unsupported_answer_rate"]),
            fmt(row["false_decline_rate"]),
            fmt(row["balanced_accuracy"]),
            fmt(row["gate_declines"]),
        ]
        for row in result["rows"]
    ]
    header = [
        "threshold",
        "unsupported answer rate",
        "false decline rate",
        "balanced accuracy",
        "gate declines",
    ]
    lines += ["", "## Grid", "", table(header, rows), ""]
    return "\n".join(lines)


def plot_sweep(result: dict[str, Any], path: str | Path) -> None:
    """Plot the trade-off curve to an image file. Needs the `plot` extra (matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SweepError('plotting needs matplotlib: pip install "rag-abstain[plot]"') from exc

    rows = result["rows"]
    xs = [row["threshold"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in (
        ("unsupported_answer_rate", "unsupported answer rate"),
        ("false_decline_rate", "false decline rate"),
        ("balanced_accuracy", "balanced accuracy"),
    ):
        ys = [row[key] if row[key] is not None else float("nan") for row in rows]
        ax.plot(xs, ys, marker="o", markersize=3, label=label)
    m = result["margin"]
    if m["separable"]:
        ax.axvspan(m["max_answer"], m["min_decline"], alpha=0.15, label="separating range")
    ax.set_xlabel("distance threshold t (declined at the gate if nearest distance > t)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Threshold sweep (in-sample) {result['run_id']}")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
