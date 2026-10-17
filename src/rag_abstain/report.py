"""Run summaries and reports (DESIGN section 9): summary.json and report.md for a run directory.

The summary holds every metric in section 7, computed from the latest record for each
(qid, repeat). The Markdown report renders it as tables, overall and per category.
"""

import json
from pathlib import Path
from typing import Any

from rag_abstain.metrics.abstention import abstention_metrics
from rag_abstain.metrics.correctness import correctness_metrics
from rag_abstain.metrics.cost import LATENCY_GROUPS, cost_metrics
from rag_abstain.metrics.determinism import determinism_metrics
from rag_abstain.metrics.retrieval import K_VALUES, retrieval_metrics
from rag_abstain.models import RunManifest
from rag_abstain.runner import read_manifest, read_records

SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"
ERROR_CHARS = 200

BARE_MODEL_CAVEAT = (
    "Baseline caveat: this system is a bare model that got the bare question with no "
    "instruction to decline. Comparing it with a grounded pipeline compares the whole pipeline "
    "against the bare model, not the prompt alone."
)


def caveats(system: dict[str, Any]) -> list[str]:
    """Caveats a report must print for this system config (DESIGN section 5.4)."""
    found = []
    if system.get("adapter") == "openai_chat":
        template = system.get("prompt_template", "{question}")
        if isinstance(template, str) and template.strip() == "{question}":
            found.append(BARE_MODEL_CAVEAT)
    return found


def summarise(run_dir: str | Path) -> dict[str, Any]:
    """Compute the summary of a run directory: run metadata, every metric, errors, caveats."""
    manifest: RunManifest = read_manifest(run_dir)
    records = read_records(run_dir)
    errors = [
        {
            "qid": r.qid,
            "repeat": r.repeat,
            "error": (r.response.error or "")[:ERROR_CHARS],
        }
        for r in sorted(records, key=lambda r: (r.repeat, r.qid))
        if r.outcome == "error"
    ]
    return {
        "run_id": manifest.run_id,
        "created_at": manifest.created_at.isoformat(),
        "harness_version": manifest.harness_version,
        "system": {
            "name": manifest.system.get("name"),
            "adapter": manifest.system.get("adapter"),
        },
        "dataset": {"path": manifest.dataset_path, "sha256": manifest.dataset_sha256},
        "repeats": manifest.repeats,
        "limit": manifest.limit,
        "classifier": manifest.classifier.get("name"),
        "records": len(records),
        "errors": errors,
        "caveats": caveats(manifest.system),
        "abstention": abstention_metrics(records),
        "correctness": correctness_metrics(records),
        "retrieval": retrieval_metrics(records),
        "determinism": determinism_metrics(records),
        "cost": cost_metrics(records),
    }


def fmt(value: Any) -> str:
    """Format a metric value for a table: n/a for None, three decimals for floats."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def cell(text: str) -> str:
    """Make text safe inside a Markdown table cell (no pipes, no line breaks)."""
    return " ".join(text.split()).replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _groups(section: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The overall block, then one block per category."""
    return [("**all**", section), *section["by_category"].items()]


def _abstention(summary: dict[str, Any]) -> list[str]:
    section = summary["abstention"]
    rows = []
    for name, block in _groups(section):
        rows.append(
            [
                name,
                fmt(block["n"]),
                fmt(block["errors"]),
                *(fmt(block["outcomes"][o]) for o in ("answered", "declined", "hedged")),
                *(
                    fmt(block[m])
                    for m in (
                        "unsupported_answer_rate",
                        "false_decline_rate",
                        "hedge_rate",
                        "decline_precision",
                        "decline_recall",
                        "decline_f1",
                        "balanced_accuracy",
                    )
                ),
            ]
        )
    header = [
        "",
        "n",
        "errors",
        "answered",
        "declined",
        "hedged",
        "unsupported answer rate",
        "false decline rate",
        "hedge rate",
        "decline precision",
        "decline recall",
        "decline F1",
        "balanced accuracy",
    ]
    stages = section["declines_by_stage"]
    return [
        "## Abstention (repeat 1)",
        "",
        "Unsupported answer rate: answered or hedged, over questions expected to be declined. "
        "False decline rate: declined, over questions expected to be answered.",
        "",
        table(header, rows),
        "",
        f"Declines by stage: gate {stages['gate']}, model {stages['model']}.",
    ]


def _correctness(summary: dict[str, Any]) -> list[str]:
    rows = []
    for name, block in _groups(summary["correctness"]):
        if block["all_answerable"]["n"] == 0:
            continue
        for view, label in (("answered_only", "answered only"), ("all_answerable", "all")):
            v = block[view]
            rows.append(
                [
                    name,
                    label,
                    fmt(v["n"]),
                    fmt(v["exact_match"]),
                    fmt(v["f1"]),
                    fmt(v["key_term_hit"]),
                ]
            )
    if not rows:
        return ["## Correctness (repeat 1)", "", "No non-error questions expected to be answered."]
    return [
        "## Correctness (repeat 1)",
        "",
        'Over questions expected to be answered. "answered only" excludes hedged and declined '
        'answers; "all" scores them 0.',
        "",
        table(["", "view", "n", "exact match", "F1", "key-term hit"], rows),
    ]


def _retrieval(summary: dict[str, Any]) -> list[str]:
    section = summary["retrieval"]
    if section["n_scored"] == 0:
        return [
            "## Retrieval (repeat 1)",
            "",
            "Not scored: no record has both gold sources and returned sources.",
        ]
    names = [f"hit_at_{k}" for k in K_VALUES] + [f"recall_at_{k}" for k in K_VALUES] + ["mrr"]
    header = ["", "scored", *(f"hit@{k}" for k in K_VALUES), *(f"recall@{k}" for k in K_VALUES)]
    rows = [
        [name, fmt(block["n_scored"]), *(fmt(block[m]) for m in names)]
        for name, block in _groups(section)
    ]
    return ["## Retrieval (repeat 1)", "", table([*header, "MRR"], rows)]


def _determinism(summary: dict[str, Any]) -> list[str]:
    section = summary["determinism"]
    lines = ["## Determinism (all repeats)", ""]
    if section["n_compared"] == 0:
        return [*lines, "Not measured: needs 2 or more repeats without errors."]
    rows = [
        [
            name,
            fmt(block["n_compared"]),
            fmt(block["identical_answer_rate"]),
            fmt(block["outcome_agreement_rate"]),
        ]
        for name, block in _groups(section)
    ]
    lines += [
        table(["", "questions compared", "identical answer rate", "outcome agreement rate"], rows),
        "",
    ]
    flips = section["outcome_flips"]
    if flips:
        lines.append("Outcome flips:")
        lines += [f"- {f['qid']}: {' -> '.join(f['outcomes'])}" for f in flips]
    else:
        lines.append("Outcome flips: none.")
    return lines


def _cost(summary: dict[str, Any]) -> list[str]:
    section = summary["cost"]
    labels = {
        "all": "all",
        "gate": "gate declines",
        "model": "model declines",
        "no_decline": "not declined",
    }
    rows = [
        [
            labels[g],
            fmt(section["latency"][g]["n"]),
            fmt(section["latency"][g]["median_s"]),
            fmt(section["latency"][g]["p95_s"]),
        ]
        for g in LATENCY_GROUPS
    ]
    return [
        "## Cost (all repeats)",
        "",
        table(["latency", "n", "median (s)", "p95 (s)"], rows),
        "",
        f"Prompt tokens: {fmt(section['prompt_tokens_total'])}. "
        f"Completion tokens: {fmt(section['completion_tokens_total'])}. "
        f"Error records: {section['errors']}.",
    ]


def _notes(summary: dict[str, Any]) -> list[str]:
    notes = []
    for section in ("abstention", "correctness", "retrieval", "determinism", "cost"):
        for name, reason in summary[section]["notes"].items():
            notes.append(f"- {section}.{name}: {reason}")
    return ["## Notes", "", *notes] if notes else []


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a summary as a Markdown report."""
    system = summary["system"]
    errors = summary["errors"]
    lines = [
        f"# Run report: {summary['run_id']}",
        "",
        f"- System: `{system['name']}` (adapter `{system['adapter']}`)",
        f"- Dataset: `{summary['dataset']['path']}` (SHA-256 {summary['dataset']['sha256'][:12]})",
        f"- Repeats: {summary['repeats']}"
        + (f", first {summary['limit']} questions" if summary["limit"] else ""),
        f"- Classifier: {summary['classifier']}",
        f"- Records: {summary['records']}, errors: {len(errors)}",
        f"- Created: {summary['created_at']}, rag-abstain {summary['harness_version']}",
        "",
    ]
    for caveat in summary["caveats"]:
        lines += [f"> {caveat}", ""]
    for part in (_abstention, _correctness, _retrieval, _determinism, _cost):
        lines += [*part(summary), ""]
    lines += ["## Errors", ""]
    if errors:
        lines.append(
            table(
                ["qid", "repeat", "error"],
                [[e["qid"], str(e["repeat"]), cell(e["error"])] for e in errors],
            )
        )
    else:
        lines.append("None.")
    lines.append("")
    notes = _notes(summary)
    if notes:
        lines += [*notes, ""]
    return "\n".join(lines)


def render_json(data: dict[str, Any]) -> str:
    """Render a summary (or any result dict) as indented JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_reports(run_dir: str | Path) -> dict[str, Any]:
    """Compute the summary and write summary.json and report.md into the run directory."""
    run_dir = Path(run_dir)
    summary = summarise(run_dir)
    (run_dir / SUMMARY_FILE).write_text(render_json(summary), encoding="utf-8")
    (run_dir / REPORT_FILE).write_text(render_markdown(summary), encoding="utf-8")
    return summary
