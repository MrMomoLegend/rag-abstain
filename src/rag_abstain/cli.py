"""Command-line interface (DESIGN section 10).

Exit codes: 0 ok, 1 usage or validation error, 2 run finished with errors in some records.
`main` is the console-script entry point; it maps typer's usage errors (exit 2 by default) to 1
so that 2 always means error records.
"""

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from rag_abstain import __version__
from rag_abstain.adapters import SystemConfigError, load_system
from rag_abstain.compare import compare_runs
from rag_abstain.compare import render_markdown as render_compare
from rag_abstain.dataset import DatasetError, validate_dataset
from rag_abstain.report import REPORT_FILE, fmt, render_json, render_markdown, write_reports
from rag_abstain.runner import RunError, run
from rag_abstain.sweep import DEFAULT_GRID, SweepError, parse_grid, plot_sweep, sweep_run
from rag_abstain.sweep import render_markdown as render_sweep

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_RUN_ERRORS = 2

app = typer.Typer(
    help="Refusal-aware evaluation for RAG systems: does your RAG system know when to say no?",
    no_args_is_help=True,
    add_completion=False,
)


class Format(StrEnum):
    """Output format for report, compare and sweep."""

    md = "md"
    json = "json"


RunDir = Annotated[Path, typer.Argument(exists=True, file_okay=False, help="A run directory.")]
FormatOption = Annotated[Format, typer.Option("--format", help="Output format.")]


def fail(message: str) -> typer.Exit:
    """Print an error to stderr and return the exit for a usage or validation error."""
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(EXIT_USAGE)


@app.callback()
def root() -> None:
    """rag-abstain command group."""


@app.command()
def version() -> None:
    """Print the rag-abstain version."""
    typer.echo(__version__)


@app.command("validate-dataset")
def validate_dataset_command(
    dataset: Annotated[Path, typer.Argument(help="Question set, .csv or .jsonl.")],
) -> None:
    """Check a question set: unique qids, valid `expected`, no empty questions."""
    report = validate_dataset(dataset)
    for warning in report.warnings:
        typer.echo(f"warning: {warning}")
    for error in report.errors:
        typer.echo(f"error: {error}", err=True)
    if not report.ok:
        typer.echo(f"{dataset}: {len(report.errors)} error(s)", err=True)
        raise typer.Exit(EXIT_USAGE)
    typer.echo(
        f"{dataset}: OK, {len(report.questions)} questions, {len(report.warnings)} warning(s)"
    )


@app.command("run")
def run_command(
    system: Annotated[Path, typer.Option("--system", help="System YAML.")],
    dataset: Annotated[Path, typer.Option("--dataset", help="Question set, .csv or .jsonl.")],
    repeats: Annotated[
        int | None,
        typer.Option(min=1, help="Times to ask each question [default: 1, or the run's]."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Questions asked in parallel.")] = 1,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N questions.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Where run directories go.")] = Path("runs"),
    resume: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="Continue this run; error records are retried."
        ),
    ] = None,
) -> None:
    """Run a system over a question set and write a run directory with its report."""
    try:
        loaded = load_system(system)
    except SystemConfigError as exc:
        raise fail(str(exc)) from exc
    try:
        result = run(
            loaded,
            dataset,
            repeats=repeats,
            workers=workers,
            limit=limit,
            out_dir=out,
            resume=resume,
        )
    except (DatasetError, RunError, OSError) as exc:
        raise fail(str(exc)) from exc
    finally:
        loaded.close()

    summary = write_reports(result.run_dir)
    abstention = summary["abstention"]
    typer.echo(f"Run: {result.run_dir}")
    typer.echo(
        f"Records: {result.new} new ({result.retried} retried), {result.skipped} skipped; "
        f"errors: {result.errors}"
    )
    typer.echo(
        f"Unsupported answer rate: {fmt(abstention['unsupported_answer_rate'])}  "
        f"False decline rate: {fmt(abstention['false_decline_rate'])}  "
        f"Hedge rate: {fmt(abstention['hedge_rate'])}  "
        f"Balanced accuracy: {fmt(abstention['balanced_accuracy'])}"
    )
    for caveat in summary["caveats"]:
        typer.echo(caveat)
    typer.echo(f"Report: {result.run_dir / REPORT_FILE}")
    if result.errors:
        typer.echo(
            f"{result.errors} record(s) are errors; rerun with --resume {result.run_dir} "
            "to retry them.",
            err=True,
        )
        raise typer.Exit(EXIT_RUN_ERRORS)


@app.command("report")
def report_command(run_dir: RunDir, format: FormatOption = Format.md) -> None:
    """Rewrite summary.json and report.md for a run and print the report."""
    try:
        summary = write_reports(run_dir)
    except (RunError, ValueError) as exc:
        raise fail(str(exc)) from exc
    typer.echo(render_json(summary) if format is Format.json else render_markdown(summary))


@app.command("compare")
def compare_command(run_a: RunDir, run_b: RunDir, format: FormatOption = Format.md) -> None:
    """Show metric deltas between two runs and every question whose outcome changed."""
    try:
        result = compare_runs(run_a, run_b)
    except (RunError, ValueError) as exc:
        raise fail(str(exc)) from exc
    typer.echo(render_json(result) if format is Format.json else render_compare(result))


@app.command("sweep")
def sweep_command(
    run_dir: RunDir,
    grid: Annotated[
        str, typer.Option(help="Thresholds as start:stop:step, inclusive.")
    ] = DEFAULT_GRID,
    plot: Annotated[
        Path | None, typer.Option(help="Also save a plot here (needs the plot extra).")
    ] = None,
    format: FormatOption = Format.md,
) -> None:
    """Replay a distance gate offline over a gate-off run (results are in-sample)."""
    try:
        result = sweep_run(run_dir, parse_grid(grid))
        if plot is not None:
            plot_sweep(result, plot)
    except (SweepError, RunError, ValueError) as exc:
        raise fail(str(exc)) from exc
    typer.echo(render_json(result) if format is Format.json else render_sweep(result))
    if plot is not None:
        typer.echo(f"Plot: {plot}")


def run_cli(argv: list[str] | None = None) -> int:
    """Run the CLI on `argv` (default: sys.argv) and return the exit code."""
    try:
        result = app(args=argv, standalone_mode=False)
    except typer.TyperException as exc:
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        else:
            typer.echo(f"Error: {exc.format_message()}", err=True)
        return EXIT_USAGE
    except typer.Abort:
        typer.echo("Aborted.", err=True)
        return EXIT_USAGE
    return result if isinstance(result, int) else EXIT_OK


def main() -> None:
    """Console-script entry point."""
    sys.exit(run_cli())
