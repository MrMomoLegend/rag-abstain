"""Command-line interface (typer app)."""

import typer

from rag_abstain import __version__

app = typer.Typer(
    help="Refusal-aware evaluation for RAG systems: does your RAG system know when to say no?",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """rag-abstain command group."""


@app.command()
def version() -> None:
    """Print the rag-abstain version."""
    typer.echo(__version__)
