# rag-abstain

Refusal-aware evaluation for RAG systems: "does your RAG system know when to say no?"
Mokith's first project outside coursework (INT-023). Public repo: github.com/MrMomoLegend/rag-abstain.
**Read `docs/DESIGN.md` before writing code. It is the spec; if code and spec disagree, ask.**

## Commands (Windows, PowerShell, venv in `venv\`)
- Activate: `venv\Scripts\activate` (do this before starting `claude`)
- Install: `python -m pip install -e ".[dev]"`
- Tests: `python -m pytest` (must be green before any commit)
- Lint: `ruff check .` and `ruff format --check .`
- CLI: `rag-abstain --help`

## Hard rules
- NEVER touch `C:\Users\mokit\ai-orchestrator` (the FYP repo, frozen at `e90dc2e` until grading). Read-only.
- `C:\Users\mokit\rag-assistant-demo` is a separate live app (Streamlit redeploys on every push). Read it; change it only when Mokith asks, in that repo, never from this one.
- NEVER read, print or commit `.env` or any key. Secrets come from environment variables named in system YAML (`${NAME}`), and must never be written into run manifests or logs.
- No test may call the network. Use `httpx.MockTransport` and `examples/toy_system.py`.
- Core dependencies stay light: pydantic, typer, httpx, pyyaml. No pandas, torch, LLM SDKs in the core.
- No claim in README or reports that isn't backed by a committed run directory. Say "unsupported answer", never "hallucination rate". Never write "eliminates hallucination", "0% hallucination" or "100% accurate".
- Division by zero in a metric -> `null` with a note, never 0.
- Writing style for docs: no em dashes, straight quotes.

## How to work
- Build in the phase order given in the kickoff prompt (A: models, data, classifier, metrics; B: adapters, runner; C: reports, sweep, compare, CLI, CI). Small, tested steps.
- After each module: run the tests, fix, then show a one-paragraph summary of what changed.
- Mokith is newer to the terminal: explain what a command does before running it.
- Ask before `git commit` and `git push`. Never `--force`. Commit messages: imperative, one line, describe the working state.
- Type hints everywhere, docstrings on public functions, no dead code, no TODO stubs left in committed code.
