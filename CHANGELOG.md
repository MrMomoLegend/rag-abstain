# Changelog

## 0.1.0-dev (unreleased)

### Added

- Question sets in CSV or JSONL, with a validator (`rag-abstain validate-dataset`) and the
  dataset's SHA-256 in every run manifest.
- Rule-based outcome classifier (`answered`, `declined`, `hedged`), ported from the final-year
  project, with hedges counted as unsupported answers.
- Adapters: `python` (call a function in-process), `http` (POST JSON, `${ENV}` headers) and
  `openai_chat` (any OpenAI-compatible chat endpoint, as a no-retrieval baseline), with
  `response_map` presets including `demo`.
- Metrics: abstention, correctness (Exact Match, token F1, key-term hit), retrieval (hit@k,
  recall@k, MRR), determinism across repeats, and cost (latency by decline stage, tokens).
  A metric that would divide by zero is `null` with a note.
- Runner: repeats, `--limit`, worker threads, an append-only `records.jsonl` flushed after every
  record, resume that retries error records, and secrets scrubbed from everything written.
- Reports: `summary.json` and `report.md` per run, overall and per category, with an error list
  and the bare-model caveat for `openai_chat` baselines.
- `rag-abstain compare` (metric deltas and changed outcomes) and `rag-abstain sweep`
  (offline distance-threshold sweep on a gate-off run, labelled in-sample, optional plot with
  the `plot` extra).
- CLI exit codes: 0 ok, 1 usage or validation error, 2 run finished with error records.
- Toy system and question set in `examples/`, tests with no network, CI on Python 3.11 to 3.13.
- `typer>=0.20` instead of `>=0.27`, so the harness installs in the same environment as a
  pipeline that uses `huggingface-hub` (which needs `typer<0.26`). CI also tests typer 0.20.0 and 0.25.1.
