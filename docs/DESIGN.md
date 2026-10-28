# rag-abstain: design

**Does your RAG system know when to say no?**

`rag-abstain` scores a retrieval-augmented generation (RAG) system on a labelled question set and
treats **abstention** (declining to answer) as a first-class result, next to answer correctness,
retrieval quality, run-to-run consistency and latency.

Status: design for v0.1 (target 13 Nov 2026). Author: Mokith Maheshkumar.

---

## 1. Why this exists

A RAG system fails in two opposite ways:

- it **answers a question its documents can't support** (an unsupported answer), or
- it **declines a question its documents do answer** (a false decline).

The popular open-source evaluation frameworks (Ragas, DeepEval, TruLens, checked Oct 2026) score
faithfulness, context precision/recall and answer relevance. None of them reports abstention as a
headline metric: how often the system declined when it should have, how often it declined when it
shouldn't have, where the decline happened (before or after the model call), and whether the
same question gets the same outcome twice.

Research benchmarks do measure this (RGB's "negative rejection", Chen et al., AAAI 2024;
UAEval4RAG, Peng et al., ACL 2025; AbstentionBench, Kirichenko et al., 2025), but as fixed
benchmarks, not as a tool you point at your own pipeline and corpus.

`rag-abstain` is that tool. It grew out of the evaluation in my final-year project, a local RAG
assistant with a retrieval-distance gate, where three findings shaped the design:

1. **Hedges are their own outcome.** The model often writes "The documents do not contain this
   information. However, ..." and then answers. Scoring that as a decline flatters the system;
   it is an unsupported answer with a disclaimer on top. `rag-abstain` reports `hedged` separately
   and counts it as unsupported.
2. **Where the decline happens matters.** A distance gate that declines before the model is called
   costs nothing and is model-independent; a decline by the model depends on the prompt and the
   model. Reporting the split explains why results do or don't transfer to another model.
3. **Determinism is not free.** With default sampling one configuration disagreed with itself
   across two runs. Even at temperature 0 a hosted model gave identical text on 28 of 30
   questions, and one question flipped between declined and answered. Repeats are built in.

## 2. Scope

**v0.1 (in):** question-set loader and validator; three adapters (Python callable, HTTP endpoint,
OpenAI-compatible chat model as a no-retrieval baseline); rule-based outcome classifier (port of
the FYP's); metrics (abstention, correctness, retrieval, determinism, latency, tokens); repeats;
resumable runs; threshold sweep from a gate-off run; run comparison; Markdown and JSON reports;
CLI; tests with no network; CI.

**v0.1 (planned, after the core):** LLM-judge outcome classifier with agreement against human
labels; an extended question set for the demo corpus; benchmark table in the README.

**Out:** a hosted service or UI; training; generating whole datasets automatically; tracing.
`rag-abstain` evaluates a system from the outside, through the same door a user would.

## 3. Concepts

| Term | Meaning |
|---|---|
| **Question set** | Labelled questions. Each has `expected` = `answer` or `decline`. |
| **System** | The thing under test, reached through an adapter and described by a YAML config. |
| **Response** | What the system returned for one question: answer text, plus optional decline flag and stage, sources, distances, timings, tokens. |
| **Outcome** | `answered`, `declined`, `hedged` or `error`, resolved per response (section 6). |
| **Run** | One system x one question set x R repeats. Written to its own directory. |
| **Unsupported answer** | Outcome `answered` or `hedged` on a question whose `expected` is `decline`. |
| **False decline** | Outcome `declined` on a question whose `expected` is `answer`. |

Terminology rule: "unsupported answer" for an answer to an unanswerable question; "hallucination"
only for factually wrong output. The tool never prints "hallucination rate".

## 4. Question set format

CSV or JSONL. One row per question. Required: `qid`, `question`, `expected`. Everything else is
optional and unlocks more metrics.

| Field | Type | Used for |
|---|---|---|
| `qid` | str, unique | joins across runs |
| `question` | str | sent to the system |
| `expected` | `answer` or `decline` | abstention metrics |
| `category` | str (free label, e.g. `answerable`, `out_of_corpus`, `related_unanswerable`) | per-category breakdown |
| `references` | list of str (CSV: `|`-separated) | Exact Match and token F1 |
| `key_terms` | list of groups (CSV: groups `;`-separated, alternatives `|`-separated) | key-term hit: every group must have at least one alternative present |
| `gold_sources` | list of str (CSV: `|`-separated) | hit@k, recall@k, MRR |
| `notes` | str | carried into reports |

Example (CSV):

```csv
qid,category,expected,question,references,key_terms,gold_sources
A03,answerable,answer,Which researchers is the CYK algorithm named after?,Cocke Younger Kasami,Cocke;Younger;Kasami,CYK algorithm.txt
B01,out_of_corpus,decline,What is the capital of Brazil?,,,
```

`rag-abstain validate-dataset <file>` checks: unique qids, valid `expected`, `expected=answer`
rows have at least one of `references` / `key_terms` (warning, not error), no empty questions.
The dataset's SHA-256 goes in every run manifest.

## 5. Systems and adapters

A system is a YAML file:

```yaml
name: demo-gate-0.53            # used in run ids and reports
adapter: python                 # python | http | openai_chat
target: rag.pipeline:answer_question
pythonpath: ../rag-assistant-demo     # optional, prepended to sys.path
params: {use_rag: true, use_threshold: true}
response_map: demo              # preset name or an explicit mapping (below)
refusal_text: "The provided documents do not contain this information."
extra_markers: ["could you clarify"]  # optional, added to the rule classifier's markers
```

### 5.1 Common response shape

Every adapter returns a `SystemResponse`:

```python
class SystemResponse(BaseModel):
    answer: str = ""
    declined: bool | None = None          # the system's own flag, if it has one
    decline_stage: Literal["gate", "model"] | None = None
    sources: list[str] = []               # in rank order, may repeat
    contexts: list[str] = []
    distances: list[float] = []           # nearest first, if the system exposes them
    latency_s: float | None = None        # measured by the harness if the system doesn't say
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict = {}                        # the untouched payload, for debugging
    error: str | None = None
```

### 5.2 `python` adapter

Imports `module:function`, calls `function(question, **params)`. The return value may be a
`SystemResponse`, a `str` (answer only), or a `dict` mapped through `response_map`.

`response_map` is a preset name or a mapping from `SystemResponse` fields to dotted paths in the
returned dict, with optional value maps:

```yaml
response_map:
  answer: answer
  sources: sources
  distances: distances
  contexts: retrieved_chunks
  latency_s: total_s
  prompt_tokens: prompt_tokens
  completion_tokens: completion_tokens
  decline_stage:
    path: refusal_reason
    values: {retrieval_distance: gate, model_judgement: model}
  declined: refused
```

The preset `demo` is exactly this mapping (it matches the dict returned by
`rag-assistant-demo`'s `rag.pipeline.answer_question`).

### 5.3 `http` adapter

POSTs JSON to `url`. `request` is a template where `{question}` is substituted (JSON-escaped);
headers may reference environment variables as `${ENV_NAME}` (never stored in the run manifest).
`response_map` as above, applied to the JSON body. Timeout and retries configurable (defaults
60 s, 1 retry with 3 s backoff). Non-2xx or invalid JSON becomes `error`, never a crash.

### 5.4 `openai_chat` adapter (no-retrieval baseline)

Calls any OpenAI-compatible `/chat/completions` endpoint with `httpx` (no SDK dependency):

```yaml
name: llama31-8b-alone
adapter: openai_chat
base_url_env: LLM_BASE_URL
api_key_env: LLM_API_KEY
model: "@cf/meta/llama-3.1-8b-instruct-fp8"
prompt_template: "{question}"     # the bare question: no instruction to decline
temperature: 0
seed: 42
max_tokens: 1024
```

Reports must label this baseline honestly: it got the bare question with no instruction to
decline, so comparing it with a grounded pipeline compares the whole pipeline against the bare
model, not the prompt alone.

## 6. Outcome resolution

For each response, in order:

1. `error` set -> `error` (excluded from rates, counted and listed).
2. `decline_stage == "gate"` -> `declined` (stage `gate`).
3. Classify the answer text with the configured classifier -> `answered` / `declined` / `hedged`.
   A `declined` result from the classifier gets stage `model` if the system didn't say.
4. Empty answer -> `declined`.

### 6.1 Rule classifier (default)

A port of the FYP's `classify()`:

- refusal markers (configurable list; default the FYP's 12 English markers) plus the system's
  `extra_markers` and `refusal_text`, matched case-insensitively;
- no marker -> `answered`;
- marker present: take the text after the first "." that follows the marker; if it is longer
  than 100 characters, or one of the continuation phrases ("however", "but ", "although",
  "according to", "based on", ...) appears in its first 80 characters -> `hedged`; else `declined`.

Known limitation, kept visible: an answer followed by a trailing disclaimer ("... The documents do
not contain the specific formula for X.") is read as `declined`. Test case
`test_answer_then_trailing_disclaimer_is_known_limitation` pins the current behaviour so a fix is a
deliberate change.

This classifier is `rule-fyp` (recorded in manifests as `rules`, its name before `rule-v2`).
`rule-v2` runs `rule-fyp` first; when that says `answered`, it also looks for two kinds of reply
that are the right response to an unanswerable question but contain no refusal marker:

- a **clarification request** ("could you clarify", "which one are you referring to", "do you
  mean"): `declined`, or `hedged` if an answer follows it (the same length and continuation test
  as after a refusal marker);
- a **premise correction** ("the premise is incorrect", "is actually ... not", "there is no
  record of", "misconception"): `declined`, explanation included.

The patterns are generic phrasings, listed in `classify.py`, and were not fitted to any question
set's answers. A bare "X, not Y" is deliberately not a pattern (too common in ordinary answers),
so a correction phrased only that way is still `answered`.

A clarification pattern counts only inside a question (its sentence ends in "?") or a
please-imperative (its sentence starts with "please"). This rule was added **after** the first
rescore of the ext100 runs, where "you're referring to" matched statements such as "The paper
you're referring to is titled ..." in 8 Llama-alone records (A06, A21, A25, A26, both repeats).
The pattern list itself was not changed. Because the rule was written after seeing ext100
answers, `rule-v2` numbers on ext100 are not a held-out evaluation of the classifier.

Known limitation, kept visible: the premise pattern "is actually ... not" also matches an answer
that corrects some other error while answering (ext100 A25, Llama alone: "the first image was
actually of the SMACS 0723 galaxy cluster, not ..."), which `rule-v2` reads as `declined`.
Regex patterns will also miss phrasings they do not list; the judge classifier with its
agreement number is the fix for both.

**Label audit, 27 Oct 2026** (`results/2026-10-26T1641_demo-gate-0.53-ext100/rescore_rule-v2`,
repeat 1): a stratified sample of 10 `answered` and 10 `declined` records, drawn with
`random.Random(20261027)`, plus every record whose label differs between `rule-fyp` and `rule-v2`
(none in this run). Each was checked by hand against its question and answer: 0 of 20 labels
wrong. The gate-on ext100 numbers are good to use. The audit does not cover the Llama-alone
`rule-v2` labels: the demo declines only with its fixed refusal text, so no `rule-v2` pattern
fires in this run.

### 6.2 Judge classifier (after the core)

An OpenAI-compatible model labels each answer `answered` / `declined` / `hedged` with a fixed
rubric prompt (temperature 0, JSON output, one retry on invalid JSON). When enabled, both labels
are stored; the configured one drives the metrics. `rag-abstain agreement` reports Cohen's kappa
and a confusion matrix between rules, judge and an optional human-label file. No judge result is
published without its agreement number.

## 7. Metrics

All metrics are reported overall and per `category`. Rates use non-error responses; error counts
are always shown next to them. With repeats, rates use run 1 and determinism uses all runs.

**Abstention**

| Metric | Definition |
|---|---|
| `unsupported_answer_rate` | (answered + hedged) / n, over `expected = decline` |
| `false_decline_rate` | declined / n, over `expected = answer` |
| `hedge_rate` | hedged / n, over all |
| `decline_precision`, `decline_recall`, `decline_f1` | decline as the positive class; hedged counts as not declined |
| `balanced_accuracy` | mean of recall on the decline class and recall on the answer class |
| `declines_by_stage` | counts of `gate` vs `model` declines |

**Correctness** (over `expected = answer` with outcome `answered`; hedged and declined score 0 in
the "all answerable" view and are excluded in the "answered only" view; both are shown)

| Metric | Definition |
|---|---|
| `exact_match`, `f1` | SQuAD normalisation (lowercase, strip punctuation, drop a/an/the), best over references |
| `key_term_hit` | 1 if every key-term group has at least one alternative in the normalised answer |

**Retrieval** (needs `gold_sources` and returned `sources`; sources are de-duplicated in rank
order and compared by basename, case-insensitive)

| Metric | Definition |
|---|---|
| `hit_at_k` | 1 if any gold source is in the top k unique sources (k = 1, 3, 5, 10) |
| `recall_at_k` | share of gold sources in the top k |
| `mrr` | 1 / rank of the first gold source, 0 if none |

**Determinism** (repeats >= 2)

| Metric | Definition |
|---|---|
| `identical_answer_rate` | share of questions whose answer text is byte-identical across all repeats |
| `outcome_agreement_rate` | share of questions with the same outcome across all repeats |
| `outcome_flips` | list of qids whose outcome changed, with the outcomes seen |

**Cost**: latency median and p95 (split by decline stage, since gate declines are near-instant),
prompt and completion token totals, error count.

## 8. Threshold sweep

If a system exposes nearest-chunk distances and the run was made with its gate **off**, every
question has a model answer. The outcome under any threshold `t` can then be computed offline:
`declined (gate)` if `distances[0] > t`, otherwise the recorded outcome. `rag-abstain sweep`
prints, for a grid of `t`:

- `unsupported_answer_rate` and `false_decline_rate` (the trade-off curve),
- `balanced_accuracy`,
- the **margin**: largest nearest distance among `expected = answer` questions vs smallest among
  `expected = decline`; if the first is below the second, any threshold between them separates
  the set perfectly.

The sweep is labelled **in-sample** in its output. A threshold chosen from it must be validated on
a different question set before it is reported as a result. Optional plot with the `plot` extra.

## 9. Runs, reports and comparison

```
runs/2026-10-19T1530_demo-gate-0.53/
  manifest.json     run id, created_at, harness version, system config (secrets removed),
                    dataset path + SHA-256, repeats, classifier config, python version
  records.jsonl     one line per (qid, repeat) attempt: question, expected, category, response,
                    outcome, scores
  summary.json      every metric in section 7
  report.md         human-readable tables
```

- **Resumable:** re-running with `--resume <run_dir>` skips (qid, repeat) pairs that already
  have a non-error result in `records.jsonl` and retries pairs whose latest record is an
  `error` (a provider quota or an outage). A retry is appended; the old error line stays in the
  file as history, and the latest record for a pair is the one every metric and report uses.
  Records are appended and flushed one at a time, so a crash or a provider quota loses at most
  one call.
- **Workers:** `--workers 1` by default (quota-friendly); thread pool when higher.
- **Compare:** `rag-abstain compare <runA> <runB>` prints metric deltas and every question whose
  outcome changed, with both answers truncated to 200 characters. This replaces hand-written
  parity reports.
- **Rescore:** `rag-abstain rescore <run_dir> --classifier rule-fyp|rule-v2` re-classifies the
  stored answers (latest record per pair) with the system's markers from the manifest and writes
  `<run_dir>/rescore_<classifier>/` as a run directory of its own (manifest with `rescored_from`,
  records, summary, report). No system is called; scores are kept, since they depend only on the
  answer. The original files are not touched, and `compare` works between the two directories.

## 10. CLI

```
rag-abstain validate-dataset DATASET
rag-abstain run --system SYSTEM.yaml --dataset DATASET [--repeats 1] [--workers 1]
                [--limit N] [--out runs/] [--resume RUN_DIR] [--classifier rules|judge]
rag-abstain report RUN_DIR [--format md|json]
rag-abstain compare RUN_A RUN_B [--format md|json]
rag-abstain sweep RUN_DIR [--grid 0.20:1.20:0.05] [--plot out.png] [--format md|json]
rag-abstain rescore RUN_DIR --classifier rule-fyp|rule-v2
rag-abstain agreement RUN_DIR [--human labels.csv]          (after the core)
```

Exit codes: 0 ok, 1 usage or validation error, 2 run finished with errors in some records.

## 11. Package layout

```
rag-abstain/
  src/rag_abstain/
    __init__.py         version
    models.py           Question, SystemResponse, Record, RunManifest (pydantic v2)
    dataset.py          load CSV/JSONL, validate, hash
    adapters/
      __init__.py       registry: name -> adapter class; load_system(yaml_path)
      base.py           Adapter protocol: __call__(question: Question) -> SystemResponse
      python_callable.py
      http.py
      openai_chat.py
      mapping.py        response_map resolution + presets
    classify.py         RuleClassifier (rule-fyp, port), RuleV2Classifier, JudgeClassifier (later)
    metrics/
      abstention.py
      correctness.py    normalise, exact_match, f1, key_term_hit
      retrieval.py
      determinism.py
      cost.py
    runner.py           run loop, repeats, resume, workers, append-only writer
    sweep.py
    compare.py
    rescore.py          re-classify a finished run offline
    report.py           Markdown + JSON rendering
    cli.py              typer app
  tests/                one test module per source module; fixtures in tests/fixtures/
  examples/
    toy_system.py       deterministic fake RAG system (no network) used by tests and the quickstart
    toy_questions.csv
    toy_system.yaml
  datasets/             question sets (added in milestone 2)
  systems/              system configs for the benchmark (milestone 2)
  docs/DESIGN.md        this file
  pyproject.toml        hatchling; console script rag-abstain = rag_abstain.cli:app
  README.md  CHANGELOG.md  LICENSE (MIT)  .github/workflows/ci.yml
```

**Dependencies:** core `pydantic>=2.10`, `typer>=0.20` (the floor `huggingface-hub` allows, so the harness installs next to Hugging Face based pipelines; the CLI handles both the newer typer that bundles its own click and the older one that uses the separate `click` package, and CI tests the oldest), `httpx>=0.28`, `pyyaml>=6`. Extras:
`plot` = `matplotlib`; `dev` = `pytest`, `ruff`. No pandas, no LLM SDKs, no torch in the core.
Python 3.11+ (CI on 3.11, 3.12, 3.13).

## 12. Testing

No test touches the network. HTTP and OpenAI-compatible adapters are tested with
`httpx.MockTransport`. The Python adapter is tested against `examples/toy_system.py`.

Required cases:

- classifier: every case from the FYP/demo test file, plus the trailing-disclaimer limitation;
- metrics: hand-built records with known answers for every metric in section 7, including empty
  categories, all-error runs, and division by zero (reported as `null`, not 0);
- correctness: SQuAD normalisation examples (articles, punctuation, case);
- retrieval: duplicates in sources, basename matching, k larger than the list;
- determinism: identical, text-differs-same-outcome, outcome flip;
- sweep: synthetic distances with a known margin and a known best threshold;
- runner: resume skips done pairs and retries error pairs, with the latest record winning; a
  raising system becomes an `error` record, not a crash;
  secrets from `${ENV}` never appear in `manifest.json`;
- CLI: `run` -> `report` -> `compare` end to end on the toy system with typer's `CliRunner`.

## 13. Benchmark plan (README table, milestone 4)

Systems, all on the same corpus and question set, temperature 0, seed 42, 2 repeats:

1. Llama 3.1 8B alone (bare question, no instruction to decline)
2. RAG pipeline, prompt-only abstention (gate off)
3. RAG pipeline with the 0.53 distance gate (the FYP/demo configuration)
4. Same as 3 with a second model (chosen when the benchmark is run)

Question sets: the FYP's 30 questions, then an extended set for the same corpus (target ~100) with
unanswerable questions in the UAEval4RAG categories that fit a text corpus (out-of-database,
false presupposition, underspecified), each confirmed absent by searching the corpus.

Every number in the README comes from a run directory committed under `results/`.

## 14. Related work

- Chen, Lin, Han, Sun. *Benchmarking Large Language Models in Retrieval-Augmented Generation* (RGB),
  AAAI 2024. Negative rejection, noise robustness, information integration, counterfactual robustness.
- Peng, Choubey, Xiong, Wu. *Unanswerability Evaluation for Retrieval Augmented Generation*
  (UAEval4RAG), ACL 2025. Six unanswerable categories; code: SalesforceAIResearch/Unanswerability_RAGE.
- Kirichenko et al. *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions*, 2025.
- Rajpurkar, Jia, Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD* (SQuAD 2.0), ACL 2018.
- Ragas, DeepEval, TruLens: faithfulness / relevance / context metrics; complementary.
