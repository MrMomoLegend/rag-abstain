# rag-abstain

**Does your RAG system know when to say no?** `rag-abstain` scores a retrieval-augmented
generation system on a labelled question set and treats declining to answer as a first-class
result, next to correctness, retrieval, consistency and latency.

**Status:** in development; benchmark results coming. The design is in
[docs/DESIGN.md](docs/DESIGN.md).

## Why

A RAG system fails in two opposite ways. It can answer a question its documents can't support
(an unsupported answer), or decline a question its documents do answer (a false decline). Both
matter, and they trade off against each other.

The popular open-source evaluation frameworks (Ragas, DeepEval, TruLens, checked Oct 2026)
score faithfulness, context precision and recall, and answer relevance. None of them reports
abstention as a headline metric. Research benchmarks such as RGB, UAEval4RAG and
AbstentionBench do measure it, but as fixed benchmarks, not as a tool you point at your own
pipeline and corpus.

`rag-abstain` is that tool. It grew out of the evaluation in my final-year project, a local RAG
assistant with a retrieval-distance gate. It reports hedges ("The documents do not contain this
information. However, ...") as their own outcome and counts them as unsupported, splits declines
by where they happened (the retrieval gate or the model), and repeats runs to check that the
same question gets the same outcome.

## What it measures

| Area | Metrics |
|---|---|
| Abstention | unsupported answer rate, false decline rate, hedge rate, decline precision / recall / F1, balanced accuracy, declines by stage (gate or model) |
| Correctness | Exact Match, token F1, key-term hit; over answered questions only and over all answerable questions |
| Retrieval | hit@k, recall@k (k = 1, 3, 5, 10), MRR against gold source documents |
| Determinism | identical answer rate, outcome agreement rate, and the questions whose outcome flipped across repeats |
| Cost | latency median and p95 by decline stage, prompt and completion tokens, errors |

Every metric is reported overall and per question category. A metric that would divide by zero
is reported as `null` with a note, never as 0.

## Quickstart

Not on PyPI yet, so install from a clone (Python 3.11 or later):

```
git clone https://github.com/MrMomoLegend/rag-abstain.git
cd rag-abstain
python -m venv venv
venv\Scripts\activate            # Windows; on macOS or Linux: source venv/bin/activate
python -m pip install -e ".[dev]"
```

The repository ships a deterministic toy RAG system (`examples/toy_system.py`, no network and no
model) and a 10-question set. The numbers below describe that toy and nothing else.

Run it with its distance gate on, then with the gate off:

```
> rag-abstain validate-dataset examples/toy_questions.csv
examples\toy_questions.csv: OK, 10 questions, 0 warning(s)

> rag-abstain run --system examples/toy_system.yaml --dataset examples/toy_questions.csv
Run: runs\2026-10-17T1128_toy-gate-0.53
Records: 10 new (0 retried), 0 skipped; errors: 0
Unsupported answer rate: 0.250  False decline rate: 0.167  Hedge rate: 0.100  Balanced accuracy: 0.792
Report: runs\2026-10-17T1128_toy-gate-0.53\report.md

> rag-abstain run --system examples/toy_system_gate_off.yaml --dataset examples/toy_questions.csv
Run: runs\2026-10-17T1128_toy-gate-off
Records: 10 new (0 retried), 0 skipped; errors: 0
Unsupported answer rate: 0.500  False decline rate: 0.000  Hedge rate: 0.100  Balanced accuracy: 0.750
Report: runs\2026-10-17T1128_toy-gate-off\report.md
```

Each run directory holds `manifest.json`, `records.jsonl`, `summary.json` and `report.md`.
Compare the two runs to see which questions the gate changed (excerpt):

```
> rag-abstain compare runs/2026-10-17T1128_toy-gate-0.53 runs/2026-10-17T1128_toy-gate-off
...
## Changed outcomes (repeat 1): 3

| qid | expected | A | B | answer A | answer B |
|---|---|---|---|---|---|
| T08 | decline | declined (gate) | answered | The provided documents do not contain this information. | The capital of Brazil is Brasilia. |
| T09 | decline | declined (gate) | declined (model) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| T10 | answer | declined (gate) | answered | The provided documents do not contain this information. | BLEU is used to evaluate the quality of machine translation output against reference translations. |
```

From the gate-off run, replay the gate offline at other thresholds (excerpt):

```
> rag-abstain sweep runs/2026-10-17T1128_toy-gate-off --grid 0.30:0.80:0.10
# Threshold sweep (in-sample): 2026-10-17T1128_toy-gate-off

> In-sample: these thresholds are scored on the same questions they would be chosen from. Validate a chosen threshold on a different question set before reporting it as a result.
...
| threshold | unsupported answer rate | false decline rate | balanced accuracy | gate declines |
|---|---|---|---|---|
| 0.30 | 0.000 | 0.500 | 0.750 | 7 |
| 0.40 | 0.000 | 0.167 | 0.917 | 5 |
| 0.50 | 0.250 | 0.167 | 0.792 | 3 |
| 0.60 | 0.250 | 0.000 | 0.875 | 2 |
| 0.70 | 0.250 | 0.000 | 0.875 | 2 |
| 0.80 | 0.500 | 0.000 | 0.750 | 1 |
```

Add `--plot sweep.png` to save a chart (needs `python -m pip install -e ".[plot]"`).

To test your own system, write a system YAML for one of the three adapters (`python`, `http`,
`openai_chat`) as described in [docs/DESIGN.md](docs/DESIGN.md), section 5. Run `rag-abstain
--help` for every command and option.

## License

MIT
