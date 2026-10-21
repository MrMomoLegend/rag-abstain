# Run report: 2026-10-21T1253_llama31-8b-alone

- System: `llama31-8b-alone` (adapter `openai_chat`)
- Dataset: `datasets\fyp30.csv` (SHA-256 d08a7b01dad5)
- Repeats: 2
- Classifier: rules
- Records: 60, errors: 0
- Created: 2026-10-21T12:53:16.763436+05:30, rag-abstain 0.1.0.dev0

> Baseline caveat: this system is a bare model that got the bare question with no instruction to decline. Comparing it with a grounded pipeline compares the whole pipeline against the bare model, not the prompt alone.

## Abstention (repeat 1)

Unsupported answer rate: answered or hedged, over questions expected to be declined. False decline rate: declined, over questions expected to be answered.

|  | n | errors | answered | declined | hedged | unsupported answer rate | false decline rate | hedge rate | decline precision | decline recall | decline F1 | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 30 | 0 | 30 | 0 | 0 | 1.000 | 0.000 | 0.000 | n/a | 0.000 | 0.000 | 0.500 |
| answerable | 15 | 0 | 15 | 0 | 0 | n/a | 0.000 | 0.000 | n/a | n/a | n/a | n/a |
| out_of_corpus | 12 | 0 | 12 | 0 | 0 | 1.000 | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a |
| related_unanswerable | 3 | 0 | 3 | 0 | 0 | 1.000 | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a |

Declines by stage: gate 0, model 0.

## Correctness (repeat 1)

Over questions expected to be answered. "answered only" excludes hedged and declined answers; "all" scores them 0.

|  | view | n | exact match | F1 | key-term hit |
|---|---|---|---|---|---|
| **all** | answered only | 15 | n/a | n/a | 0.867 |
| **all** | all | 15 | n/a | n/a | 0.867 |
| answerable | answered only | 15 | n/a | n/a | 0.867 |
| answerable | all | 15 | n/a | n/a | 0.867 |

## Retrieval (repeat 1)

Not scored: no record has both gold sources and returned sources.

## Determinism (all repeats)

|  | questions compared | identical answer rate | outcome agreement rate |
|---|---|---|---|
| **all** | 30 | 0.633 | 1.000 |
| answerable | 15 | 0.467 | 1.000 |
| out_of_corpus | 12 | 1.000 | 1.000 |
| related_unanswerable | 3 | 0.000 | 1.000 |

Outcome flips: none.

## Cost (all repeats)

| latency | n | median (s) | p95 (s) |
|---|---|---|---|
| all | 60 | 11.965 | 41.551 |
| gate declines | 0 | n/a | n/a |
| model declines | 0 | n/a | n/a |
| not declined | 60 | 11.965 | 41.551 |

Prompt tokens: 1314. Completion tokens: 15791. Error records: 0.

## Errors

None.

## Notes

- abstention.decline_precision: the system declined nothing
- correctness.answered_only.exact_match: no answered expected=answer records with this score
- correctness.all_answerable.exact_match: no expected=answer records with this score
- correctness.answered_only.f1: no answered expected=answer records with this score
- correctness.all_answerable.f1: no expected=answer records with this score
- retrieval.hit_at_1: no non-error records with both gold_sources and returned sources
- retrieval.hit_at_3: no non-error records with both gold_sources and returned sources
- retrieval.hit_at_5: no non-error records with both gold_sources and returned sources
- retrieval.hit_at_10: no non-error records with both gold_sources and returned sources
- retrieval.recall_at_1: no non-error records with both gold_sources and returned sources
- retrieval.recall_at_3: no non-error records with both gold_sources and returned sources
- retrieval.recall_at_5: no non-error records with both gold_sources and returned sources
- retrieval.recall_at_10: no non-error records with both gold_sources and returned sources
- retrieval.mrr: no non-error records with both gold_sources and returned sources
- cost.latency.gate: no non-error records with a latency in this group
- cost.latency.model: no non-error records with a latency in this group
