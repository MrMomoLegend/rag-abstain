# Run report: 2026-10-21T1310_demo-gate-off

- System: `demo-gate-off` (adapter `python`)
- Dataset: `datasets\fyp30.csv` (SHA-256 d08a7b01dad5)
- Repeats: 2
- Classifier: rules
- Records: 60, errors: 0
- Created: 2026-10-21T13:10:00.045410+05:30, rag-abstain 0.1.0.dev0

## Abstention (repeat 1)

Unsupported answer rate: answered or hedged, over questions expected to be declined. False decline rate: declined, over questions expected to be answered.

|  | n | errors | answered | declined | hedged | unsupported answer rate | false decline rate | hedge rate | decline precision | decline recall | decline F1 | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 30 | 0 | 14 | 16 | 0 | 0.000 | 0.067 | 0.000 | 0.938 | 1.000 | 0.968 | 0.967 |
| answerable | 15 | 0 | 14 | 1 | 0 | n/a | 0.067 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| out_of_corpus | 12 | 0 | 0 | 12 | 0 | 0.000 | n/a | 0.000 | 1.000 | 1.000 | 1.000 | n/a |
| related_unanswerable | 3 | 0 | 0 | 3 | 0 | 0.000 | n/a | 0.000 | 1.000 | 1.000 | 1.000 | n/a |

Declines by stage: gate 0, model 16.

## Correctness (repeat 1)

Over questions expected to be answered. "answered only" excludes hedged and declined answers; "all" scores them 0.

|  | view | n | exact match | F1 | key-term hit |
|---|---|---|---|---|---|
| **all** | answered only | 14 | n/a | n/a | 0.929 |
| **all** | all | 15 | n/a | n/a | 0.867 |
| answerable | answered only | 14 | n/a | n/a | 0.929 |
| answerable | all | 15 | n/a | n/a | 0.867 |

## Retrieval (repeat 1)

|  | scored | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 15 | 0.933 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 | 0.967 |
| answerable | 15 | 0.933 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 | 0.967 |
| out_of_corpus | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| related_unanswerable | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Determinism (all repeats)

|  | questions compared | identical answer rate | outcome agreement rate |
|---|---|---|---|
| **all** | 30 | 0.933 | 1.000 |
| answerable | 15 | 0.867 | 1.000 |
| out_of_corpus | 12 | 1.000 | 1.000 |
| related_unanswerable | 3 | 1.000 | 1.000 |

Outcome flips: none.

## Cost (all repeats)

| latency | n | median (s) | p95 (s) |
|---|---|---|---|
| all | 60 | 1.265 | 7.138 |
| gate declines | 0 | n/a | n/a |
| model declines | 32 | 1.173 | 1.872 |
| not declined | 28 | 1.728 | 15.810 |

Prompt tokens: 70282. Completion tokens: 1440. Error records: 0.

## Errors

None.

## Notes

- correctness.answered_only.exact_match: no answered expected=answer records with this score
- correctness.all_answerable.exact_match: no expected=answer records with this score
- correctness.answered_only.f1: no answered expected=answer records with this score
- correctness.all_answerable.f1: no expected=answer records with this score
- cost.latency.gate: no non-error records with a latency in this group
