# Run report: 2026-10-26T1649_demo-gate-off-ext100

- System: `demo-gate-off-ext100` (adapter `python`)
- Dataset: `datasets\ext100.csv` (SHA-256 b17d8d7b7f70)
- Repeats: 2
- Classifier: rules
- Records: 200, errors: 0
- Created: 2026-10-26T16:49:13.240127+05:30, rag-abstain 0.1.0.dev0

## Abstention (repeat 1)

Unsupported answer rate: answered or hedged, over questions expected to be declined. False decline rate: declined, over questions expected to be answered.

|  | n | errors | answered | declined | hedged | unsupported answer rate | false decline rate | hedge rate | decline precision | decline recall | decline F1 | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 100 | 0 | 46 | 54 | 0 | 0.140 | 0.220 | 0.000 | 0.796 | 0.860 | 0.827 | 0.820 |
| answerable | 50 | 0 | 39 | 11 | 0 | n/a | 0.220 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| false_presupposition | 14 | 0 | 1 | 13 | 0 | 0.071 | n/a | 0.000 | 1.000 | 0.929 | 0.963 | n/a |
| look_answerable | 12 | 0 | 0 | 12 | 0 | 0.000 | n/a | 0.000 | 1.000 | 1.000 | 1.000 | n/a |
| out_of_database | 14 | 0 | 0 | 14 | 0 | 0.000 | n/a | 0.000 | 1.000 | 1.000 | 1.000 | n/a |
| underspecified | 10 | 0 | 6 | 4 | 0 | 0.600 | n/a | 0.000 | 1.000 | 0.400 | 0.571 | n/a |

Declines by stage: gate 0, model 54.

## Correctness (repeat 1)

Over questions expected to be answered. "answered only" excludes hedged and declined answers; "all" scores them 0.

|  | view | n | exact match | F1 | key-term hit |
|---|---|---|---|---|---|
| **all** | answered only | 39 | 0.256 | 0.641 | 0.923 |
| **all** | all | 50 | 0.200 | 0.500 | 0.720 |
| answerable | answered only | 39 | 0.256 | 0.641 | 0.923 |
| answerable | all | 50 | 0.200 | 0.500 | 0.720 |

## Retrieval (repeat 1)

|  | scored | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 50 | 0.900 | 0.980 | 0.980 | 0.980 | 0.900 | 0.980 | 0.980 | 0.980 | 0.933 |
| answerable | 50 | 0.900 | 0.980 | 0.980 | 0.980 | 0.900 | 0.980 | 0.980 | 0.980 | 0.933 |
| false_presupposition | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| look_answerable | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| out_of_database | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| underspecified | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Determinism (all repeats)

|  | questions compared | identical answer rate | outcome agreement rate |
|---|---|---|---|
| **all** | 100 | 0.940 | 1.000 |
| answerable | 50 | 0.880 | 1.000 |
| false_presupposition | 14 | 1.000 | 1.000 |
| look_answerable | 12 | 1.000 | 1.000 |
| out_of_database | 14 | 1.000 | 1.000 |
| underspecified | 10 | 1.000 | 1.000 |

Outcome flips: none.

## Cost (all repeats)

| latency | n | median (s) | p95 (s) |
|---|---|---|---|
| all | 200 | 1.359 | 3.539 |
| gate declines | 0 | n/a | n/a |
| model declines | 108 | 1.290 | 2.087 |
| not declined | 92 | 1.865 | 4.583 |

Prompt tokens: 234986. Completion tokens: 3033. Error records: 0.

## Errors

None.

## Notes

- cost.latency.gate: no non-error records with a latency in this group
