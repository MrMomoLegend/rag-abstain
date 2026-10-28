# Run report: 2026-10-26T1708_llama31-8b-alone-ext100+rule-v2

- System: `llama31-8b-alone-ext100` (adapter `openai_chat`)
- Dataset: `datasets\ext100.csv` (SHA-256 b17d8d7b7f70)
- Repeats: 2
- Classifier: rule-v2
- Records: 200, errors: 0
- Created: 2026-10-26T17:08:01.396900+05:30, rag-abstain 0.1.0.dev0

> Baseline caveat: this system is a bare model that got the bare question with no instruction to decline. Comparing it with a grounded pipeline compares the whole pipeline against the bare model, not the prompt alone.

## Abstention (repeat 1)

Unsupported answer rate: answered or hedged, over questions expected to be declined. False decline rate: declined, over questions expected to be answered.

|  | n | errors | answered | declined | hedged | unsupported answer rate | false decline rate | hedge rate | decline precision | decline recall | decline F1 | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 100 | 0 | 94 | 6 | 0 | 0.920 | 0.040 | 0.000 | 0.667 | 0.080 | 0.143 | 0.520 |
| answerable | 50 | 0 | 48 | 2 | 0 | n/a | 0.040 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| false_presupposition | 14 | 0 | 13 | 1 | 0 | 0.929 | n/a | 0.000 | 1.000 | 0.071 | 0.133 | n/a |
| look_answerable | 12 | 0 | 12 | 0 | 0 | 1.000 | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a |
| out_of_database | 14 | 0 | 14 | 0 | 0 | 1.000 | n/a | 0.000 | n/a | 0.000 | 0.000 | n/a |
| underspecified | 10 | 0 | 7 | 3 | 0 | 0.700 | n/a | 0.000 | 1.000 | 0.300 | 0.462 | n/a |

Declines by stage: gate 0, model 6.

## Correctness (repeat 1)

Over questions expected to be answered. "answered only" excludes hedged and declined answers; "all" scores them 0.

|  | view | n | exact match | F1 | key-term hit |
|---|---|---|---|---|---|
| **all** | answered only | 48 | 0.000 | 0.101 | 0.354 |
| **all** | all | 50 | 0.000 | 0.097 | 0.340 |
| answerable | answered only | 48 | 0.000 | 0.101 | 0.354 |
| answerable | all | 50 | 0.000 | 0.097 | 0.340 |

## Retrieval (repeat 1)

Not scored: no record has both gold sources and returned sources.

## Determinism (all repeats)

|  | questions compared | identical answer rate | outcome agreement rate |
|---|---|---|---|
| **all** | 100 | 0.650 | 1.000 |
| answerable | 50 | 0.620 | 1.000 |
| false_presupposition | 14 | 0.786 | 1.000 |
| look_answerable | 12 | 0.667 | 1.000 |
| out_of_database | 14 | 0.571 | 1.000 |
| underspecified | 10 | 0.700 | 1.000 |

Outcome flips: none.

## Cost (all repeats)

| latency | n | median (s) | p95 (s) |
|---|---|---|---|
| all | 200 | 10.122 | 34.693 |
| gate declines | 0 | n/a | n/a |
| model declines | 12 | 3.773 | 27.337 |
| not declined | 188 | 10.638 | 34.954 |

Prompt tokens: 4922. Completion tokens: 43090. Error records: 0.

## Errors

None.

## Notes

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
