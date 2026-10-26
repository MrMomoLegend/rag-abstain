# Compare: 2026-10-26T1649_demo-gate-off-ext100 (A) vs 2026-10-26T1641_demo-gate-0.53-ext100 (B)

- A: `demo-gate-off-ext100`
- B: `demo-gate-0.53-ext100`

## Metrics

| metric | A | B | B - A |
|---|---|---|---|
| unsupported answer rate | 0.140 | 0.140 | +0.000 |
| false decline rate | 0.220 | 0.240 | +0.020 |
| hedge rate | 0.000 | 0.000 | +0.000 |
| decline F1 | 0.827 | 0.819 | -0.008 |
| balanced accuracy | 0.820 | 0.810 | -0.010 |
| exact match (all answerable) | 0.200 | 0.200 | +0.000 |
| F1 (all answerable) | 0.500 | 0.492 | -0.008 |
| key-term hit (all answerable) | 0.720 | 0.700 | -0.020 |
| hit@3 | 0.980 | 0.980 | +0.000 |
| MRR | 0.933 | 0.933 | +0.000 |
| identical answer rate | 0.940 | 0.980 | +0.040 |
| outcome agreement rate | 1.000 | 1.000 | +0.000 |
| median latency (s) | 1.359 | 1.228 | -0.131 |
| p95 latency (s) | 3.539 | 3.335 | -0.204 |
| prompt tokens | 234986 | 197820 | -37166 |
| completion tokens | 3033 | 2699 | -334 |
| error records | 0 | 0 | +0 |

## Changed outcomes (repeat 1): 16

| qid | expected | A | B | answer A | answer B |
|---|---|---|---|---|---|
| A34 | answer | answered | declined (gate) | The Bell Labs Audrey system, built in 1952, was designed for single-speaker digit recognition. | The provided documents do not contain this information. |
| O01 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O02 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O03 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O05 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O08 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O10 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O11 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O12 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| O13 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| U02 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| U08 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| U09 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| L06 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| L08 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |
| L10 | decline | declined (model) | declined (gate) | The provided documents do not contain this information. | The provided documents do not contain this information. |

