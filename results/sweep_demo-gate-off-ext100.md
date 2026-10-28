# Threshold sweep (in-sample): 2026-10-26T1649_demo-gate-off-ext100

> In-sample: these thresholds are scored on the same questions they would be chosen from. Validate a chosen threshold on a different question set before reporting it as a result.

Questions: 100 (errors excluded: 0).

## Margin

- Largest nearest distance, expected=answer: 0.667
- Smallest nearest distance, expected=decline: 0.243
- Margin: -0.424
- Not separable: no threshold gates exactly the expected=decline questions.

Highest balanced accuracy on this set: 0.830 at t = 0.40 (in-sample).

## Grid

| threshold | unsupported answer rate | false decline rate | balanced accuracy | gate declines |
|---|---|---|---|---|
| 0.30 | 0.000 | 0.620 | 0.690 | 72 |
| 0.35 | 0.000 | 0.480 | 0.760 | 57 |
| 0.40 | 0.000 | 0.340 | 0.830 | 43 |
| 0.45 | 0.100 | 0.240 | 0.830 | 26 |
| 0.50 | 0.140 | 0.240 | 0.810 | 20 |
| 0.55 | 0.140 | 0.240 | 0.810 | 11 |
| 0.60 | 0.140 | 0.240 | 0.810 | 8 |
| 0.65 | 0.140 | 0.240 | 0.810 | 4 |
| 0.70 | 0.140 | 0.220 | 0.820 | 0 |

