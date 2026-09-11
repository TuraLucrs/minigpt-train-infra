# v0.9 matrix execution report

All metrics below come from verified raw benchmark artifacts. Missing hardware points remain incomplete.

## v09-ascend-a3

Evidence class: incomplete_or_development_matrix. Successful points: 12/12.
Preserved failed/interrupted attempts: 0.

| A/B case | Complete | Baseline token/s | Candidate token/s |
|---|---|---:|---:|
| kv-long-tp2 | True | 6.73849 | 5.36011 |
| tp-short-2-vs-8 | True | 5.80095 | 5.79604 |
| batching-short-tp8 | True | 21.2728 | 20.9877 |

Cross-backend profiler ratios require matching metric definitions; unknown metrics are unavailable, never zero.
