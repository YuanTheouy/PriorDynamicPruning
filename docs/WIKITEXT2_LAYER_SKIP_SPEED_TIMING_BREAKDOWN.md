# WikiText-2 Layer-Skip Speed Timing Breakdown

Last updated: 2026-06-09

This document records the corrected OPAL speed timing breakdown after physically skipping the prefix feature extraction path. It isolates `Full`, `OPAL BCE best-on-val`, and `OPAL Exact-K best-on-val`; full method speed comparisons remain in the generated server speed report.

## Key Takeaways

- Corrected OPAL inference is faster than Full on both Qwen3-8B and Llama3.1-8B-Instruct.
- Router/mask inference costs about 9%-12% of per-window latency.
- The main skipped forward is visibly faster than Full forward, so physical layer skipping is working.
- OPAL is not the fastest possible skipping method because it pays router overhead; its role is the quality/speed tradeoff, not pure static-mask speed.
- Previous OPAL-slower rows are superseded by this corrected timing-breakdown run.

## Qwen3-8B

- model path: `/workspace/Models/Qwen3-8B`
- run label: `qwen3_seed42_opal_timing_breakdown`
- seq_len / router_prefix_tokens: `1536` / `1024`
- windows / batch size: `96` / `8`
- skip_count / keep_count: `9` / `27`

| mode | method | PPL ref | windows/s | tok/s | latency/window ms | router ms/win | forward ms/win | router % | speedup | avg kept | unique masks | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch1_true_skip | Full | 8.7518 | 7.3249 | 11251.1 | 136.52 | 0.00 | 136.52 | 0.0% | 1.000x | 36.00 | 1 | 1.000 |
| batch1_true_skip | OPAL BCE best-on-val | 17.5360 | 8.5452 | 13125.4 | 117.02 | 12.58 | 104.44 | 10.8% | 1.167x | 27.00 | 1 | 1.000 |
| batch1_true_skip | OPAL Exact-K best-on-val | 17.5403 | 8.5297 | 13101.7 | 117.24 | 12.63 | 104.61 | 10.8% | 1.164x | 27.00 | 2 | 1.000 |
| grouped_by_mask | Full | 8.7518 | 7.8982 | 12131.6 | 126.61 | 0.00 | 126.61 | 0.0% | 1.000x | 36.00 | 1 | 1.000 |
| grouped_by_mask | OPAL BCE best-on-val | 17.5360 | 9.3519 | 14364.5 | 106.93 | 9.61 | 97.32 | 9.0% | 1.184x | 27.00 | 1 | 1.000 |
| grouped_by_mask | OPAL Exact-K best-on-val | 17.5403 | 9.3306 | 14331.9 | 107.17 | 9.64 | 97.53 | 9.0% | 1.181x | 27.00 | 2 | 1.000 |
| mixed_batch_naive | Full | 8.7518 | 7.8745 | 12095.2 | 126.99 | 0.00 | 126.99 | 0.0% | 1.000x | 36.00 | 1 | 1.000 |
| mixed_batch_naive | OPAL BCE best-on-val | 17.5360 | 9.3481 | 14358.6 | 106.97 | 9.96 | 97.02 | 9.3% | 1.187x | 27.00 | 1 | 1.000 |
| mixed_batch_naive | OPAL Exact-K best-on-val | 17.5403 | 9.2578 | 14220.0 | 108.02 | 9.98 | 98.03 | 9.2% | 1.176x | 27.00 | 2 | 1.000 |

## Llama3.1-8B-Instruct

- model path: `/workspace/Models/Llama-3.1-8B-Instruct`
- run label: `llama31_seed42_opal_timing_breakdown`
- seq_len / router_prefix_tokens: `1536` / `1024`
- windows / batch size: `96` / `8`
- skip_count / keep_count: `8` / `24`

| mode | method | PPL ref | windows/s | tok/s | latency/window ms | router ms/win | forward ms/win | router % | speedup | avg kept | unique masks | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch1_true_skip | Full | 6.5076 | 7.9730 | 12246.5 | 125.42 | 0.00 | 125.42 | 0.0% | 1.000x | 32.00 | 1 | 1.000 |
| batch1_true_skip | OPAL BCE best-on-val | 16.0023 | 9.1563 | 14064.0 | 109.21 | 13.21 | 96.00 | 12.1% | 1.148x | 24.00 | 8 | 1.000 |
| batch1_true_skip | OPAL Exact-K best-on-val | 15.9588 | 9.1548 | 14061.7 | 109.23 | 13.18 | 96.05 | 12.1% | 1.148x | 24.00 | 9 | 1.000 |
| grouped_by_mask | Full | 6.5076 | 8.5948 | 13201.7 | 116.35 | 0.00 | 116.35 | 0.0% | 1.000x | 32.00 | 1 | 1.000 |
| grouped_by_mask | OPAL BCE best-on-val | 16.0023 | 9.9773 | 15325.1 | 100.23 | 9.96 | 90.27 | 9.9% | 1.161x | 24.00 | 14 | 1.000 |
| grouped_by_mask | OPAL Exact-K best-on-val | 15.9588 | 9.9646 | 15305.6 | 100.36 | 10.03 | 90.32 | 10.0% | 1.159x | 24.00 | 16 | 1.000 |
| mixed_batch_naive | Full | 6.5076 | 8.5853 | 13187.0 | 116.48 | 0.00 | 116.48 | 0.0% | 1.000x | 32.00 | 1 | 1.000 |
| mixed_batch_naive | OPAL BCE best-on-val | 16.0023 | 9.0416 | 13888.0 | 110.60 | 10.27 | 100.33 | 9.3% | 1.053x | 24.00 | 14 | 1.000 |
| mixed_batch_naive | OPAL Exact-K best-on-val | 15.9588 | 8.9577 | 13759.0 | 111.64 | 10.34 | 101.30 | 9.3% | 1.043x | 24.00 | 16 | 1.000 |

## Interpretation

The corrected measurements show OPAL has real wall-clock speedup, but its gain is reduced by router overhead. On Qwen3-8B, OPAL reaches about 1.16x-1.18x speedup. On Llama3.1-8B-Instruct, OPAL reaches about 1.15x-1.16x in batch1/grouped settings and about 1.04x-1.05x under naive mixed batching.

For paper writing, the safe claim is: OPAL improves quality under the same skip budget and achieves measured wall-clock speedup after physical layer skipping, with a non-trivial 9%-12% router overhead. Static or simpler routers can be faster, but they have worse PPL in the main prefix1024 WikiText-2 comparisons.
