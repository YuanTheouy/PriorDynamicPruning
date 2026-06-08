# Qwen3-8B WikiText-2 Prefix1024 Three-Seed Results

Last updated: 2026-06-08

## Setup

| field | value |
|---|---|
| model | `/workspace/Models/Qwen3-8B` |
| dataset | `/workspace/datasets/wikitext/wikitext-2-raw-v1` |
| seeds | `42`, `13`, `3407` |
| seq_len / router_prefix_tokens | `1536` / `1024` |
| scored suffix tokens | `512` |
| requested label_samples | `2000` |
| actual clean label rows | `1625` |
| eval_windows requested | `512` |
| skip_rate / skip_count | `0.25` / `9` |
| protected_head / protected_tail | `4` / `2` |
| teacher | clean `final Delta_NLL` forward-greedy set labels |

This is not a beam2 teacher run. It only changes the context setting from `seq1024/prefix256` to `seq1536/prefix1024`.

## Per-Seed Results

| seed | method | loss | epoch | val NLL | val PPL | val unique | test NLL | test PPL | test unique | exact-K |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | Raw final | BCE | final | NA | NA | NA | 2.9313 | 18.7511 | 177 | 1.000 |
| 42 | OPAL final | BCE | final | NA | NA | NA | 2.9464 | 19.0376 | 180 | 1.000 |
| 42 | Raw best-on-val | BCE | 3 | 2.9361 | 18.8423 | 11 | 2.8783 | 17.7837 | 13 | 1.000 |
| 42 | OPAL best-on-val | BCE | 1 | 2.9287 | 18.7030 | 1 | 2.8643 | 17.5360 | 1 | 1.000 |
| 42 | Raw best-on-val | Exact-K CE | 3 | 2.9388 | 18.8939 | 10 | 2.8806 | 17.8252 | 11 | 1.000 |
| 42 | OPAL best-on-val | Exact-K CE | 2 | 2.9293 | 18.7138 | 2 | 2.8645 | 17.5403 | 2 | 1.000 |
| 13 | Raw final | BCE | final | NA | NA | NA | 2.9456 | 19.0220 | 148 | 1.000 |
| 13 | OPAL final | BCE | final | NA | NA | NA | 2.9732 | 19.5543 | 179 | 1.000 |
| 13 | Raw best-on-val | BCE | 7 | 2.9310 | 18.7467 | 40 | 2.8790 | 17.7967 | 36 | 1.000 |
| 13 | OPAL best-on-val | BCE | 11 | 2.9336 | 18.7949 | 9 | 2.8747 | 17.7205 | 11 | 1.000 |
| 13 | Raw best-on-val | Exact-K CE | 7 | 2.9319 | 18.7639 | 39 | 2.8756 | 17.7361 | 37 | 1.000 |
| 13 | OPAL best-on-val | Exact-K CE | 2 | 2.9343 | 18.8075 | 3 | 2.8732 | 17.6940 | 3 | 1.000 |
| 3407 | Raw final | BCE | final | NA | NA | NA | 2.9617 | 19.3311 | 144 | 1.000 |
| 3407 | OPAL final | BCE | final | NA | NA | NA | 2.9525 | 19.1537 | 183 | 1.000 |
| 3407 | Raw best-on-val | BCE | 8 | 2.9458 | 19.0266 | 33 | 2.8757 | 17.7382 | 37 | 1.000 |
| 3407 | OPAL best-on-val | BCE | 2 | 2.9287 | 18.7030 | 1 | 2.8642 | 17.5358 | 2 | 1.000 |
| 3407 | Raw best-on-val | Exact-K CE | 8 | 2.9415 | 18.9435 | 35 | 2.8726 | 17.6830 | 44 | 1.000 |
| 3407 | OPAL best-on-val | Exact-K CE | 10 | 2.9306 | 18.7390 | 21 | 2.8682 | 17.6049 | 27 | 1.000 |

## Mean/Std Summary

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| Raw BCE final | 3 | 19.0347 | 0.2370 | 156.33 | `[18.7511, 19.0220, 19.3311]` |
| OPAL BCE final | 3 | 19.2485 | 0.2213 | 180.67 | `[19.0376, 19.5543, 19.1537]` |
| Raw BCE best | 3 | 17.7729 | 0.0251 | 28.67 | `[17.7837, 17.7967, 17.7382]` |
| OPAL BCE best | 3 | 17.5974 | 0.0870 | 4.67 | `[17.5360, 17.7205, 17.5358]` |
| Raw Exact-K CE best | 3 | 17.7481 | 0.0587 | 30.67 | `[17.8252, 17.7361, 17.6830]` |
| OPAL Exact-K CE best | 3 | 17.6131 | 0.0630 | 10.67 | `[17.5403, 17.6940, 17.6049]` |
| PuDDing-style | 3 | 23.0576 | 1.0998 | 3.67 | `[23.5776, 21.5282, 24.0670]` |
| IG-style | 3 | 22.7616 | 1.1255 | 1.33 | `[23.2062, 21.2157, 23.8629]` |
| layerwise_hidden_router | 3 | 20.0723 | 0.1662 | 170.33 | `[20.3070, 19.9446, 19.9653]` |

## Related-Work Baselines

| seed | method | test NLL | test PPL | unique_masks | selected_candidate_distribution |
|---:|---|---:|---:|---:|---|
| 42 | PuDDing-style | 3.1603 | 23.5776 | 3 | `{"random_diverse_v3": 7, "random_diverse_v7": 128, "random_diverse_seed42": 58}` |
| 42 | IG-style | 3.1444 | 23.2062 | 2 | `{"random_diverse_v7": 102, "random_diverse_seed42": 91}` |
| 42 | layerwise_hidden_router | 3.0110 | 20.3070 | 175 | `{}` |
| 13 | PuDDing-style | 3.0694 | 21.5282 | 4 | `{"random_diverse_v3": 6, "random_diverse_v7": 183, "ends_heavy": 3, "random_diverse_v6": 1}` |
| 13 | IG-style | 3.0547 | 21.2157 | 1 | `{"random_diverse_v7": 193}` |
| 13 | layerwise_hidden_router | 2.9930 | 19.9446 | 170 | `{}` |
| 3407 | PuDDing-style | 3.1808 | 24.0670 | 4 | `{"random_diverse_v6": 149, "random_diverse_v8": 36, "ends_heavy": 4, "random_diverse_v5": 4}` |
| 3407 | IG-style | 3.1723 | 23.8629 | 1 | `{"random_diverse_v6": 193}` |
| 3407 | layerwise_hidden_router | 2.9940 | 19.9653 | 166 | `{}` |

## Required Judgments

- OPAL BCE best-on-val beats Raw BCE best-on-val on every seed: `True`.
- OPAL BCE best-on-val beats Raw BCE best-on-val by mean PPL: `17.5974` vs `17.7729`, gain `0.1754`.
- OPAL Exact-K CE best-on-val beats Raw Exact-K CE best-on-val on every seed: `True`.
- OPAL Exact-K CE best-on-val beats Raw Exact-K CE best-on-val by mean PPL: `17.6131` vs `17.7481`, gain `0.1350`.
- OPAL BCE best-on-val is the best mean-PPL row among these prefix1024 router rows: `17.5974`.
- OPAL BCE best-on-val beats PuDDing-style, IG-style, and layerwise_hidden_router by mean PPL: `True`.
- OPAL BCE best-on-val beats layerwise_hidden_router by mean PPL: `17.5974` vs `20.0723`, gain `2.4749`.
- PuDDing-style and IG-style do not collapse to ends_heavy under prefix1024; they mostly select random-diverse candidate masks.
- Final checkpoints do not show the same OPAL-over-Raw win: Raw BCE final mean `19.0347` beats OPAL BCE final mean `19.2485`.
- OPAL BCE best-on-val is low-diversity: unique masks `[1, 11, 2]`, mean `4.67`.
- OPAL Exact-K CE improves diversity on seed3407 (`27` unique masks), but mean PPL is slightly worse than OPAL BCE best: `17.6131` vs `17.5974`.

## Conclusion

Prefix1024 is now a three-seed WikiText-2 rescue for validation-selected OPAL on Qwen3-8B. The safe wording is:

> On clean Qwen3-8B WikiText-2 with 25% layer skipping (K=9), increasing the router-visible prefix from 256 to 1024 tokens makes validation-selected OPAL outperform Raw, PuDDing-style, IG-style, and layerwise hidden routing across three seeds. The best mean PPL is OPAL BCE best-on-val (`17.5974`), ahead of Raw BCE best-on-val (`17.7729`) and layerwise_hidden_router (`20.0723`). However, the OPAL BCE selected checkpoints are low-diversity, so this should be framed as a validation-selected prefix1024 OPAL rescue rather than evidence that high mask diversity is the source of the gain.
