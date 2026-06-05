# Qwen3-8B WikiText-2 K9 Three-Seed BCE vs Exact-K CE

Last updated: 2026-06-05

## Setup

| field | value |
|---|---|
| model | `/workspace/Models/Qwen3-8B` |
| dataset | `/workspace/datasets/wikitext/wikitext-2-raw-v1` |
| seeds | `42`, `13`, `3407` |
| seq_len / router_prefix_tokens | `1024` / `256` |
| label_samples | `2000` |
| eval_windows requested / actual | `512` / `289` test windows |
| skip_rate / skip_count | `0.25` / `9` |
| protected_head / protected_tail | `4` / `2` |
| teacher | clean `final Delta_NLL` greedy set labels |
| losses compared | BCE vs Exact-K CE |

## Per-Seed Results

| seed | method | epoch | val NLL | val PPL | test NLL | test PPL | unique_masks | exact-K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | Full | final | NA | NA | 2.2523 | 9.5100 | 1 | 1.000 |
| 42 | Raw BCE final | final | NA | NA | 3.0151 | 20.3915 | 224 | 1.000 |
| 42 | OPAL BCE final | final | NA | NA | 3.0341 | 20.7827 | 247 | 1.000 |
| 42 | Raw BCE best | 1 | 2.9909 | 19.9038 | 2.9484 | 19.0762 | 6 | 1.000 |
| 42 | OPAL BCE best | 1 | 2.9883 | 19.8515 | 2.9466 | 19.0412 | 1 | 1.000 |
| 42 | Raw ExactK best | 1 | 2.9909 | 19.9038 | 2.9484 | 19.0762 | 6 | 1.000 |
| 42 | OPAL ExactK best | 1 | 2.9883 | 19.8515 | 2.9465 | 19.0389 | 2 | 1.000 |
| 13 | Full | final | NA | NA | 2.2523 | 9.5100 | 1 | 1.000 |
| 13 | Raw BCE final | final | NA | NA | 3.0227 | 20.5474 | 214 | 1.000 |
| 13 | OPAL BCE final | final | NA | NA | 3.0494 | 21.1016 | 260 | 1.000 |
| 13 | Raw BCE best | 2 | 2.9911 | 19.9077 | 2.9483 | 19.0742 | 17 | 1.000 |
| 13 | OPAL BCE best | 7 | 2.9846 | 19.7778 | 2.9479 | 19.0666 | 7 | 1.000 |
| 13 | Raw ExactK best | 2 | 2.9945 | 19.9746 | 2.9581 | 19.2620 | 18 | 1.000 |
| 13 | OPAL ExactK best | 1 | 2.9883 | 19.8515 | 2.9466 | 19.0412 | 1 | 1.000 |
| 3407 | Full | final | NA | NA | 2.2523 | 9.5100 | 1 | 1.000 |
| 3407 | Raw BCE final | final | NA | NA | 3.0160 | 20.4089 | 224 | 1.000 |
| 3407 | OPAL BCE final | final | NA | NA | 3.0649 | 21.4327 | 261 | 1.000 |
| 3407 | Raw BCE best | 1 | 2.9866 | 19.8191 | 2.9397 | 18.9095 | 7 | 1.000 |
| 3407 | OPAL BCE best | 2 | 2.9883 | 19.8515 | 2.9466 | 19.0412 | 1 | 1.000 |
| 3407 | Raw ExactK best | 1 | 2.9834 | 19.7540 | 2.9385 | 18.8881 | 5 | 1.000 |
| 3407 | OPAL ExactK best | 1 | 2.9883 | 19.8515 | 2.9466 | 19.0412 | 1 | 1.000 |

## Mean/Std Summary

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| Full | 3 | 9.5100 | 0.0000 | 1.00 | `[9.5100, 9.5100, 9.5100]` |
| Raw BCE final | 3 | 20.4493 | 0.0698 | 220.67 | `[20.3915, 20.5474, 20.4089]` |
| OPAL BCE final | 3 | 21.1057 | 0.2653 | 256.00 | `[20.7827, 21.1016, 21.4327]` |
| Raw BCE best | 3 | 19.0200 | 0.0781 | 10.00 | `[19.0762, 19.0742, 18.9095]` |
| OPAL BCE best | 3 | 19.0497 | 0.0120 | 3.00 | `[19.0412, 19.0666, 19.0412]` |
| Raw ExactK best | 3 | 19.0754 | 0.1526 | 9.67 | `[19.0762, 19.2620, 18.8881]` |
| OPAL ExactK best | 3 | 19.0405 | 0.0011 | 1.33 | `[19.0389, 19.0412, 19.0412]` |

## Readout

- OPAL Exact-K CE slightly improves OPAL best-on-val over OPAL BCE best: `19.0405` vs `19.0497` mean PPL.
- OPAL Exact-K CE does **not** beat the best raw-prefix baseline: Raw BCE best has lower mean PPL, `19.0200` vs `19.0405`.
- OPAL Exact-K CE beats Raw Exact-K CE by mean PPL, `19.0405` vs `19.0754`, but this is not enough because Raw BCE best remains stronger.
- Exact-K CE does not solve the low-diversity issue. OPAL Exact-K selects epoch 1 on all three seeds and has unique masks `[2, 1, 1]`.
- Final 40-epoch BCE checkpoints remain clearly worse than validation-selected checkpoints, confirming the earlier surrogate-mismatch diagnosis.

Current Qwen3 K9 conclusion:

> On clean Qwen3-8B WikiText-2 with 25% layer skipping (K=9), dynamic routers beat fixed and candidate-library baselines, but validation-selected Raw BCE remains the best skip method among this loss ablation. OPAL Exact-K CE gives a small OPAL-side improvement but remains low-diversity and does not establish an OPAL-over-Raw win.
