# WikiText-2 Public LM Sanity Results

Last updated: 2026-06-02

## Status

This file records the real server result pasted from `/workspace/PriorDynamicPruning` after the `maskcfg` layer-mask fix. The earlier pre-fix smoke run where Full/static/Raw/OPAL all had identical NLL was invalid because the custom layer mask was not being applied through the Qwen2 forward path.

Current verdict: the final 40-epoch OPAL-SetBCE checkpoint is better than the static baselines, but it is not better than Raw-SetBCE on the main WikiText-2 run. A validation-selected OPAL checkpoint at epoch 2 does beat Raw-SetBCE and the layerwise hidden router on seed42 test PPL, but it collapses to one test mask; cite this as checkpoint-selection rescue evidence, not as a broad dynamic-routing win.

## Setup

- model path: `/workspace/ckpts/Qwen2.5-1.5B`
- model name: `Qwen2.5-1.5B`
- dataset: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- label split: `train`
- static selection split: `validation`
- final eval split: `test`
- num_layers: 28
- seq_len: 1024
- router_prefix_tokens: 256
- eval_windows requested: 512
- eval_windows actual: 289
- eval_tokens: 221,952
- skip_rate: 0.25
- skip_count: 7
- kept_layers: 21
- protected_head / protected_tail: 4 / 2
- mask application: `config.custom_layer_mask`
- target leakage guard: train labels only; validation only for static selection; test eval does not load greedy labels

WikiText-2 raw has thousands of original text rows, but this benchmark operates on contiguous token windows. With `seq_len=1024` and `router_prefix_tokens=256`, each full window contributes 768 scored suffix tokens; the pasted main test result therefore used 289 actual test windows.

## Main Result

| method | K skipped | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | eval_tokens | unique masks | kept layers | exact-K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0 | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 221,952 | 1 | 28.0 | 1.000 |
| Static uniform | 7 | 4.0673 | 58.3987 | 1.8519 | 49.2338 | 221,952 | 1 | 21.0 | 1.000 |
| Static ends_heavy | 7 | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 221,952 | 1 | 21.0 | 1.000 |
| Static best-on-val C6 | 7 | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 221,952 | 1 | 21.0 | 1.000 |
| Raw-SetBCE | 7 | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 221,952 | 8 | 21.0 | 1.000 |
| OPAL-SetBCE | 7 | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 221,952 | 15 | 21.0 | 1.000 |

## Validation-Selected OPAL Checkpoint

The OPAL-H4 40-epoch run was later evaluated checkpoint-by-checkpoint on validation. Epoch 2 has the best validation PPL and gives the following test result:

| method | selected by | K skipped | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | eval_tokens | unique masks | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OPAL-SetBCE best-on-val epoch2 | validation NLL | 7 | 2.7486 | 15.6204 | 0.5332 | 6.4555 | 221,952 | 1 | 1.000 |
| OPAL-SetBCE best-on-val minuniq8 epoch33 | validation NLL, `unique_masks>=8` | 7 | 2.7644 | 15.8695 | 0.5490 | 6.7046 | 221,952 | 13 | 1.000 |

This is better than the final OPAL checkpoint, Raw-SetBCE, and the related `layerwise_hidden_router` seed42 result. The caveat is `unique_masks=1`, so the best validation checkpoint behaves like a single high-quality OPAL-selected skip mask on test.

The dynamic-constrained validation selection (`WIKITEXT_VALCKPT_MIN_UNIQUE_MASKS=8`) selects epoch 33. It has 13 unique masks on test and improves over the final OPAL checkpoint, but remains slightly behind Raw-SetBCE and `layerwise_hidden_router`.

## Smoke Result

`seq512/pref128/m32` after the mask fix:

| method | K skipped | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | eval_tokens | unique masks | kept layers | exact-K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0 | 2.2147 | 9.1589 | 0.0000 | 0.0000 | 12,288 | 1 | 28.0 | 1.000 |
| Static uniform | 7 | 4.0762 | 58.9225 | 1.8615 | 49.7637 | 12,288 | 1 | 21.0 | 1.000 |
| Static ends_heavy | 7 | 2.8972 | 18.1225 | 0.6824 | 8.9636 | 12,288 | 1 | 21.0 | 1.000 |
| Static best-on-val C6 | 7 | 2.8972 | 18.1225 | 0.6824 | 8.9636 | 12,288 | 1 | 21.0 | 1.000 |
| Raw-SetBCE | 7 | 3.6132 | 37.0830 | 1.3984 | 27.9242 | 12,288 | 27 | 21.0 | 1.000 |
| OPAL-SetBCE | 7 | 3.9848 | 53.7724 | 1.7700 | 44.6135 | 12,288 | 31 | 21.0 | 1.000 |

## Router Training And Overlap

The full training/validation ledger is maintained in `docs/WIKITEXT2_PUBLIC_LM_RELATED_RESULTS.md`. The key train diagnostics pasted from the server are:

| method | epochs | loss first ↓ | loss best ↓ | best epoch | loss last ↓ | overlap@7 ↑ | hamming ↓ | exact match ↑ | unique predicted masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw-SetBCE | 40 | 0.207188 | 0.082307 | 40 | 0.082307 | 0.940286 | 0.029857 | 0.640500 | 39 |
| OPAL-H4-SetBCE | 40 | 0.564387 | 0.010671 | 40 | 0.010671 | 0.996429 | 0.001786 | 0.975500 | 53 |
| OPAL-H9-SetBCE | 40 | 0.564667 | 0.015140 | 40 | 0.015140 | 0.992214 | 0.003893 | 0.951500 | 46 |

All three train losses are best at epoch 40, but OPAL-H4 validation PPL is best at epoch 2. This is why the current diagnosis is overfitting/objective mismatch rather than insufficient training.

## Leakage Check

- Greedy skip-set labels are built only from WikiText-2 train windows.
- Static best-on-val C6 is selected only on validation PPL.
- Final PPL is reported on test windows.
- Eval code does not load greedy labels for Full/static/router PPL.
- Routers see only the first `router_prefix_tokens` tokens.
- PPL is scored only on suffix tokens by setting prefix labels to `-100`.

## Interpretation

The final OPAL-SetBCE checkpoint beats Static best-on-val C6 by 0.0867 NLL and 1.4408 PPL, but Raw-SetBCE beats that final OPAL checkpoint by 0.0054 NLL and 0.0855 PPL. Validation checkpoint selection changes the seed42 result: OPAL best-on-val epoch2 reaches PPL 15.6204, ahead of Raw-SetBCE PPL 15.8283 and `layerwise_hidden_router` PPL 15.7491. Because the selected checkpoint has only one unique test mask, it is checkpoint-selection rescue rather than dynamic-mask proof. The dynamic-constrained epoch33 checkpoint has 13 test unique masks and PPL 15.8695, improving final OPAL but not Raw/layerwise. WikiText-2 should therefore be treated as appendix-level public LM sanity / checkpoint-selection evidence, not a main OPAL superiority claim.
