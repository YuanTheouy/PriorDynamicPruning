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

This is better than the final OPAL checkpoint, Raw-SetBCE, and the related `layerwise_hidden_router` seed42 result. The caveat is `unique_masks=1`, so the best validation checkpoint behaves like a single high-quality OPAL-selected skip mask on test.

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

The compact server log pasted into the chat did not include the exact first/best/last training losses or the overlap/hamming summary values. They are written by the server runner to:

- Raw training metrics: `policy_ckpts/wikitext2_public_lm_sanity/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/raw_embedding_bce/training_metrics.json`
- OPAL training metrics: `policy_ckpts/wikitext2_public_lm_sanity/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/prefix_hk_raw_attn_bce/training_metrics.json`
- Raw overlap: `results/wikitext2_public_lm_sanity/diagnostics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/raw_setbce_train_overlap.summary.json`
- OPAL overlap: `results/wikitext2_public_lm_sanity/diagnostics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/opal_setbce_train_overlap.summary.json`

The related-baseline runner will read these JSON files and regenerate this report with exact loss/overlap values on the server. I am not filling those missing scalar values by guess.

## Leakage Check

- Greedy skip-set labels are built only from WikiText-2 train windows.
- Static best-on-val C6 is selected only on validation PPL.
- Final PPL is reported on test windows.
- Eval code does not load greedy labels for Full/static/router PPL.
- Routers see only the first `router_prefix_tokens` tokens.
- PPL is scored only on suffix tokens by setting prefix labels to `-100`.

## Interpretation

The final OPAL-SetBCE checkpoint beats Static best-on-val C6 by 0.0867 NLL and 1.4408 PPL, but Raw-SetBCE beats that final OPAL checkpoint by 0.0054 NLL and 0.0855 PPL. Validation checkpoint selection changes the seed42 result: OPAL best-on-val epoch2 reaches PPL 15.6204, ahead of Raw-SetBCE PPL 15.8283 and `layerwise_hidden_router` PPL 15.7491. Because the selected checkpoint has only one unique test mask and still lacks multi-seed confirmation, WikiText-2 should be treated as appendix-level public LM sanity / checkpoint-selection evidence, not a main OPAL superiority claim.
