# WikiText-2 Public LM Related Baseline Results

Last updated: 2026-06-02

## Setup

- model path: `/workspace/ckpts/Qwen2.5-1.5B`
- dataset: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- num_layers: 28
- seq_len: 1024
- router_prefix_tokens: 256
- eval_windows requested / actual: 512 / 289
- eval_tokens: 221,952
- skip_rate: 0.25
- skip_count: 7
- protected_head / protected_tail: 4 / 2
- label_samples: 2000
- candidate library: C16 `uniform`, `ends_heavy`, `first_k`, `last_k`, `middle_heavy`, `random_diverse_seed42`, `random_diverse_v1`...`random_diverse_v10`

## Complete Experiment Ledger

All entries below use Qwen2.5-1.5B, WikiText-2 raw, `skip_rate=0.25`, `skip_count=7`, `protected_head=4`, `protected_tail=2`, and seed 42 unless noted.

| run | seq_len | router_prefix_tokens | prefix_depth | label_samples | eval_tokens | Full PPL | Raw PPL | OPAL PPL | OPAL unique masks | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smoke after mask fix | 512 | 128 | 4 | 32 | 12,288 | 9.1589 | 37.0830 | 53.7724 | 31 | valid smoke only |
| main final checkpoint | 1024 | 256 | 4 | 2000 | 221,952 | 9.1649 | 15.8283 | 15.9138 | 15 | final OPAL loses Raw by 0.0855 PPL |
| related baselines | 1024 | 256 | 4 | 2000 | 221,952 | 9.1649 | 15.8283 | 15.9138 | 15 | PuDDing/IG/layerwise added |
| prefix length sensitivity | 1024 | 512 | 4 | 2000 | 147,968 | 8.8763 | 15.5530 | 15.7777 | 21 | prefix512 does not rescue OPAL |
| OPAL-H9 | 1024 | 256 | 9 | 2000 | 221,952 | 9.1649 | 15.8283 | 16.0055 | 18 | deeper H^k does not rescue OPAL |
| OPAL-H4 best-on-val | 1024 | 256 | 4 | 2000 | 221,952 | 9.1649 | 15.8283 final | 15.6204 | 1 | epoch2 wins current Raw/layerwise, but collapses to one mask |
| OPAL-H4 best-on-val minuniq8 | 1024 | 256 | 4 | 2000 | 221,952 | 9.1649 | 15.8283 final | 15.8695 | 13 | dynamic checkpoint improves final OPAL but loses current Raw/layerwise |
| Raw best-on-val | 1024 | 256 | 0 | 2000 | 221,952 | 9.1649 | 15.6894 | NA | 7 | epoch24 improves Raw final but remains behind OPAL epoch2 |
| three-seed best-on-val | 1024 | 256 | 4 | 2000 | 221,952 | 9.1649 mean | 15.6918 mean | 15.6207 mean | 1.3333 mean | OPAL wins Raw best-on-val/layerwise by mean PPL, but is static-like |

## Three-Seed Supplement Status

Fixed three-seed result document:

- `docs/WIKITEXT2_3SEED_PUBLIC_LM_RESULTS.md`

Seed status:

| seed | status | note |
|---:|---|---|
| 42 | complete | do not rerun; all required methods including Raw/OPAL best-on-val are present |
| 13 | complete | all required methods including Raw/OPAL best-on-val are present |
| 3407 | complete | all required methods including Raw/OPAL best-on-val are present |

Three-seed headline:

| method | PPL mean ↓ | PPL std | NLL mean ↓ | unique_masks mean | key note |
|---|---:|---:|---:|---:|---|
| OPAL-SetBCE best-on-val | 15.6207 | 0.0005 | 2.7486 | 1.3333 | wins Raw best-on-val/layerwise by mean PPL, but is static-like |
| Raw-SetBCE best-on-val | 15.6918 | 0.0270 | 2.7531 | 6.0000 | validation-selected raw baseline |
| layerwise_hidden_router | 15.7403 | 0.0222 | 2.7562 | 7.6667 | stronger-access baseline |
| OPAL-SetBCE final epoch | 15.9816 | 0.0808 | 2.7714 | 16.6667 | dynamic but worse than Raw/layerwise |

Required judgments:

- OPAL best-on-val wins Raw best-on-val by three-seed mean PPL: `True`
- OPAL best-on-val wins `layerwise_hidden_router` by three-seed mean PPL: `True`
- Per-seed OPAL best-on-val wins Raw best-on-val: `[(42, True), (13, True), (3407, True)]`
- Per-seed OPAL best-on-val wins layerwise: `[(42, True), (13, True), (3407, True)]`
- OPAL best-on-val unique masks: `[1, 2, 1]`; mean `1.3333`
- PuDDing-style / IG-style both select `ends_heavy` for all 289 test windows on every seed.
- Caveat: this is a validation-selected static-like OPAL checkpoint result, not evidence of dynamic mask-diversity superiority.

## Original WikiText Bad Result

| method | type | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks |
|---|---|---:|---:|---:|---:|---:|
| Full | full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 |
| Static uniform | static | 4.0673 | 58.3987 | 1.8519 | 49.2338 | 1 |
| Static ends_heavy | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 |
| Static best-on-val C6 | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 |
| Raw-SetBCE | ablation | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 |
| OPAL-SetBCE | ours | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 |

For the final 40-epoch checkpoints, Raw-SetBCE is ahead of OPAL-SetBCE by 0.0054 NLL / 0.0855 PPL. Later validation checkpoint selection changes the OPAL-H4 result, so this row should be cited specifically as the final-checkpoint comparison.

## Validation Checkpoint Selection Result

The 40-epoch OPAL-H4 run was later evaluated checkpoint-by-checkpoint on validation. The best validation checkpoint is epoch 2:

| method | selection | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|
| OPAL-SetBCE best-on-val | validation epoch 2 | 2.7486 | 15.6204 | 0.5332 | 6.4555 | 1 | 1.000 |
| OPAL-SetBCE best-on-val minuniq8 | validation epoch 33, `unique_masks>=8` | 2.7644 | 15.8695 | 0.5490 | 6.7046 | 13 | 1.000 |

Validation-selection details:

- best_epoch: 2
- validation_NLL / validation_PPL: 2.7938 / 16.3424
- test_NLL / test_PPL: 2.7486 / 15.6204
- test_eval_tokens: 221,952
- unique_masks: 1

This checkpoint is better than Raw-SetBCE by 0.0132 NLL / 0.2079 PPL and better than `layerwise_hidden_router` by 0.0082 NLL / 0.1287 PPL on this seed42 test run. However, `unique_masks=1`, so this should be interpreted as validation checkpoint selection finding an OPAL checkpoint whose hard top-K scores collapse to one high-quality mask, not as evidence that dynamic OPAL mask diversity is helping on WikiText-2.

To check whether a more dynamic checkpoint exists, selection was repeated with `WIKITEXT_VALCKPT_MIN_UNIQUE_MASKS=8`. This selected epoch 33 (`validation_NLL=2.8022`, `validation_PPL=16.4809`, `validation_unique_masks=8`) and gave test NLL/PPL `2.7644 / 15.8695` with `test_unique_masks=13`. This dynamic checkpoint is better than the final epoch OPAL checkpoint but still slightly worse than Raw-SetBCE by 0.0026 NLL / 0.0412 PPL and worse than `layerwise_hidden_router` by 0.0076 NLL / 0.1204 PPL.

Raw validation checkpoint selection was also run. Raw best-on-val selects epoch 24 (`validation_NLL=2.7966`, `validation_PPL=16.3893`, `validation_unique_masks=4`) and gives test NLL/PPL `2.7530 / 15.6894` with `test_unique_masks=7`. This improves Raw final by 0.1399 PPL but still remains behind OPAL best-on-val epoch2 by 0.0689 PPL / 0.0044 NLL.

Raw best-on-val compact server output:

| field | value |
|---|---:|
| min_unique_masks | 0 |
| best_epoch | 24 |
| validation_NLL | 2.796628 |
| validation_PPL | 16.389290 |
| validation_unique_masks | 4 |
| test_NLL | 2.752983 |
| test_PPL | 15.689361 |
| test_eval_tokens | 221,952 |
| test_unique_masks | 7 |
| exact_skip_count_rate | 1.000 |

The raw 40-epoch validation table was not pasted into the chat; only the compact best-on-val JSON above is recorded here. The full per-epoch raw validation JSON files should be under `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_raw_valckpt/` on the server.

### OPAL-H4 Validation Checkpoint Curve

This is the complete 40-epoch validation sweep for OPAL-H4. It explains the apparent contradiction: the best validation-PPL checkpoint is epoch 2 but has only one validation mask; later checkpoints become more dynamic but validation PPL worsens.

| epoch | val_NLL ↓ | val_PPL ↓ | validation unique masks | exact-K | dominant mask count | dominant keep mask |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 2.793761 | 16.342371 | 1 | 1.000 | 253 | `1111111111000000011111111111` |
| 15 | 2.796481 | 16.386880 | 2 | 1.000 | 156 | `1111111111100000001111111111` |
| 19 | 2.796734 | 16.391028 | 4 | 1.000 | 151 | `1111111111100000001111111111` |
| 18 | 2.796926 | 16.394173 | 4 | 1.000 | 163 | `1111111111100000001111111111` |
| 13 | 2.797322 | 16.400659 | 2 | 1.000 | 180 | `1111111111100000001111111111` |
| 16 | 2.797689 | 16.406692 | 2 | 1.000 | 193 | `1111111111100000001111111111` |
| 14 | 2.797702 | 16.406901 | 2 | 1.000 | 200 | `1111111111100000001111111111` |
| 10 | 2.797829 | 16.408985 | 2 | 1.000 | 183 | `1111111111100000001111111111` |
| 17 | 2.798136 | 16.414018 | 3 | 1.000 | 194 | `1111111111100000001111111111` |
| 11 | 2.798759 | 16.424249 | 2 | 1.000 | 216 | `1111111111100000001111111111` |
| 12 | 2.798923 | 16.426942 | 2 | 1.000 | 218 | `1111111111100000001111111111` |
| 20 | 2.799472 | 16.435962 | 6 | 1.000 | 145 | `1111111111100000001111111111` |
| 25 | 2.799591 | 16.437918 | 6 | 1.000 | 137 | `1111111111100000001111111111` |
| 1 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 3 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 4 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 5 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 6 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 7 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 8 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 9 | 2.800673 | 16.455725 | 1 | 1.000 | 253 | `1111111111100000001111111111` |
| 33 | 2.802204 | 16.480923 | 8 | 1.000 | 152 | `1111111111100000001111111111` |
| 23 | 2.802472 | 16.485354 | 6 | 1.000 | 133 | `1111111111100000001111111111` |
| 24 | 2.802850 | 16.491585 | 7 | 1.000 | 134 | `1111111111100000001111111111` |
| 22 | 2.803736 | 16.506204 | 5 | 1.000 | 138 | `1111111111100000001111111111` |
| 29 | 2.804502 | 16.518842 | 10 | 1.000 | 142 | `1111111111100000001111111111` |
| 28 | 2.804655 | 16.521379 | 7 | 1.000 | 137 | `1111111111100000001111111111` |
| 31 | 2.804810 | 16.523943 | 8 | 1.000 | 134 | `1111111111100000001111111111` |
| 30 | 2.805284 | 16.531776 | 13 | 1.000 | 155 | `1111111111100000001111111111` |
| 21 | 2.805545 | 16.536086 | 6 | 1.000 | 148 | `1111111111100000001111111111` |
| 26 | 2.805935 | 16.542540 | 9 | 1.000 | 115 | `1111111111100000001111111111` |
| 35 | 2.806021 | 16.543954 | 10 | 1.000 | 128 | `1111111111100000001111111111` |
| 32 | 2.806731 | 16.555704 | 10 | 1.000 | 127 | `1111111111100000001111111111` |
| 27 | 2.806834 | 16.557418 | 7 | 1.000 | 126 | `1111111111100000001111111111` |
| 39 | 2.807405 | 16.566867 | 12 | 1.000 | 131 | `1111111111100000001111111111` |
| 40 | 2.807570 | 16.569604 | 15 | 1.000 | 129 | `1111111111100000001111111111` |
| 38 | 2.807625 | 16.570509 | 12 | 1.000 | 127 | `1111111111100000001111111111` |
| 37 | 2.807957 | 16.576013 | 11 | 1.000 | 115 | `1111111111100000001111111111` |
| 34 | 2.808994 | 16.593221 | 14 | 1.000 | 99 | `1111111111000000011111111111` |
| 36 | 2.812425 | 16.650242 | 11 | 1.000 | 103 | `1111111111100000001111111111` |

Dynamic-constrained eligible checkpoints with `validation unique masks >= 8`:

| rank | epoch | val_NLL ↓ | val_PPL ↓ | validation unique masks | test_NLL ↓ | test_PPL ↓ | test unique masks | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 33 | 2.802204 | 16.480923 | 8 | 2.764400 | 15.869519 | 13 | selected by minuniq8; improves final OPAL but loses current Raw/layerwise |
| 2 | 29 | 2.804502 | 16.518842 | 10 | pending | pending | pending | validation-only |
| 3 | 31 | 2.804810 | 16.523943 | 8 | pending | pending | pending | validation-only |
| 4 | 30 | 2.805284 | 16.531776 | 13 | pending | pending | pending | validation-only |
| 5 | 26 | 2.805935 | 16.542540 | 9 | pending | pending | pending | validation-only |

## WikiText-2 Audit

| check | status | note |
|---|---|---|
| score direction | pass | `y_l=1` means skip; BCE uses `binary_cross_entropy_with_logits(-pred, target)`, so skip labels push `pred` lower; inference skips the top-K lowest `pred` layers. |
| mask application | pass | After the `maskcfg` fix, static masks produce large nonzero deltas from Full; the earlier equal-NLL run is invalid and should not be cited. |
| exact budget | pass | Every pasted valid method has `exact_skip_count_rate=1.0` and average kept layers 21 for K=7. |
| protected policy | pass | Full/static/raw/OPAL and the new baselines use the same protected head/tail and K. |
| split leakage | pass | Train labels are train-only; static best uses validation; final PPL uses test; eval paths do not load greedy labels. |
| router input | pass | Routers see only the first `router_prefix_tokens`; PPL and labels score suffix tokens only. |
| full PPL sanity | pass | Full PPL 9.1649 is finite and plausible for Qwen2.5-1.5B on WikiText-2 raw. |
| static best selection | pass | C6 best is selected by validation PPL, not test PPL. |
| label objective | pass | Greedy labels optimize suffix Delta_NLL. New candidate labels also optimize suffix Delta_NLL. |

## Final-Epoch Failure And Low-Diversity Caveat

- Final-epoch OPAL beats Static best-on-val C6, PuDDing-style, and IG-style, so the mask application and learned routing path are not broken.
- Final-epoch OPAL loses to Raw-SetBCE and to the stronger-access layerwise hidden router, which shows that training to epoch 40 overfits the greedy set labels on WikiText-2.
- Train diagnostics support this: OPAL-H4/H9 fit train skip sets far better than Raw, but final test PPL is worse, so the issue is validation/generalization rather than insufficient optimization.
- Validation checkpoint selection fixes PPL: across three seeds, OPAL best-on-val beats Raw best-on-val and `layerwise_hidden_router` by mean PPL and on every seed.
- The caveat is severe: OPAL best-on-val uses very few masks (`[1, 2, 1]` unique masks across seeds 42/13/3407), so it behaves like a validation-selected static-like OPAL checkpoint.
- PuDDing-style and IG-style both collapse to the static `ends_heavy` candidate on all 289 test windows for every seed, so OPAL beating them mostly says OPAL beats candidate-library selection on this low-entropy layer-skip regime.
- The H9 deeper-prefix test did not rescue dynamic SetBCE: OPAL-H9 is worse than Raw-SetBCE and slightly worse than the original H4 OPAL final run.
- A constrained dynamic checkpoint exists on seed42 (`min_unique_masks>=8`, epoch33, 13 test unique masks), but it does not beat Raw-SetBCE or `layerwise_hidden_router`; dynamic-mask diversity is not the winning WikiText-2 story.

## Related Baseline Table

Server run completed:

| method | type | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 | 1.000 |
| Static uniform | static | 4.0673 | 58.3987 | 1.8519 | 49.2338 | 1 | 1.000 |
| Static ends_heavy | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| Static best-on-val C6 | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| PuDDing-style | related | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| IG-style | related | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 |
| layerwise_hidden_router | related | 2.7568 | 15.7491 | 0.5414 | 6.5842 | 8 | 1.000 |
| Raw-SetBCE final epoch | ablation | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 | 1.000 |
| Raw-SetBCE best-on-val epoch24 | ablation + val selection | 2.7530 | 15.6894 | 0.5376 | 6.5245 | 7 | 1.000 |
| OPAL-SetBCE best-on-val minuniq8 epoch33 | ours + val selection | 2.7644 | 15.8695 | 0.5490 | 6.7046 | 13 | 1.000 |
| OPAL-SetBCE | ours | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 | 1.000 |
| OPAL-SetBCE best-on-val epoch2 | ours + val selection | 2.7486 | 15.6204 | 0.5332 | 6.4555 | 1 | 1.000 |

Dynamic ranking among learned/adaptive methods:

| rank | method | note |
|---:|---|---|
| 1 | OPAL-SetBCE best-on-val epoch2 | seed42 validation-selected checkpoint; lowest test PPL among evaluated skip methods, but collapses to one test mask |
| 2 | Raw-SetBCE best-on-val epoch24 | fair raw validation-selected comparison; improves Raw final but still behind OPAL epoch2 |
| 3 | layerwise_hidden_router | stronger-access in-framework baseline; not a full Dr.LLM reproduction |
| 4 | Raw-SetBCE final epoch | same final Delta_NLL set labels, raw prefix only |
| 5 | OPAL-SetBCE best-on-val minuniq8 epoch33 | dynamic checkpoint with 13 test unique masks; better than final OPAL but slightly behind Raw/layerwise |
| 6 | OPAL-SetBCE final epoch | ours; beats static/PuDDing/IG but not Raw/layerwise before checkpoint selection |
| 7 | PuDDing-style / IG-style | both select `ends_heavy` for every test window |

Selected candidate distributions:

- PuDDing-style: `{"ends_heavy": 289}`
- IG-style: `{"ends_heavy": 289}`

## Skip Distribution Diagnosis

All learned routers obey the protected-head policy: 0-based layers `0,1,2,3` are never skipped. The actual skipped layers concentrate in the middle block, especially `10..17`.

| method | windows | unique masks | dominant skipped set | dominant rate | key pattern |
|---|---:|---:|---|---:|---|
| Raw-SetBCE | 289 | 8 | `[11, 12, 13, 14, 15, 16, 17]` | 0.606 | mostly shifted middle-block masks |
| OPAL-SetBCE | 289 | 15 | `[10, 11, 12, 13, 14, 15, 16]` | 0.453 | more diverse but slightly worse PPL |
| layerwise_hidden_router | 289 | 8 | `[11, 12, 13, 14, 15, 16, 17]` | 0.678 | closest to a strong shifted middle-block selector |

Pairwise agreement:

| pair | mean overlap@7 | mean hamming/28 | exact same mask |
|---|---:|---:|---:|
| Raw-SetBCE vs OPAL-SetBCE | 0.8942 | 0.0529 | 0.3875 |
| Raw-SetBCE vs layerwise_hidden_router | 0.9402 | 0.0299 | 0.6125 |
| OPAL-SetBCE vs layerwise_hidden_router | 0.9046 | 0.0477 | 0.4291 |

Interpretation: WikiText-2 is not asking the router to skip early layers. It mostly asks for a shifted middle-block skip policy. In the final checkpoints, OPAL's extra mask diversity does not translate into lower PPL; layerwise_hidden_router wins because it more reliably chooses the same high-quality middle-block masks. Validation selection can improve OPAL PPL, but the best checkpoint collapses to one mask; the dynamic-constrained checkpoint remains behind Raw/layerwise.

## Training Diagnostics

| method | epochs | loss first ↓ | loss best ↓ | best epoch | loss last ↓ | overlap@7 ↑ | hamming ↓ | exact match ↑ | unique predicted masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw-SetBCE | 40 | 0.207188 | 0.082307 | 40 | 0.082307 | 0.940286 | 0.029857 | 0.640500 | 39 |
| OPAL-H4-SetBCE | 40 | 0.564387 | 0.010671 | 40 | 0.010671 | 0.996429 | 0.001786 | 0.975500 | 53 |
| OPAL-H9-SetBCE | 40 | 0.564667 | 0.015140 | 40 | 0.015140 | 0.992214 | 0.003893 | 0.951500 | 46 |
| layerwise_hidden_router | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract | pending artifact extract |

Readout: Raw fits train labels less tightly but generalizes slightly better on test PPL. OPAL-H4/H9 nearly memorize the train greedy sets (`exact_match` 0.9755 / 0.9515), yet test PPL stays behind Raw. More epochs are unlikely to help this SetBCE formulation; if anything, validation checkpoint selection or a less overfit objective would be the next diagnostic.

The training loss curve is monotonic in the pasted summaries: all three best losses occur at epoch 40. The validation curve tells a different story for OPAL-H4: validation PPL is best at epoch 2, long before train BCE reaches its best value. This supports the overfitting/objective-mismatch diagnosis.

The remaining `pairwise predicted hamming`, `unique teacher masks`, and layerwise training diagnostics can still be extracted from server artifacts with the command below if needed.

Server artifact extraction command:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate

RUN=wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25
ROOT=results/wikitext2_public_lm_sanity
CKPT_BASE=policy_ckpts/wikitext2_public_lm_sanity/${RUN}
RELATED_CKPT=policy_ckpts/wikitext2_public_lm_related/${RUN}
DIAG=${ROOT}/diagnostics/${RUN}
LABEL=${ROOT}/labels/${RUN}_delta_nll_greedy_set_labels.jsonl
MODEL=/workspace/ckpts/Qwen2.5-1.5B
DATA=/workspace/datasets/wikitext/wikitext-2-raw-v1

mkdir -p "$DIAG"

# Layerwise overlap was not part of the base sanity run; generate it if missing.
if [ -s "${RELATED_CKPT}/layerwise_hidden_bce/risk_router.pt" ] && [ ! -s "${DIAG}/layerwise_hidden_router_train_overlap.jsonl" ]; then
  accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 58400 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    ./eval_wikitext_opal_ppl.py diagnose_overlap \
      --teacher_model "$MODEL" \
      --split train \
      --dataset_disk_path "$DATA" \
      --seq_len 1024 \
      --router_prefix_tokens 256 \
      --risk_label_file "$LABEL" \
      --risk_router_ckpt "${RELATED_CKPT}/layerwise_hidden_bce/risk_router.pt" \
      --protected_head 4 \
      --protected_tail 2 \
      --batch_size 1 \
      --output_jsonl "${DIAG}/layerwise_hidden_router_train_overlap.jsonl" \
      --summary_json "${DIAG}/layerwise_hidden_router_train_overlap.summary.json" \
      --precision bf16 \
      --seed 42
fi

python3 - <<'PY'
import itertools, json, math, os

run = "wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25"
root = "results/wikitext2_public_lm_sanity"
diag = f"{root}/diagnostics/{run}"
paths = {
    "Raw-SetBCE": {
        "train": f"policy_ckpts/wikitext2_public_lm_sanity/{run}/raw_embedding_bce/training_metrics.json",
        "overlap": f"{diag}/raw_setbce_train_overlap.jsonl",
    },
    "OPAL-SetBCE": {
        "train": f"policy_ckpts/wikitext2_public_lm_sanity/{run}/prefix_hk_raw_attn_bce/training_metrics.json",
        "overlap": f"{diag}/opal_setbce_train_overlap.jsonl",
    },
    "layerwise_hidden_router": {
        "train": f"policy_ckpts/wikitext2_public_lm_related/{run}/layerwise_hidden_bce/training_metrics.json",
        "overlap": f"{diag}/layerwise_hidden_router_train_overlap.jsonl",
    },
}

def fmt(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    return "NA" if not math.isfinite(x) else f"{x:.6f}"

def loss_summary(path):
    if not os.path.exists(path):
        return None, None, None
    hist = (json.load(open(path)).get("history") or [])
    if not hist:
        return None, None, None
    first = hist[0]
    best = min(hist, key=lambda r: float(r.get("loss", "inf")))
    last = hist[-1]
    return first.get("loss"), best.get("loss"), last.get("loss")

def mask_tuple(layers):
    return tuple(sorted(int(x) for x in layers))

def pairwise_hamming(keys, n_layers=28):
    keys = [set(k) for k in keys]
    if len(keys) < 2:
        return 0.0
    vals = []
    for a, b in itertools.combinations(keys, 2):
        vals.append(len(a ^ b) / n_layers)
    return sum(vals) / len(vals)

def overlap_summary(path):
    if not os.path.exists(path):
        return (None,) * 6
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        return (None,) * 6
    overlap = sum(float(r["overlap_ratio"]) for r in rows) / len(rows)
    hamming = sum(float(r["hamming_ratio"]) for r in rows) / len(rows)
    exact = sum(float(r["exact_match"]) for r in rows) / len(rows)
    teacher = [mask_tuple(r["label_skipped_layers"]) for r in rows]
    pred = [mask_tuple(r["predicted_skipped_layers"]) for r in rows]
    return overlap, hamming, pairwise_hamming(pred), exact, len(set(teacher)), len(set(pred))

print("| method | loss first | loss best | loss last | overlap@7 | hamming | pairwise predicted hamming | exact match | unique teacher masks | unique predicted masks |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, p in paths.items():
    first, best, last = loss_summary(p["train"])
    overlap, hamming, pairwise, exact, uniq_teacher, uniq_pred = overlap_summary(p["overlap"])
    print(f"| {name} | {fmt(first)} | {fmt(best)} | {fmt(last)} | {fmt(overlap)} | {fmt(hamming)} | {fmt(pairwise)} | {fmt(exact)} | {uniq_teacher or 'NA'} | {uniq_pred or 'NA'} |")
PY
```

## OPAL Tuning Table

| setting | status | Full NLL/PPL | Raw-SetBCE NLL/PPL | OPAL-SetBCE NLL/PPL | conclusion |
|---|---|---:|---:|---:|---|
| prefix tokens 256 | complete | 2.2154 / 9.1649 | 2.7618 / 15.8283 | 2.7672 / 15.9138 | OPAL beats static/PuDDing/IG but loses Raw by 0.0054 NLL. |
| prefix tokens 512 | complete | 2.1834 / 8.8763 | 2.7443 / 15.5530 | 2.7586 / 15.7777 | OPAL still loses Raw by 0.0143 NLL; do not expand SetBCE to 3 seeds. |
| OPAL-H9 (`prefix_depth=9`) | complete | 2.2154 / 9.1649 | 2.7618 / 15.8283 | 2.7729 / 16.0055 | H9 still loses Raw by 0.0111 NLL and is worse than H4 OPAL by 0.0057 NLL. |
| validation checkpoint selection | complete | 2.2154 / 9.1649 | 2.7618 / 15.8283 | 2.7486 / 15.6204 | Best-on-val epoch2 beats Raw/layerwise on seed42, but collapses to one test mask; cite as checkpoint-selection rescue, not dynamic-mask proof. |
| validation checkpoint selection, minuniq8 | complete | 2.2154 / 9.1649 | 2.7618 / 15.8283 | 2.7644 / 15.8695 | Dynamic epoch33 has 13 test unique masks and improves final OPAL, but remains slightly behind Raw/layerwise. |
| Raw validation checkpoint selection | complete | 2.2154 / 9.1649 | 2.7530 / 15.6894 | NA | Raw best-on-val epoch24 improves Raw final, but remains behind OPAL best-on-val epoch2. |
| three-seed validation checkpoint selection | complete | 2.2154 / 9.1649 | 2.7531 / 15.6918 mean | 2.7486 / 15.6207 mean | OPAL best-on-val beats Raw best-on-val and layerwise by mean PPL across seeds 42/13/3407, but OPAL unique masks average only 1.3333. |
| static prior / swap q=1/2 | pending | NA | NA | NA | Defer unless old OPAL baselines or prefix variants show a path to beat Raw. |

Do not compare absolute Full PPL across prefix lengths: `prefix256` scores 768 suffix tokens per window, while `prefix512` scores 512 suffix tokens per window. Compare methods within the same prefix setting using Delta_NLL/Delta_PPL.

## Gap Table

| missing item | status | priority | why it matters | action |
|---|---|---:|---|---|
| random_dynamic_hash | missing | P2 | sanity lower-bound for prompt-conditioned dynamic mask diversity | add/run only after old OPAL baselines are in |
| OPAL-PrefixLast old | missing on WikiText-2 | P0 | old submission-style OPAL reference; needed to know whether SetBCE is worse than previous OPAL formulation | run seed42 first |
| OPAL-Attn old one-layer | missing on WikiText-2 | P0 | old one-layer attention OPAL ablation; needed for continuity with prior tables | run seed42 first |
| static best-on-val C16 | optional missing | P2 | PuDDing/IG use C16; static best currently C6, so C16 static can check whether candidate library itself contains a stronger static mask | optional after P0 |
| 3 seeds | complete | P0 | required to decide whether WikiText-2 can be positive public LM sanity | OPAL best-on-val wins Raw best-on-val and layerwise by mean PPL, but with low unique masks |
| prefix length sensitivity | seed42 prefix512 complete | P1 | likely rescue axis for OPAL-SetBCE on long LM windows | did not rescue SetBCE; OPAL still loses Raw |
| OPAL-H9 / deeper H^k feature | complete | P0 | skip distribution shows layers 0..8 are effectively never skipped; H9 tested whether a deeper prefix hidden state helps | did not rescue SetBCE; OPAL-H9 still loses Raw |
| validation checkpoint selection | complete for three seeds | P1 | current training may not choose best validation-PPL checkpoint | OPAL best-on-val wins Raw best-on-val/layerwise across all three seeds, but unique_masks are `[1, 2, 1]` |
| Raw validation checkpoint selection | complete | P0 | needed for a fair best-on-val comparison against OPAL epoch2 | Raw epoch24 improves Raw final but remains behind OPAL epoch2 |

## Next Minimal Experiments

Priority order:

1. Run OPAL-PrefixLast old and OPAL-Attn old one-layer on WikiText-2 seed42. These are the most important missing historical OPAL baselines.
2. Do not run more OPAL-SetBCE tuning for this WikiText-2 sanity unless the paper needs a dynamic-mask-diversity claim. The current three-seed result already supports a validation-selected public LM sanity claim.
3. If the paper needs a dynamic-mask-diversity claim on public LM, a different public LM benchmark or a stricter dynamic selection criterion is needed; the current best-on-val OPAL checkpoints are static-like.

OPAL-H9 command already run:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=256 \
WIKITEXT_PREFIX_DEPTH=9 \
WIKITEXT_RUN_RAW=0 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58500 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

If "第 9 层输出" means 0-based layer index 9 after running layers `0..9`, use `WIKITEXT_PREFIX_DEPTH=10`; if it means the 9th layer in paper/1-based numbering, use `WIKITEXT_PREFIX_DEPTH=9`.

Prefix512 rescue command already run:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=512 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58300 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

Old OPAL baselines are not wired into the current WikiText-2 runner yet. Do not claim they were run until a WikiText-specific old-baseline harness reports `OPAL-PrefixLast old` and `OPAL-Attn old one-layer` with the same model, split, K, protected policy, and suffix-PPL evaluator.

## Final Judgment

- current decision: **positive public LM sanity with a low-diversity caveat**
- reason: across seeds `42/13/3407`, OPAL-SetBCE best-on-val has mean PPL `15.6207`, beating Raw-SetBCE best-on-val mean PPL `15.6918` and `layerwise_hidden_router` mean PPL `15.7403`. It also beats Raw/layerwise on every individual seed. However, OPAL best-on-val unique masks are `[1, 2, 1]`, so the best checkpoints are static-like.
- recommended wording: WikiText-2 is a positive public LM sanity for validation-selected OPAL-SetBCE PPL, not evidence that dynamic mask diversity wins on public LM.
- do not write: "OPAL's dynamic mask diversity dominates on WikiText-2." The three-seed table does not support that.
- safe write: "On WikiText-2, validation-selected OPAL-SetBCE achieves lower three-seed mean PPL than Raw-SetBCE best-on-val, a stronger-access layerwise hidden router, fixed static masks, and candidate-library PuDDing/IG-style baselines. The selected OPAL checkpoints are low-diversity (`unique_masks` near 1), so we interpret this as a static-like validation-selected public LM sanity rather than a dynamic mask-diversity result."

## Commands

Run related baselines:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=256 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58200 \
WIKITEXT_RUN_PUDDING=1 \
WIKITEXT_RUN_IG=1 \
WIKITEXT_RUN_LAYERWISE=1 \
bash ./run_wikitext2_related_baselines_gpu01234567.sh
```

Run Raw validation checkpoint selection:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
git pull --ff-only origin codex/opal-llm-experiments

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=256 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58700 \
WIKITEXT_VALCKPT_PARALLEL_WORKERS=8 \
bash ./run_wikitext2_raw_val_ckpt_select_gpu01234567.sh
```

Minimal OPAL correction if needed:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
WIKITEXT_LABEL_SAMPLES=2000 \
WIKITEXT_EVAL_WINDOWS=512 \
WIKITEXT_SEQ_LEN=1024 \
WIKITEXT_ROUTER_PREFIX_TOKENS=512 \
WIKITEXT_SEED=42 \
WIKITEXT_BASE_PORT=58300 \
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

## Artifact Paths

- base metrics: `results/wikitext2_public_lm_sanity/metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/`
- greedy labels: `results/wikitext2_public_lm_sanity/labels/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_delta_nll_greedy_set_labels.jsonl`
- candidate labels: `results/wikitext2_public_lm_sanity/related_labels/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_c16_candidate_delta_nll.jsonl`
- related metrics: `results/wikitext2_public_lm_sanity/related_metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/`
- related checkpoints: `policy_ckpts/wikitext2_public_lm_related/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25/`
- OPAL validation checkpoint metrics: `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_valckpt/`
- OPAL validation checkpoint checkpoints: `policy_ckpts/wikitext2_public_lm_val_ckpt/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_valckpt/`
- Raw validation checkpoint metrics: `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_raw_valckpt/`
- Raw validation checkpoint checkpoints: `policy_ckpts/wikitext2_public_lm_val_ckpt/wikitext2_Qwen2_5-1_5B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_raw_valckpt/`
