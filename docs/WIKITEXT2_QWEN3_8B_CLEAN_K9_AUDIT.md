# WikiText-2 Qwen3-8B Clean K9 Audit Log

Last updated: 2026-06-04

This note records the Qwen3-8B WikiText-2 K9 experiments and debugging artifacts that were not covered by the original Qwen2.5-1.5B WikiText-2 result docs.

## Fixed Setting

| field | value |
|---|---|
| model | `/workspace/Models/Qwen3-8B` |
| dataset | `/workspace/datasets/wikitext/wikitext-2-raw-v1` |
| split protocol | WikiText-2 token stream windows |
| seed | `42` |
| seq_len | `1024` |
| router_prefix_tokens | `256` |
| label_samples | `2000` train windows |
| eval_windows requested / actual | `512` requested, `289` actual test windows |
| eval_tokens | `221,952` |
| skip_rate | `0.25` |
| resolved skip_count | `9` |
| protected_head / protected_tail | `4` / `2` |
| allowed layers | `4..33` in 0-based Qwen3-8B layer indexing |
| clean label/run id | `wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_seq1024_pref256_m2000_seed42_skip0p25` |

## Invalid Old Qwen3 Artifacts

The old Qwen3 label file was invalid and must not be cited:

`results/wikitext2_public_lm_sanity/labels/wikitext2_Qwen3-8B_maskcfg_seq1024_pref256_m2000_seed42_skip0p25_delta_nll_greedy_set_labels.jsonl`

Hard evidence from that label file:

| statistic | value |
|---|---:|
| NLL_full mean | 10.3924 |
| NLL_full median | 10.4163 |
| PPL_full mean | 49367.42 |
| PPL_full median | 33399.01 |
| PPL_full max | 609379.67 |
| NLL_skip mean | 8.035 |
| Delta_NLL mean | -2.357 |

This matches the bad old Full PPL scale (`~35191`). The teacher labels were poisoned because the Full scoring path was already broken. Any router trained from that label file is invalid.

Invalid downstream symptoms from the poisoned chain:

| artifact/result | value | status |
|---|---:|---|
| old Full PPL | `35191.2789` | invalid |
| old Raw best-on-val epoch | `20` | invalid |
| old Raw validation PPL | `188.7621` | invalid |
| old Raw test PPL | `223.7019` | invalid |
| old Raw test unique_masks | `145` | invalid |

Code safeguards added after this finding:

| commit | change |
|---|---|
| `d7446c2` | use `AutoModelForCausalLM` in the WikiText label builder so Qwen3 is not loaded through a Qwen2 class |
| `773d19b` | add `check_wikitext_label_sanity.py`, label reuse fail-fast, and resolved `_K` tags in run ids |
| `513be4e` | add WikiText Exact-K CE router loss support for the loss ablation below |

## Clean Label Sanity

Clean label file:

`results/wikitext2_public_lm_sanity/labels/wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_seq1024_pref256_m2000_seed42_skip0p25_delta_nll_greedy_set_labels.jsonl`

| statistic | mean | median | min | max | p1 | p5 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NLL_full | 2.3008 | 2.3102 | 0.8620 | 3.3659 | 1.4754 | 1.7193 | 2.8464 | 3.0250 |
| PPL_full | 10.5508 | 10.0764 | 2.3680 | 28.9590 | 4.3729 | 5.5807 | 17.2257 | 20.5945 |
| NLL_skip | 2.9386 | 2.9428 | 1.2876 | 3.9171 | 2.1362 | 2.3494 | 3.4700 | 3.6732 |
| PPL_skip | 19.9718 | 18.9694 | 3.6239 | 50.2545 | 8.4671 | 10.4792 | 32.1381 | 39.3793 |
| Delta_NLL | 0.6378 | 0.6304 | 0.2580 | 2.3678 | 0.3820 | 0.4532 | 0.8541 | 0.9952 |
| Delta_PPL | 9.4210 | 8.7690 | 1.0356 | 32.7864 | 3.3725 | 4.5761 | 16.2158 | 20.6018 |

Readout: the clean label file is sane. Full NLL/PPL are near the clean Full test metric scale, and skipped masks are consistently worse than Full (`Delta_NLL > 0`).

## Clean Public LM Sanity

Run id:

`wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_seq1024_pref256_m2000_seed42_skip0p25`

| method | NLL | PPL | Delta_NLL | Delta_PPL | eval_tokens | unique_masks | average_kept_layers | exact_skip_count_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 2.2523 | 9.5100 | 0.0000 | 0.0000 | 221,952 | 1 | 36.0 | 1.000 |
| Static uniform | 7.3000 | 1480.3265 | 5.0477 | 1470.8165 | 221,952 | 1 | 27.0 | 1.000 |
| Static ends_heavy | 3.7233 | 41.4008 | 1.4710 | 31.8908 | 221,952 | 1 | 27.0 | 1.000 |
| Static best-on-val C6 | 3.2548 | 25.9140 | 1.0024 | 16.4040 | 221,952 | 1 | 27.0 | 1.000 |
| Raw-SetBCE final | 3.0151 | 20.3915 | 0.7628 | 10.8814 | 221,952 | 224 | 27.0 | 1.000 |
| OPAL-SetBCE final | 3.0341 | 20.7827 | 0.7818 | 11.2727 | 221,952 | 247 | 27.0 | 1.000 |

Readout: clean Qwen3 dynamic BCE routers are valid and beat static baselines, but final OPAL-SetBCE is slightly behind final Raw-SetBCE.

## Clean Related Baselines

Run id:

`wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_seq1024_pref256_m2000_seed42_skip0p25`

| method | NLL | PPL | Delta_NLL | Delta_PPL | unique_masks | selected_candidate_distribution |
|---|---:|---:|---:|---:|---:|---|
| Full | 2.2523 | 9.5100 | 0.0000 | 0.0000 | 1 | `{}` |
| Static uniform | 7.3000 | 1480.3265 | 5.0477 | 1470.8165 | 1 | `{}` |
| Static ends_heavy | 3.7233 | 41.4008 | 1.4710 | 31.8908 | 1 | `{}` |
| Static best-on-val C6 | 3.2548 | 25.9140 | 1.0024 | 16.4040 | 1 | `{}` |
| PuDDing-style | 3.2625 | 26.1159 | 1.0102 | 16.6059 | 5 | `{"random_diverse_seed42":195,"random_diverse_v7":88,"random_diverse_v1":4,"random_diverse_v3":1,"ends_heavy":1}` |
| IG-style | 3.2648 | 26.1749 | 1.0125 | 16.6649 | 2 | `{"random_diverse_v7":135,"random_diverse_seed42":154}` |
| layerwise_hidden_router | 3.0952 | 22.0914 | 0.8428 | 12.5814 | 204 | `{}` |
| Raw-SetBCE final | 3.0151 | 20.3915 | 0.7628 | 10.8814 | 224 | `{}` |
| OPAL-SetBCE final | 3.0341 | 20.7827 | 0.7818 | 11.2727 | 247 | `{}` |

Readout: final Raw/OPAL BCE beat Static/PuDDing/IG/layerwise on clean Qwen3 K9, but OPAL final still trails Raw final by `0.0190` NLL / `0.3913` PPL.

## BCE Validation Checkpoint Sweep

These rows use the same clean greedy labels. The key diagnostic is that lower train BCE and higher train-label imitation do not monotonically improve validation PPL.

### Raw-SetBCE

| epoch | train_BCE | val_NLL | val_PPL | unique_masks | exact_K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.374393 | 2.990910 | 19.9038 | 8 | 1.000 |
| 2 | 0.349563 | 3.043756 | 20.9839 | 28 | 1.000 |
| 3 | 0.343050 | 3.051619 | 21.1496 | 56 | 1.000 |
| 4 | 0.337542 | 2.998837 | 20.0622 | 38 | 1.000 |
| 5 | 0.331196 | 3.008967 | 20.2664 | 34 | 1.000 |
| 6 | 0.322741 | 3.019805 | 20.4873 | 50 | 1.000 |
| 7 | 0.317986 | 3.029829 | 20.6937 | 79 | 1.000 |
| 8 | 0.310860 | 3.018657 | 20.4638 | 55 | 1.000 |
| 9 | 0.304188 | 3.045103 | 21.0122 | 73 | 1.000 |
| 10 | 0.298413 | 3.104773 | 22.3041 | 64 | 1.000 |
| 11 | 0.291948 | 3.030052 | 20.6983 | 90 | 1.000 |
| 12 | 0.284876 | 3.038237 | 20.8684 | 88 | 1.000 |
| 13 | 0.278806 | 3.035444 | 20.8102 | 103 | 1.000 |
| 14 | 0.271766 | 3.046608 | 21.0439 | 116 | 1.000 |
| 15 | 0.265238 | 3.065272 | 21.4403 | 109 | 1.000 |
| 16 | 0.260239 | 3.112410 | 22.4751 | 129 | 1.000 |
| 17 | 0.255239 | 3.026836 | 20.6318 | 109 | 1.000 |
| 18 | 0.247980 | 3.037271 | 20.8483 | 117 | 1.000 |
| 19 | 0.242098 | 3.077168 | 21.6969 | 132 | 1.000 |
| 20 | 0.235148 | 3.039426 | 20.8932 | 106 | 1.000 |
| 21 | 0.228367 | 3.073470 | 21.6168 | 181 | 1.000 |
| 22 | 0.223444 | 3.044972 | 21.0094 | 135 | 1.000 |
| 23 | 0.218410 | 3.061691 | 21.3637 | 193 | 1.000 |
| 24 | 0.209973 | 3.066591 | 21.4686 | 161 | 1.000 |
| 25 | 0.204412 | 3.056868 | 21.2609 | 153 | 1.000 |
| 26 | 0.197990 | 3.064815 | 21.4305 | 178 | 1.000 |
| 27 | 0.194082 | 3.042340 | 20.9542 | 183 | 1.000 |
| 28 | 0.189392 | 3.074661 | 21.6426 | 168 | 1.000 |
| 29 | 0.182056 | 3.069361 | 21.5281 | 179 | 1.000 |
| 30 | 0.176451 | 3.049506 | 21.1049 | 170 | 1.000 |
| 31 | 0.171219 | 3.075237 | 21.6550 | 189 | 1.000 |
| 32 | 0.167588 | 3.062392 | 21.3786 | 189 | 1.000 |
| 33 | 0.160754 | 3.087107 | 21.9136 | 201 | 1.000 |
| 34 | 0.155927 | 3.093048 | 22.0442 | 191 | 1.000 |
| 35 | 0.149100 | 3.065284 | 21.4405 | 194 | 1.000 |
| 36 | 0.144549 | 3.091514 | 22.0104 | 194 | 1.000 |
| 37 | 0.139568 | 3.070732 | 21.5577 | 184 | 1.000 |
| 38 | 0.134186 | 3.064746 | 21.4290 | 196 | 1.000 |
| 39 | 0.129668 | 3.087750 | 21.9277 | 200 | 1.000 |
| 40 | 0.125054 | 3.070935 | 21.5621 | 199 | 1.000 |

### OPAL-SetBCE

| epoch | train_BCE | val_NLL | val_PPL | unique_masks | exact_K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.556382 | 2.988282 | 19.8515 | 1 | 1.000 |
| 2 | 0.370633 | 2.988282 | 19.8515 | 1 | 1.000 |
| 3 | 0.358127 | 2.988282 | 19.8515 | 1 | 1.000 |
| 4 | 0.356355 | 2.988282 | 19.8515 | 1 | 1.000 |
| 5 | 0.354624 | 2.988282 | 19.8515 | 1 | 1.000 |
| 6 | 0.352938 | 2.992013 | 19.9258 | 2 | 1.000 |
| 7 | 0.350425 | 3.010585 | 20.2993 | 4 | 1.000 |
| 8 | 0.347609 | 3.008746 | 20.2620 | 5 | 1.000 |
| 9 | 0.344081 | 3.011204 | 20.3118 | 11 | 1.000 |
| 10 | 0.339437 | 3.040322 | 20.9120 | 24 | 1.000 |
| 11 | 0.333630 | 3.032787 | 20.7550 | 35 | 1.000 |
| 12 | 0.326672 | 3.040850 | 20.9230 | 38 | 1.000 |
| 13 | 0.318680 | 3.012819 | 20.3447 | 40 | 1.000 |
| 14 | 0.307667 | 3.049618 | 21.1073 | 81 | 1.000 |
| 15 | 0.292456 | 3.047349 | 21.0594 | 85 | 1.000 |
| 16 | 0.273545 | 3.072908 | 21.6046 | 111 | 1.000 |
| 17 | 0.256243 | 3.059674 | 21.3206 | 134 | 1.000 |
| 18 | 0.236678 | 3.064502 | 21.4238 | 162 | 1.000 |
| 19 | 0.221407 | 3.059527 | 21.3175 | 178 | 1.000 |
| 20 | 0.201331 | 3.071655 | 21.5776 | 208 | 1.000 |
| 21 | 0.188151 | 3.049933 | 21.1139 | 174 | 1.000 |
| 22 | 0.169136 | 3.059689 | 21.3209 | 208 | 1.000 |
| 23 | 0.156487 | 3.059454 | 21.3159 | 192 | 1.000 |
| 24 | 0.139813 | 3.070771 | 21.5585 | 204 | 1.000 |
| 25 | 0.125039 | 3.063360 | 21.3993 | 213 | 1.000 |
| 26 | 0.113403 | 3.096658 | 22.1239 | 231 | 1.000 |
| 27 | 0.099242 | 3.095870 | 22.1065 | 224 | 1.000 |
| 28 | 0.095791 | 3.071453 | 21.5732 | 207 | 1.000 |
| 29 | 0.085695 | 3.070920 | 21.5617 | 212 | 1.000 |
| 30 | 0.073236 | 3.107218 | 22.3587 | 234 | 1.000 |
| 31 | 0.072787 | 3.096413 | 22.1185 | 227 | 1.000 |
| 32 | 0.064767 | 3.082556 | 21.8141 | 232 | 1.000 |
| 33 | 0.058876 | 3.071748 | 21.5796 | 224 | 1.000 |
| 34 | 0.051917 | 3.085964 | 21.8886 | 227 | 1.000 |
| 35 | 0.051517 | 3.088983 | 21.9547 | 231 | 1.000 |
| 36 | 0.045002 | 3.086957 | 21.9103 | 238 | 1.000 |
| 37 | 0.042204 | 3.074817 | 21.6459 | 229 | 1.000 |
| 38 | 0.037628 | 3.090628 | 21.9909 | 233 | 1.000 |
| 39 | 0.037042 | 3.073014 | 21.6069 | 226 | 1.000 |
| 40 | 0.038534 | 3.070349 | 21.5494 | 214 | 1.000 |

Best validation checkpoints:

| method | best_epoch | validation_NLL | validation_PPL | validation_unique_masks |
|---|---:|---:|---:|---:|
| Raw-SetBCE | 1 | 2.990910 | 19.9038 | 8 |
| OPAL-SetBCE | 1 | 2.988282 | 19.8515 | 1 |

## Train-Label Overlap Diagnostic

Only epoch 1 and epoch 40 were checked to avoid another full diagnostic sweep.

| method | epoch | train overlap@9 | hamming | exact_match | unique_pred_masks | paired val_PPL |
|---|---:|---:|---:|---:|---:|---:|
| Raw-SetBCE | 1 | 0.6518 | 0.1741 | 0.0025 | 11 | 19.9038 |
| Raw-SetBCE | 40 | 0.9514 | 0.0243 | 0.6245 | 1521 | 21.5621 |
| OPAL-SetBCE | 1 | 0.6492 | 0.1754 | 0.0025 | 1 | 19.8515 |
| OPAL-SetBCE | 40 | 0.9827 | 0.0086 | 0.8555 | 1606 | 21.5494 |

Readout: train-label imitation improves strongly from epoch 1 to epoch 40, but validation PPL gets worse. This is the concrete evidence for surrogate mismatch between hard greedy-mask BCE imitation and validation PPL.

## Current Interpretation

- The old Qwen3 results with Full PPL around `35191` and Raw best-on-val test PPL `223.70` are invalid.
- The clean Qwen3 K9 run is valid.
- Clean final Raw/OPAL BCE beat static/candidate/layerwise baselines, but OPAL final trails Raw final slightly.
- Clean best-on-val BCE selects very early checkpoints. OPAL best-on-val has the best validation PPL among Raw/OPAL BCE, but is low diversity (`unique_masks=1`).
- Because BCE gets lower while validation PPL worsens, Exact-K CE was tested on the same clean labels. It gives a small OPAL-side gain, but does not beat Raw BCE best and remains low-diversity.

Safe paper wording for Qwen3 after Exact-K:

> On clean Qwen3-8B WikiText-2 with 25% layer skipping (K=9), dynamic Raw/OPAL routers outperform fixed and candidate-library baselines. However, validation-selected Raw BCE remains ahead of OPAL under the tested BCE and Exact-K CE objectives; Exact-K CE provides only a small low-diversity OPAL-side improvement.

## Three-Seed Exact-K CE Loss Ablation

Detailed three-seed table: `docs/WIKITEXT2_QWEN3_3SEED_EXACTK_RESULTS.md`.

This ablation used the same clean greedy labels and compared BCE with Exact-K CE under the same Qwen3 K9 setting.

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| Full | 3 | 9.5100 | 0.0000 | 1.00 | `[9.5100, 9.5100, 9.5100]` |
| Raw BCE final | 3 | 20.4493 | 0.0698 | 220.67 | `[20.3915, 20.5474, 20.4089]` |
| OPAL BCE final | 3 | 21.1057 | 0.2653 | 256.00 | `[20.7827, 21.1016, 21.4327]` |
| Raw BCE best | 3 | 19.0200 | 0.0781 | 10.00 | `[19.0762, 19.0742, 18.9095]` |
| OPAL BCE best | 3 | 19.0497 | 0.0120 | 3.00 | `[19.0412, 19.0666, 19.0412]` |
| Raw ExactK best | 3 | 19.0754 | 0.1526 | 9.67 | `[19.0762, 19.2620, 18.8881]` |
| OPAL ExactK best | 3 | 19.0405 | 0.0011 | 1.33 | `[19.0389, 19.0412, 19.0412]` |

Per-seed OPAL Exact-K checkpoints:

| seed | best_epoch | validation_PPL | test_PPL | unique_masks |
|---:|---:|---:|---:|---:|
| 42 | 1 | 19.8515 | 19.0389 | 2 |
| 13 | 1 | 19.8515 | 19.0412 | 1 |
| 3407 | 1 | 19.8515 | 19.0412 | 1 |

Readout:

- Exact-K CE slightly improves OPAL best-on-val over OPAL BCE best: `19.0405` vs `19.0497` mean PPL.
- Exact-K CE does not make OPAL beat the strongest raw-prefix baseline: Raw BCE best remains better, `19.0200` vs `19.0405` mean PPL.
- Exact-K CE also does not solve dynamic diversity. OPAL Exact-K selects epoch 1 on every seed and has unique masks `[2, 1, 1]`.
- Raw Exact-K is worse than Raw BCE best on mean PPL, so Exact-K CE is not a general win for this Qwen3 K9 setup.

Updated Qwen3 conclusion:

> On clean Qwen3-8B WikiText-2 with 25% layer skipping (K=9), dynamic routers beat fixed and candidate-library baselines, but validation-selected Raw BCE remains the best skip method among the BCE/Exact-K CE loss ablation. OPAL Exact-K CE gives a small OPAL-side improvement but remains low-diversity and does not establish an OPAL-over-Raw win.
