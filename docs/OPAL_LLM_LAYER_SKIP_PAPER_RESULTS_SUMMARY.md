# OPAL LLM Layer-Skip Paper Results Summary

Last updated: 2026-06-09

This is the paper-facing summary for the public LM layer-skipping experiments. It consolidates the clean WikiText-2 PPL results, related-work style baselines, Exact-K CE ablation, Llama3.1 transfer check, and speed timing breakdown.

## Paper-Safe Main Claim

OPAL should be written as a validation-selected quality/speed layer-skipping method:

> On clean WikiText-2 with 25% layer skipping, OPAL with prefix1024 and validation checkpoint selection improves over Raw-SetBCE, PuDDing-style, IG-style, and layerwise hidden routing on Qwen3-8B and Llama3.1-8B-Instruct, while achieving measured wall-clock speedup over Full after physical layer skipping.

Do not write that final-epoch OPAL is always best. Do not write that OPAL is the fastest skipping implementation. The strongest evidence is validation-selected OPAL under prefix1024.

## Global Settings

| field | Qwen3-8B | Llama3.1-8B-Instruct |
|---|---|---|
| model path | `/workspace/Models/Qwen3-8B` | `/workspace/Models/Llama-3.1-8B-Instruct` |
| dataset | `/workspace/datasets/wikitext/wikitext-2-raw-v1` | `/workspace/datasets/wikitext/wikitext-2-raw-v1` |
| seeds | `42`, `13`, `3407` | `42`, `13`, `3407` |
| seq_len / router_prefix_tokens | `1536` / `1024` | `1536` / `1024` |
| scored suffix tokens | `512` | `512` |
| skip_rate / skip_count | `0.25` / `9` | `0.25` / `8` |
| protected_head / protected_tail | `4` / `2` | `4` / `2` |
| teacher | clean final Delta_NLL greedy set labels | clean final Delta_NLL greedy set labels |
| search | greedy, not beam2 | greedy, not beam2 |

## Qwen3-8B Main WikiText-2 PPL

Prefix1024 Qwen3-8B uses `1625` clean train label rows. The key result is that validation-selected OPAL wins, while final-epoch OPAL does not.

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| Raw BCE final | 3 | 19.0347 | 0.2370 | 156.33 | `[18.7511, 19.0220, 19.3311]` |
| OPAL BCE final | 3 | 19.2485 | 0.2213 | 180.67 | `[19.0376, 19.5543, 19.1537]` |
| Raw BCE best-on-val | 3 | 17.7729 | 0.0251 | 28.67 | `[17.7837, 17.7967, 17.7382]` |
| OPAL BCE best-on-val | 3 | 17.5974 | 0.0870 | 4.67 | `[17.5360, 17.7205, 17.5358]` |
| Raw Exact-K CE best-on-val | 3 | 17.7481 | 0.0587 | 30.67 | `[17.8252, 17.7361, 17.6830]` |
| OPAL Exact-K CE best-on-val | 3 | 17.6131 | 0.0630 | 10.67 | `[17.5403, 17.6940, 17.6049]` |

Qwen3 judgment:

- OPAL BCE best-on-val beats Raw BCE best-on-val on all three seeds.
- OPAL Exact-K CE best-on-val beats Raw Exact-K CE best-on-val on all three seeds.
- Final Raw BCE beats final OPAL BCE, so the claim must be validation-selected.
- OPAL BCE best-on-val is low-diversity: unique masks `[1, 11, 2]`.

## Qwen3-8B Related-Work Baselines

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| OPAL BCE best-on-val | 3 | 17.5974 | 0.0870 | 4.67 | `[17.5360, 17.7205, 17.5358]` |
| OPAL Exact-K CE best-on-val | 3 | 17.6131 | 0.0630 | 10.67 | `[17.5403, 17.6940, 17.6049]` |
| Raw BCE best-on-val | 3 | 17.7729 | 0.0251 | 28.67 | `[17.7837, 17.7967, 17.7382]` |
| Raw Exact-K CE best-on-val | 3 | 17.7481 | 0.0587 | 30.67 | `[17.8252, 17.7361, 17.6830]` |
| layerwise_hidden_router | 3 | 20.0723 | 0.1662 | 170.33 | `[20.3070, 19.9446, 19.9653]` |
| PuDDing-style | 3 | 23.0576 | 1.0998 | 3.67 | `[23.5776, 21.5282, 24.0670]` |
| IG-style | 3 | 22.7616 | 1.1255 | 1.33 | `[23.2062, 21.2157, 23.8629]` |

PuDDing-style and IG-style do not collapse to `ends_heavy` under prefix1024; they mostly select random-diverse library masks. Layerwise hidden routing has high mask diversity but worse PPL than OPAL best-on-val.

## Llama3.1-8B-Instruct Transfer Result

Llama3.1 uses `1572` clean train label rows and `K=8`. This checks that the Qwen3 prefix1024 result is not architecture-specific.

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| Full | 3 | 6.5076 | 0.0000 | 1.00 | `[6.5076, 6.5076, 6.5076]` |
| Static best-on-val C6 | 3 | 37.9201 | 3.6100 | 1.00 | `[40.3085, 40.6336, 32.8183]` |
| PuDDing-style | 3 | 24.4955 | 2.4254 | 4.33 | `[22.1651, 27.8404, 23.4812]` |
| IG-style | 3 | 24.3045 | 2.4773 | 1.67 | `[22.1129, 27.7673, 23.0332]` |
| layerwise_hidden_router | 3 | 17.3994 | 0.3013 | 137.00 | `[17.8157, 17.1124, 17.2701]` |
| Raw BCE final | 3 | 16.5592 | 0.1242 | 79.67 | `[16.4910, 16.7334, 16.4531]` |
| OPAL BCE final | 3 | 17.1170 | 0.1977 | 108.00 | `[17.3362, 16.8571, 17.1578]` |
| Raw BCE best-on-val | 3 | 16.2121 | 0.0247 | 18.67 | `[16.1786, 16.2202, 16.2373]` |
| OPAL BCE best-on-val | 3 | 16.0948 | 0.1238 | 11.00 | `[16.0023, 16.2698, 16.0123]` |
| Raw Exact-K CE best-on-val | 3 | 16.1966 | 0.0627 | 26.67 | `[16.1990, 16.2721, 16.1186]` |
| OPAL Exact-K CE best-on-val | 3 | 16.0387 | 0.1418 | 8.00 | `[15.9588, 16.2380, 15.9195]` |

Llama3.1 judgment:

- OPAL BCE best-on-val beats Raw BCE best-on-val by mean PPL, but loses on seed13.
- OPAL Exact-K CE best-on-val beats Raw Exact-K CE best-on-val on every seed.
- OPAL Exact-K CE best-on-val is the best mean-PPL skipping row.
- Final Raw BCE beats final OPAL BCE, so the claim must be validation-selected.
- The winning OPAL checkpoints are still low-diversity: Exact-K unique masks `[19, 2, 3]`.

## Beam2 Check

Beam2 was run only as a seed42 diagnostic on Qwen3-8B prefix1024 with `m1625`.

| method | seed | test PPL | unique_masks |
|---|---:|---:|---:|
| Full | 42 | 8.7518 | 1 |
| Raw final | 42 | 18.3150 | 155 |
| OPAL final | 42 | 19.3496 | 174 |
| Raw BCE best-on-val | 42 | 17.6664 | 15 |
| OPAL BCE best-on-val | 42 | 17.5360 | 1 |
| layerwise_hidden_router | 42 | 19.2931 | 159 |
| PuDDing-style | 42 | 23.5776 | 3 |
| IG-style | 42 | 23.2062 | 2 |

Beam2 did not improve OPAL over the greedy seed42 OPAL best result (`17.5360`). It is not part of the main claim.

## Speed Timing Breakdown

Corrected timing breakdown isolates `Full`, `OPAL BCE best-on-val`, and `OPAL Exact-K best-on-val`. It records actual router/mask overhead and physical skipped-forward time.

### Qwen3-8B Speed

| mode | method | PPL ref | latency/window ms | router ms/win | forward ms/win | router % | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| batch1_true_skip | Full | 8.7518 | 136.52 | 0.00 | 136.52 | 0.0% | 1.000x |
| batch1_true_skip | OPAL BCE best-on-val | 17.5360 | 117.02 | 12.58 | 104.44 | 10.8% | 1.167x |
| batch1_true_skip | OPAL Exact-K best-on-val | 17.5403 | 117.24 | 12.63 | 104.61 | 10.8% | 1.164x |
| grouped_by_mask | Full | 8.7518 | 126.61 | 0.00 | 126.61 | 0.0% | 1.000x |
| grouped_by_mask | OPAL BCE best-on-val | 17.5360 | 106.93 | 9.61 | 97.32 | 9.0% | 1.184x |
| grouped_by_mask | OPAL Exact-K best-on-val | 17.5403 | 107.17 | 9.64 | 97.53 | 9.0% | 1.181x |
| mixed_batch_naive | Full | 8.7518 | 126.99 | 0.00 | 126.99 | 0.0% | 1.000x |
| mixed_batch_naive | OPAL BCE best-on-val | 17.5360 | 106.97 | 9.96 | 97.02 | 9.3% | 1.187x |
| mixed_batch_naive | OPAL Exact-K best-on-val | 17.5403 | 108.02 | 9.98 | 98.03 | 9.2% | 1.176x |

### Llama3.1-8B-Instruct Speed

| mode | method | PPL ref | latency/window ms | router ms/win | forward ms/win | router % | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| batch1_true_skip | Full | 6.5076 | 125.42 | 0.00 | 125.42 | 0.0% | 1.000x |
| batch1_true_skip | OPAL BCE best-on-val | 16.0023 | 109.21 | 13.21 | 96.00 | 12.1% | 1.148x |
| batch1_true_skip | OPAL Exact-K best-on-val | 15.9588 | 109.23 | 13.18 | 96.05 | 12.1% | 1.148x |
| grouped_by_mask | Full | 6.5076 | 116.35 | 0.00 | 116.35 | 0.0% | 1.000x |
| grouped_by_mask | OPAL BCE best-on-val | 16.0023 | 100.23 | 9.96 | 90.27 | 9.9% | 1.161x |
| grouped_by_mask | OPAL Exact-K best-on-val | 15.9588 | 100.36 | 10.03 | 90.32 | 10.0% | 1.159x |
| mixed_batch_naive | Full | 6.5076 | 116.48 | 0.00 | 116.48 | 0.0% | 1.000x |
| mixed_batch_naive | OPAL BCE best-on-val | 16.0023 | 110.60 | 10.27 | 100.33 | 9.3% | 1.053x |
| mixed_batch_naive | OPAL Exact-K best-on-val | 15.9588 | 111.64 | 10.34 | 101.30 | 9.3% | 1.043x |

Speed judgment:

- OPAL has real wall-clock speedup after physical layer skipping.
- Router/mask inference costs about 9%-12% per window.
- Qwen3 OPAL reaches about `1.16x-1.18x`.
- Llama3.1 OPAL reaches about `1.15x-1.16x` in batch1/grouped settings and `1.04x-1.05x` under naive mixed batching.
- Static/Raw-style masks can be faster because they pay less routing overhead, but their PPL is worse in the prefix1024 main comparisons.

## Required Caveats

- The strongest OPAL wins use validation checkpoint selection.
- Final-epoch OPAL does not beat final-epoch Raw on Qwen3 or Llama3.1.
- Winning OPAL checkpoints are often low-diversity, especially Qwen3 OPAL BCE best with unique masks `[1, 11, 2]`.
- Beam2 is not part of the main result and did not improve OPAL in the seed42 diagnostic.
- OPAL should be described as a quality/speed tradeoff method, not the fastest possible skipping policy.

## Source Documents

- `docs/WIKITEXT2_QWEN3_PREF1024_3SEED_RESULTS.md`
- `docs/WIKITEXT2_LLAMA31_PREF1024_3SEED_RESULTS.md`
- `docs/WIKITEXT2_PUBLIC_LM_RELATED_RESULTS.md`
- `docs/WIKITEXT2_LAYER_SKIP_SPEED_TIMING_BREAKDOWN.md`
- `docs/WIKITEXT2_QWEN3_8B_CLEAN_K9_AUDIT.md`
