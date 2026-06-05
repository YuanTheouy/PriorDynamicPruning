# Qwen3-8B WikiText-2 Prefix1024 Rescue Results

Last updated: 2026-06-05

## Setup

| field | value |
|---|---|
| model | `/workspace/Models/Qwen3-8B` |
| dataset | `/workspace/datasets/wikitext/wikitext-2-raw-v1` |
| seed | `42` |
| seq_len / router_prefix_tokens | `1536` / `1024` |
| scored suffix tokens | `512` |
| requested label_samples | `2000` |
| actual clean label rows | `1625` |
| eval_windows requested | `512` |
| skip_rate / skip_count | `0.25` / `9` |
| protected_head / protected_tail | `4` / `2` |
| teacher | clean `final Delta_NLL` greedy set labels |
| label/run id | `wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_pref1024_seq1536_pref1024_m2000_seed42_skip0p25_K9` |

This is not a beam2 teacher run. It only changes the context setting from `seq1024/prefix256` to `seq1536/prefix1024`.

## Generated Server Reports

The server generated these report files on 2026-06-05:

| file | size | status |
|---|---:|---|
| `docs/WIKITEXT2_QWEN3_PREF1024_SANITY_RESULTS.md` | 2.5K | generated on server |
| `docs/WIKITEXT2_QWEN3_PREF1024_RELATED_RESULTS.md` | 36K | generated on server |
| `docs/WIKITEXT2_QWEN3_PREF1024_EXACTK_LOSS_RESULTS.md` | 1.8K before fix; updated after fix | generated on server |

## Final-Checkpoint PPL

The final checkpoint rows completed successfully.

| setting | method | loss | epoch | test NLL | test PPL | unique_masks | exact-K |
|---|---|---|---:|---:|---:|---:|---:|
| seq1024/prefix256 | Raw final | BCE | final | 3.0151 | 20.3915 | 224 | 1.000 |
| seq1024/prefix256 | OPAL final | BCE | final | 3.0341 | 20.7827 | 247 | 1.000 |
| seq1536/prefix1024 | Raw final | BCE | final | 2.9313 | 18.7511 | 177 | 1.000 |
| seq1536/prefix1024 | OPAL final | BCE | final | 2.9464 | 19.0376 | 180 | 1.000 |

Readout:

- Prefix1024 improves final Raw PPL by `1.6404` versus prefix256: `20.3915 -> 18.7511`.
- Prefix1024 improves final OPAL PPL by `1.7451` versus prefix256: `20.7827 -> 19.0376`.
- Prefix1024 therefore helps both dynamic routers, but final Raw still beats final OPAL by `0.2865` PPL.

## Validation-Selected Rows

The first prefix1024 valckpt attempt failed before training/evaluation because the label file had fewer rows than the requested sample count:

```text
Label file has 1625 rows, expected at least 2000
```

Reason:

```text
seq_len=1536 leaves fewer WikiText-2 train windows than seq_len=1024.
The label builder wrote the actual available 1625 rows.
The valckpt scripts were still checking WIKITEXT_LABEL_SAMPLES=2000.
```

The fix was to rerun only the valckpt jobs with:

```bash
export WIKITEXT_LABEL_SAMPLES=1625
```

The fixed validation-selected results are:

| method | loss | epoch | val NLL | val PPL | val unique | test NLL | test PPL | test unique | exact-K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw best-on-val | BCE | 3 | 2.9361 | 18.8423 | 11 | 2.8783 | 17.7837 | 13 | 1.000 |
| OPAL best-on-val | BCE | 1 | 2.9287 | 18.7030 | 1 | 2.8643 | 17.5360 | 1 | 1.000 |
| Raw best-on-val | Exact-K CE | 3 | 2.9388 | 18.8939 | 10 | 2.8806 | 17.8252 | 11 | 1.000 |
| OPAL best-on-val | Exact-K CE | 2 | 2.9293 | 18.7138 | 2 | 2.8645 | 17.5403 | 2 | 1.000 |

Source JSONs:

- Raw BCE best: `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_pref1024_seq1536_pref1024_m2000_seed42_skip0p25_K9_raw_valckpt/raw_best_val_epoch003_test.json`
- OPAL BCE best: `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_pref1024_seq1536_pref1024_m2000_seed42_skip0p25_K9_valckpt/opal_best_val_epoch001_test.json`
- Raw Exact-K CE best: `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_pref1024_seq1536_pref1024_m2000_seed42_skip0p25_K9_raw_exactk_valckpt/raw_best_val_epoch003_test.json`
- OPAL Exact-K CE best: `results/wikitext2_public_lm_sanity/val_ckpt_metrics/wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_pref1024_seq1536_pref1024_m2000_seed42_skip0p25_K9_opal_exactk_valckpt/opal_best_val_epoch002_test.json`

## Current Conclusion

Prefix1024 is a real rescue axis. It substantially improves both Raw and OPAL final PPL, and validation checkpoint selection makes OPAL beat Raw on seed42:

- OPAL BCE best beats Raw BCE best by `0.2477` PPL: `17.5360` vs `17.7837`.
- OPAL Exact-K CE best beats Raw Exact-K CE best by `0.2849` PPL: `17.5403` vs `17.8252`.
- OPAL BCE best is the best row among this prefix1024 seed42 table, slightly ahead of OPAL Exact-K CE.

Caveat: the selected OPAL masks are still low-diversity (`unique_masks=1` for BCE, `2` for Exact-K CE). This is a promising seed42 rescue, not yet a three-seed claim.
