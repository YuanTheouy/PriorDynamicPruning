# WikiText-2 3-Seed Public LM Results

Last updated: 2026-06-02

## Status

This document is reserved for the fixed three-seed WikiText-2 public LM sanity experiment. It must be regenerated from server JSON artifacts after seeds `13` and `3407` finish:

```bash
cd /workspace/PriorDynamicPruning
source ~/venvs/planrec/bin/activate
python3 ./summarize_wikitext2_3seed_public_lm_results.py
```

Do not rerun completed seed42 experiments. Seed42 already has:

- Full
- Static ends_heavy
- Static best-on-val
- PuDDing-style
- IG-style
- layerwise_hidden_router
- Raw-SetBCE final epoch
- Raw-SetBCE best-on-val
- OPAL-SetBCE final epoch
- OPAL-SetBCE best-on-val

## Fixed Setup

- model: `/workspace/ckpts/Qwen2.5-1.5B`
- dataset: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- seeds: `42`, `13`, `3407`
- seq_len: `1024`
- router_prefix_tokens: `256`
- label_samples: `2000`
- eval_windows: `512`
- skip_rate: `0.25`
- skip_count: `7`
- protected_head / protected_tail: `4 / 2`

## Current Seed42 Anchor

| method | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique_masks | exact-K | note |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 | 1.000 | complete |
| Static ends_heavy | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | complete |
| Static best-on-val | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | C6 selected `ends_heavy` |
| PuDDing-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | selected `ends_heavy` for all 289 test windows |
| IG-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | selected `ends_heavy` for all 289 test windows |
| layerwise_hidden_router | 2.7568 | 15.7491 | 0.5414 | 6.5842 | 8 | 1.000 | complete |
| Raw-SetBCE final epoch | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 | 1.000 | complete |
| Raw-SetBCE best-on-val | 2.7530 | 15.6894 | 0.5376 | 6.5245 | 7 | 1.000 | epoch24 |
| OPAL-SetBCE final epoch | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 | 1.000 | complete |
| OPAL-SetBCE best-on-val | 2.7486 | 15.6204 | 0.5332 | 6.4555 | 1 | 1.000 | epoch2; static-like low-diversity checkpoint |

## Pending Seeds

| seed | status | required action |
|---:|---|---|
| 42 | complete | do not rerun |
| 13 | pending | run fixed command block from `WIKITEXT2_PUBLIC_LM_RELATED_RESULTS.md` or the chat command |
| 3407 | pending | run fixed command block from `WIKITEXT2_PUBLIC_LM_RELATED_RESULTS.md` or the chat command |

## Required Final Checks

After all artifacts exist, this document must contain:

- each seed's complete result table
- three-seed mean/std table
- whether OPAL best-on-val wins Raw best-on-val
- whether OPAL best-on-val wins `layerwise_hidden_router`
- whether PuDDing-style / IG-style still collapse to `ends_heavy`
- whether OPAL best-on-val has very low `unique_masks`

Decision rule:

- If OPAL best-on-val three-seed mean wins Raw best-on-val and `layerwise_hidden_router`, WikiText-2 can be used as a positive public LM sanity.
- If OPAL only wins seed42, WikiText-2 stays appendix-level.
- If OPAL wins but `unique_masks` is near 1, describe it as a validation-selected static-like OPAL checkpoint, not as a dynamic mask-diversity win.
