# WikiText-2 Public LM Sanity Results

Date: 2026-06-02

## Status

Pending server execution. I did not run GPU experiments locally because the required GPU, model checkpoints, and WikiText-2 cache are only available in `/workspace/PriorDynamicPruning` on the 8xA100 server.

The server runner will overwrite/update this file after smoke or main finishes:

```bash
bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
```

## Planned Setup

- dataset: `wikitext/wikitext-2-raw-v1`
- preferred local dataset path: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- train split: greedy labels and router training
- validation split: static best-on-val C6 selection
- test split: final PPL
- seq_len: 1024 for main, 512 for smoke
- label_samples: 2000 for main, 32 for smoke
- eval_windows: 512 for main, 32 for smoke
- seed: 42
- skip_rate: 0.25
- protected_head: 4
- protected_tail: 2
- router context: first `router_prefix_tokens` tokens only
- scored tokens: suffix tokens only, to avoid target leakage

## Result Table Template

| method | K skipped | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0 | TBD | TBD | 0 | 0 | 1 |
| Static uniform | K | TBD | TBD | TBD | TBD | 1 |
| Static ends_heavy | K | TBD | TBD | TBD | TBD | 1 |
| Static best-on-val C6 | K | TBD | TBD | TBD | TBD | 1 |
| Raw-SetBCE | K | TBD | TBD | TBD | TBD | TBD |
| OPAL-SetBCE | K | TBD | TBD | TBD | TBD | TBD |

## Required Fields To Fill After Server Run

- model path / model name / num_layers
- dataset split / seq_len / eval_windows / eval_tokens
- skip_rate / K / protected layers / allowed layers
- full PPL
- all baseline PPL / Delta_NLL / Delta_PPL
- OPAL-SetBCE training loss first / best / last
- OPAL-SetBCE overlap@K / hamming / unique masks
- target/test leakage check

## Leakage Check

Expected final report should state:

- Greedy labels are built only on train.
- Static C6 best mask is selected only on validation.
- Test PPL evaluation does not load greedy labels.
- Router uses only the input prefix, not target suffix tokens.
