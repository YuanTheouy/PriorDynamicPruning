# WikiText-2 3-Seed Public LM Results

Last updated: 2026-06-02, generated from server JSON artifacts.

## Fixed Setup

- model: `/workspace/ckpts/Qwen2.5-1.5B`
- dataset: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- seeds: `[42, 13, 3407]`
- seq_len: `1024`
- router_prefix_tokens: `256`
- label_samples: `2000`
- eval_windows: `512`
- skip_rate: `0.25`
- skip_count: `7`
- protected_head / protected_tail: `4 / 2`

## Seed 42

| method | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique_masks | exact_skip_count_rate | selected/notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 | 1.000 | ok |
| Static ends_heavy | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ok |
| Static best-on-val | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | C6 selected ends_heavy by validation |
| PuDDing-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ends_heavy (289) |
| IG-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ends_heavy (289) |
| layerwise_hidden_router | 2.7568 | 15.7491 | 0.5414 | 6.5842 | 8 | 1.000 | ok |
| Raw-SetBCE final epoch | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 | 1.000 | ok |
| Raw-SetBCE best-on-val | 2.7530 | 15.6894 | 0.5376 | 6.5245 | 7 | 1.000 | epoch=24; val_PPL=16.3893; val_unique=4 |
| OPAL-SetBCE final epoch | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 | 1.000 | ok |
| OPAL-SetBCE best-on-val | 2.7486 | 15.6204 | 0.5332 | 6.4555 | 1 | 1.000 | epoch=2; val_PPL=16.3424; val_unique=1 |

- OPAL best-on-val wins Raw best-on-val: `True`
- OPAL best-on-val wins layerwise_hidden_router: `True`
- OPAL best-on-val unique_masks: `1`; best_epoch: `2`
- PuDDing-style selected candidates: `ends_heavy (289)`
- IG-style selected candidates: `ends_heavy (289)`

## Seed 13

| method | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique_masks | exact_skip_count_rate | selected/notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 | 1.000 | ok |
| Static ends_heavy | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ok |
| Static best-on-val | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | C6 selected ends_heavy by validation |
| PuDDing-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ends_heavy (289) |
| IG-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ends_heavy (289) |
| layerwise_hidden_router | 2.7546 | 15.7151 | 0.5392 | 6.5502 | 7 | 1.000 | ok |
| Raw-SetBCE final epoch | 2.7581 | 15.7696 | 0.5427 | 6.6047 | 13 | 1.000 | ok |
| Raw-SetBCE best-on-val | 2.7515 | 15.6661 | 0.5361 | 6.5012 | 6 | 1.000 | epoch=14; val_PPL=16.3894; val_unique=4 |
| OPAL-SetBCE final epoch | 2.7770 | 16.0711 | 0.5616 | 6.9062 | 19 | 1.000 | ok |
| OPAL-SetBCE best-on-val | 2.7486 | 15.6212 | 0.5332 | 6.4563 | 2 | 1.000 | epoch=6; val_PPL=16.3819; val_unique=2 |

- OPAL best-on-val wins Raw best-on-val: `True`
- OPAL best-on-val wins layerwise_hidden_router: `True`
- OPAL best-on-val unique_masks: `2`; best_epoch: `6`
- PuDDing-style selected candidates: `ends_heavy (289)`
- IG-style selected candidates: `ends_heavy (289)`

## Seed 3407

| method | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique_masks | exact_skip_count_rate | selected/notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 | 1.000 | ok |
| Static ends_heavy | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ok |
| Static best-on-val | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | C6 selected ends_heavy by validation |
| PuDDing-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ends_heavy (289) |
| IG-style | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 | 1.000 | ends_heavy (289) |
| layerwise_hidden_router | 2.7573 | 15.7568 | 0.5419 | 6.5919 | 8 | 1.000 | ok |
| Raw-SetBCE final epoch | 2.7574 | 15.7584 | 0.5420 | 6.5935 | 9 | 1.000 | ok |
| Raw-SetBCE best-on-val | 2.7549 | 15.7199 | 0.5395 | 6.5550 | 5 | 1.000 | epoch=25; val_PPL=16.3830; val_unique=5 |
| OPAL-SetBCE final epoch | 2.7701 | 15.9599 | 0.5547 | 6.7950 | 16 | 1.000 | ok |
| OPAL-SetBCE best-on-val | 2.7486 | 15.6204 | 0.5332 | 6.4555 | 1 | 1.000 | epoch=1; val_PPL=16.3424; val_unique=1 |

- OPAL best-on-val wins Raw best-on-val: `True`
- OPAL best-on-val wins layerwise_hidden_router: `True`
- OPAL best-on-val unique_masks: `1`; best_epoch: `1`
- PuDDing-style selected candidates: `ends_heavy (289)`
- IG-style selected candidates: `ends_heavy (289)`

## Three-Seed Mean/Std

| method | seeds present | NLL mean | NLL std | PPL mean | PPL std | Delta_NLL mean | Delta_NLL std | Delta_PPL mean | Delta_PPL std | unique_masks mean | unique_masks std | exact-K mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 3 | 2.2154 | 0.0000 | 9.1649 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.000 |
| Static ends_heavy | 3 | 2.8539 | 0.0000 | 17.3546 | 0.0000 | 0.6385 | 0.0000 | 8.1897 | 0.0000 | 1.0000 | 0.0000 | 1.000 |
| Static best-on-val | 3 | 2.8539 | 0.0000 | 17.3546 | 0.0000 | 0.6385 | 0.0000 | 8.1897 | 0.0000 | 1.0000 | 0.0000 | 1.000 |
| PuDDing-style | 3 | 2.8539 | 0.0000 | 17.3546 | 0.0000 | 0.6385 | 0.0000 | 8.1897 | 0.0000 | 1.0000 | 0.0000 | 1.000 |
| IG-style | 3 | 2.8539 | 0.0000 | 17.3546 | 0.0000 | 0.6385 | 0.0000 | 8.1897 | 0.0000 | 1.0000 | 0.0000 | 1.000 |
| layerwise_hidden_router | 3 | 2.7562 | 0.0014 | 15.7403 | 0.0222 | 0.5408 | 0.0014 | 6.5754 | 0.0222 | 7.6667 | 0.5774 | 1.000 |
| Raw-SetBCE final epoch | 3 | 2.7591 | 0.0024 | 15.7855 | 0.0375 | 0.5437 | 0.0024 | 6.6206 | 0.0375 | 10.0000 | 2.6458 | 1.000 |
| Raw-SetBCE best-on-val | 3 | 2.7531 | 0.0017 | 15.6918 | 0.0270 | 0.5378 | 0.0017 | 6.5269 | 0.0270 | 6.0000 | 1.0000 | 1.000 |
| OPAL-SetBCE final epoch | 3 | 2.7714 | 0.0051 | 15.9816 | 0.0808 | 0.5560 | 0.0051 | 6.8167 | 0.0808 | 16.6667 | 2.0817 | 1.000 |
| OPAL-SetBCE best-on-val | 3 | 2.7486 | 0.0000 | 15.6207 | 0.0005 | 0.5332 | 0.0000 | 6.4558 | 0.0005 | 1.3333 | 0.5774 | 1.000 |

## Required Judgments

- OPAL best-on-val wins Raw best-on-val by three-seed mean PPL: `True`
- OPAL best-on-val wins layerwise_hidden_router by three-seed mean PPL: `True`
- Per-seed OPAL vs Raw best-on-val wins: `[(42, True), (13, True), (3407, True)]`
- Per-seed OPAL vs layerwise wins: `[(42, True), (13, True), (3407, True)]`
- OPAL best-on-val unique_masks per available seed: `[1.0, 2.0, 1.0]`
- Caveat: OPAL best-on-val unique_masks is very low; describe this as a validation-selected static-like OPAL checkpoint, not a dynamic mask-diversity win.
- PuDDing-style selected candidate distributions: `[(42, 'ends_heavy (289)'), (13, 'ends_heavy (289)'), (3407, 'ends_heavy (289)')]`
- IG-style selected candidate distributions: `[(42, 'ends_heavy (289)'), (13, 'ends_heavy (289)'), (3407, 'ends_heavy (289)')]`

Conclusion: OPAL best-on-val supports a positive WikiText-2 public LM sanity for validation-selected PPL, with the explicit low-unique-mask caveat.

## Missing Artifacts

- none
