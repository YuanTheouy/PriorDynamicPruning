# Llama3.1-8B-Instruct WikiText-2 Prefix1024 Three-Seed Results

Last updated: 2026-06-09

## Setup

| field | value |
|---|---|
| model | `/workspace/Models/Llama-3.1-8B-Instruct` |
| dataset | `/workspace/datasets/wikitext/wikitext-2-raw-v1` |
| seeds | `42`, `13`, `3407` |
| seq_len / router_prefix_tokens | `1536` / `1024` |
| scored suffix tokens | `512` |
| actual label rows | `1572` |
| eval_windows requested | `512` |
| skip_rate / skip_count | `0.25` / `8` |
| protected_head / protected_tail | `4` / `2` |
| teacher | clean `final Delta_NLL` forward-greedy set labels |
| search | greedy, not beam2 |

This run checks whether the Qwen3-8B prefix1024 result transfers to a second public 8B LM architecture. It uses Llama3.1-8B-Instruct with 32 layers, so 25% skipping gives `K=8`.

## Per-Seed Results

| seed | method | loss | epoch | test NLL | test PPL | test unique | exact-K |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | Full | none | full | 1.8730 | 6.5076 | 1 | 1.000 |
| 42 | Static uniform | none | static | 3.7046 | 40.6336 | 1 | 1.000 |
| 42 | Static ends_heavy | none | static | 4.5730 | 96.8353 | 1 | 1.000 |
| 42 | Static best-on-val C6 | none | static | 3.6966 | 40.3085 | 1 | 1.000 |
| 42 | PuDDing-style | related | final | 3.0985 | 22.1651 | 2 | 1.000 |
| 42 | IG-style | related | final | 3.0962 | 22.1129 | 1 | 1.000 |
| 42 | layerwise_hidden_router | related | final | 2.8801 | 17.8157 | 143 | 1.000 |
| 42 | Raw final | BCE | final | 2.8028 | 16.4910 | 88 | 1.000 |
| 42 | OPAL final | BCE | final | 2.8528 | 17.3362 | 118 | 1.000 |
| 42 | Raw best-on-val | BCE | 6 | 2.7837 | 16.1786 | 16 | 1.000 |
| 42 | OPAL best-on-val | BCE | 17 | 2.7727 | 16.0023 | 18 | 1.000 |
| 42 | Raw best-on-val | Exact-K CE | 5 | 2.7850 | 16.1990 | 15 | 1.000 |
| 42 | OPAL best-on-val | Exact-K CE | 10 | 2.7700 | 15.9588 | 19 | 1.000 |
| 13 | Full | none | full | 1.8730 | 6.5076 | 1 | 1.000 |
| 13 | Static uniform | none | static | 3.7046 | 40.6336 | 1 | 1.000 |
| 13 | Static ends_heavy | none | static | 4.5730 | 96.8353 | 1 | 1.000 |
| 13 | Static best-on-val C6 | none | static | 3.7046 | 40.6336 | 1 | 1.000 |
| 13 | PuDDing-style | related | final | 3.3265 | 27.8404 | 4 | 1.000 |
| 13 | IG-style | related | final | 3.3239 | 27.7673 | 2 | 1.000 |
| 13 | layerwise_hidden_router | related | final | 2.8398 | 17.1124 | 124 | 1.000 |
| 13 | Raw final | BCE | final | 2.8174 | 16.7334 | 77 | 1.000 |
| 13 | OPAL final | BCE | final | 2.8248 | 16.8571 | 85 | 1.000 |
| 13 | Raw best-on-val | BCE | 4 | 2.7863 | 16.2202 | 16 | 1.000 |
| 13 | OPAL best-on-val | BCE | 17 | 2.7893 | 16.2698 | 12 | 1.000 |
| 13 | Raw best-on-val | Exact-K CE | 9 | 2.7894 | 16.2721 | 24 | 1.000 |
| 13 | OPAL best-on-val | Exact-K CE | 3 | 2.7874 | 16.2380 | 2 | 1.000 |
| 3407 | Full | none | full | 1.8730 | 6.5076 | 1 | 1.000 |
| 3407 | Static uniform | none | static | 3.7046 | 40.6336 | 1 | 1.000 |
| 3407 | Static ends_heavy | none | static | 4.5730 | 96.8353 | 1 | 1.000 |
| 3407 | Static best-on-val C6 | none | static | 3.4910 | 32.8183 | 1 | 1.000 |
| 3407 | PuDDing-style | related | final | 3.1562 | 23.4812 | 7 | 1.000 |
| 3407 | IG-style | related | final | 3.1369 | 23.0332 | 2 | 1.000 |
| 3407 | layerwise_hidden_router | related | final | 2.8490 | 17.2701 | 144 | 1.000 |
| 3407 | Raw final | BCE | final | 2.8005 | 16.4531 | 74 | 1.000 |
| 3407 | OPAL final | BCE | final | 2.8425 | 17.1578 | 121 | 1.000 |
| 3407 | Raw best-on-val | BCE | 6 | 2.7873 | 16.2373 | 24 | 1.000 |
| 3407 | OPAL best-on-val | BCE | 5 | 2.7734 | 16.0123 | 3 | 1.000 |
| 3407 | Raw best-on-val | Exact-K CE | 13 | 2.7800 | 16.1186 | 41 | 1.000 |
| 3407 | OPAL best-on-val | Exact-K CE | 3 | 2.7675 | 15.9195 | 3 | 1.000 |

## Mean/Std Summary

| method | seeds | test PPL mean | test PPL std | unique_masks mean | PPL values |
|---|---:|---:|---:|---:|---|
| Full | 3 | 6.5076 | 0.0000 | 1.00 | `[6.5076, 6.5076, 6.5076]` |
| Static best-on-val C6 | 3 | 37.9201 | 3.6100 | 1.00 | `[40.3085, 40.6336, 32.8183]` |
| PuDDing-style | 3 | 24.4955 | 2.4254 | 4.33 | `[22.1651, 27.8404, 23.4812]` |
| IG-style | 3 | 24.3045 | 2.4773 | 1.67 | `[22.1129, 27.7673, 23.0332]` |
| layerwise_hidden_router | 3 | 17.3994 | 0.3013 | 137.00 | `[17.8157, 17.1124, 17.2701]` |
| Raw BCE final | 3 | 16.5592 | 0.1242 | 79.67 | `[16.4910, 16.7334, 16.4531]` |
| OPAL BCE final | 3 | 17.1170 | 0.1977 | 108.00 | `[17.3362, 16.8571, 17.1578]` |
| Raw BCE best | 3 | 16.2121 | 0.0247 | 18.67 | `[16.1786, 16.2202, 16.2373]` |
| OPAL BCE best | 3 | 16.0948 | 0.1238 | 11.00 | `[16.0023, 16.2698, 16.0123]` |
| Raw Exact-K CE best | 3 | 16.1966 | 0.0627 | 26.67 | `[16.1990, 16.2721, 16.1186]` |
| OPAL Exact-K CE best | 3 | 16.0387 | 0.1418 | 8.00 | `[15.9588, 16.2380, 15.9195]` |

## Required Judgments

- OPAL BCE best-on-val beats Raw BCE best-on-val by mean PPL: `16.0948` vs `16.2121`, gain `0.1173`.
- OPAL BCE best-on-val beats Raw BCE best-on-val on seeds 42 and 3407, but loses on seed13: `[(42, True), (13, False), (3407, True)]`.
- OPAL Exact-K CE best-on-val beats Raw Exact-K CE best-on-val on every seed: `True`.
- OPAL Exact-K CE best-on-val beats Raw Exact-K CE best-on-val by mean PPL: `16.0387` vs `16.1966`, gain `0.1579`.
- OPAL Exact-K CE best-on-val is the best mean-PPL skipping row in this Llama3.1 run.
- OPAL Exact-K CE best-on-val beats layerwise_hidden_router by mean PPL: `16.0387` vs `17.3994`, gain `1.3607`.
- OPAL Exact-K CE best-on-val beats PuDDing-style and IG-style by a large margin.
- Final checkpoints do not show the OPAL-over-Raw win: Raw BCE final mean `16.5592` beats OPAL BCE final mean `17.1170`.
- OPAL Exact-K CE best-on-val remains low-diversity: unique masks `[19, 2, 3]`, mean `8.00`.
- Speed has not yet been recorded in this document; `docs/WIKITEXT2_LAYER_SKIP_SPEED_RESULTS.md` is reserved for the latency/throughput table.

## Conclusion

The Llama3.1-8B-Instruct prefix1024 result supports the same paper story as Qwen3-8B, but with a stronger Exact-K CE ablation:

> On Llama3.1-8B-Instruct WikiText-2 with 25% layer skipping (K=8), validation-selected OPAL outperforms Raw, PuDDing-style, IG-style, and layerwise hidden routing across three seeds. The best mean PPL is OPAL Exact-K CE best-on-val (`16.0387`), ahead of Raw Exact-K CE best-on-val (`16.1966`), Raw BCE best-on-val (`16.2121`), and layerwise_hidden_router (`17.3994`). The caveat is that the winning checkpoints are validation-selected and low-diversity, while final-epoch OPAL does not beat final-epoch Raw.

