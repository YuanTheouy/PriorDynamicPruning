# LLM Eval Downstream Results

Last updated: 2026-06-02

## Executive Summary

This is the no-compensation public downstream sanity benchmark over six `lm-evaluation-harness` multiple-choice tasks and three seeds (`42,13,3407`) using Qwen2.5-1.5B with 25% hard layer skipping (`skip_count=7`).

Key readout: OPAL-SetBCE best-on-val beats Raw-SetBCE best-on-val and `layerwise_hidden_router` on six-task mean `acc_norm`/fallback-`acc`, but it does not beat Static ends_heavy, Static best-on-val, IG-style, or PuDDing-style. This is therefore not an "OPAL dominates all downstream baselines" result.

Safe paper wording:

> On six public LM downstream multiple-choice tasks, validation-selected OPAL improves over raw-prefix and layerwise learned routers, but the fixed ends-heavy/static baseline remains stronger; OPAL's selected checkpoints are nearly static-like with very low mask diversity.

Do not write:

- OPAL is best on downstream public LM benchmarks.
- OPAL dominates static or IG-style downstream baselines.
- OPAL's downstream gains come from high dynamic mask diversity.

## Fixed Setup

- model: `/workspace/ckpts/Qwen2.5-1.5B`
- evaluator: `lm-evaluation-harness` Python API / CLI-compatible task definitions
- tasks: `piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge`
- seeds: `42,13,3407`
- max_length: `1024`
- router_prefix_tokens: `256`
- skip_rate / skip_count: `0.25 / 7`
- protected_head / protected_tail: `4 / 2`
- average kept layers: `21 / 28` for skipped methods
- compensation: `none`
- dataset cache note: server run used locally uploaded HF datasets cache for the six public tasks after HF Xet/CAS timeouts.

## Aggregate Readout

| comparison | six-task mean criterion | result |
|---|---:|---|
| OPAL best-on-val vs Raw-SetBCE best-on-val | `0.4923` vs `0.4896` acc_norm/fallback-acc | OPAL wins narrowly |
| OPAL best-on-val vs layerwise_hidden_router | `0.4923` vs `0.4855` | OPAL wins |
| OPAL best-on-val vs PuDDing-style | `0.4923` vs `0.5024` | OPAL loses |
| OPAL best-on-val vs IG-style | `0.4923` vs `0.5105` | OPAL loses |
| OPAL best-on-val vs Static best-on-val | `0.4923` vs `0.5105` | OPAL loses |
| OPAL best-on-val vs Static ends_heavy | `0.4923` vs `0.5105` | OPAL loses |

Mask-diversity caveat: OPAL best-on-val has very low `unique_masks` by seed (`[3.0, 2.5, 1.3333]`, mean `2.2778`), so it should be described as a validation-selected, static-like OPAL checkpoint rather than evidence of rich dynamic routing. IG-style collapses exactly to `ends_heavy` on all seeds. PuDDing-style does not strictly collapse to a single mask, but its candidate selection is overwhelmingly `ends_heavy`.

## Per-Seed Task Results

| seed | task | method | acc | acc_norm | retention_acc | retention_acc_norm | unique_masks | exact_skip_count_rate | average_kept_layers |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | piqa | Full | 0.7541 | 0.7590 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 42 | piqa | Static ends_heavy | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 42 | piqa | Static best-on-val | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 42 | piqa | PuDDing-style | 0.7144 | 0.7144 | 0.9473 | 0.9412 | 6 | 1.000 | 21.00 |
| 42 | piqa | IG-style | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 42 | piqa | layerwise_hidden_router | 0.7051 | 0.7013 | 0.9351 | 0.9240 | 69 | 1.000 | 21.00 |
| 42 | piqa | Raw-SetBCE best-on-val | 0.7116 | 0.7165 | 0.9437 | 0.9441 | 26 | 1.000 | 21.00 |
| 42 | piqa | OPAL-SetBCE best-on-val | 0.7122 | 0.7116 | 0.9444 | 0.9376 | 2 | 1.000 | 21.00 |
| 42 | openbookqa | Full | 0.3180 | 0.4080 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 42 | openbookqa | Static ends_heavy | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 42 | openbookqa | Static best-on-val | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 42 | openbookqa | PuDDing-style | 0.2760 | 0.3660 | 0.8679 | 0.8971 | 15 | 1.000 | 21.00 |
| 42 | openbookqa | IG-style | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 42 | openbookqa | layerwise_hidden_router | 0.2400 | 0.3440 | 0.7547 | 0.8431 | 110 | 1.000 | 21.00 |
| 42 | openbookqa | Raw-SetBCE best-on-val | 0.2540 | 0.3580 | 0.7987 | 0.8775 | 65 | 1.000 | 21.00 |
| 42 | openbookqa | OPAL-SetBCE best-on-val | 0.2660 | 0.3560 | 0.8365 | 0.8725 | 5 | 1.000 | 21.00 |
| 42 | winogrande | Full | 0.6385 | NA | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 42 | winogrande | Static ends_heavy | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 42 | winogrande | Static best-on-val | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 42 | winogrande | PuDDing-style | 0.5249 | NA | 0.8220 | 0.8220 | 16 | 1.000 | 21.00 |
| 42 | winogrande | IG-style | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 42 | winogrande | layerwise_hidden_router | 0.5272 | NA | 0.8257 | 0.8257 | 116 | 1.000 | 21.00 |
| 42 | winogrande | Raw-SetBCE best-on-val | 0.5138 | NA | 0.8047 | 0.8047 | 108 | 1.000 | 21.00 |
| 42 | winogrande | OPAL-SetBCE best-on-val | 0.5383 | NA | 0.8430 | 0.8430 | 5 | 1.000 | 21.00 |
| 42 | hellaswag | Full | 0.5029 | 0.6790 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 42 | hellaswag | Static ends_heavy | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 42 | hellaswag | Static best-on-val | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 42 | hellaswag | PuDDing-style | 0.3978 | 0.5272 | 0.7911 | 0.7764 | 8 | 1.000 | 21.00 |
| 42 | hellaswag | IG-style | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 42 | hellaswag | layerwise_hidden_router | 0.3863 | 0.5089 | 0.7681 | 0.7494 | 120 | 1.000 | 21.00 |
| 42 | hellaswag | Raw-SetBCE best-on-val | 0.3850 | 0.5019 | 0.7655 | 0.7391 | 32 | 1.000 | 21.00 |
| 42 | hellaswag | OPAL-SetBCE best-on-val | 0.3823 | 0.5012 | 0.7602 | 0.7381 | 2 | 1.000 | 21.00 |
| 42 | arc_easy | Full | 0.7538 | 0.7205 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 42 | arc_easy | Static ends_heavy | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 42 | arc_easy | Static best-on-val | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 42 | arc_easy | PuDDing-style | 0.6397 | 0.5728 | 0.8487 | 0.7950 | 5 | 1.000 | 21.00 |
| 42 | arc_easy | IG-style | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 42 | arc_easy | layerwise_hidden_router | 0.6313 | 0.5408 | 0.8375 | 0.7506 | 52 | 1.000 | 21.00 |
| 42 | arc_easy | Raw-SetBCE best-on-val | 0.6435 | 0.5455 | 0.8537 | 0.7570 | 26 | 1.000 | 21.00 |
| 42 | arc_easy | OPAL-SetBCE best-on-val | 0.6406 | 0.5412 | 0.8498 | 0.7512 | 2 | 1.000 | 21.00 |
| 42 | arc_challenge | Full | 0.4121 | 0.4522 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 42 | arc_challenge | Static ends_heavy | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 42 | arc_challenge | Static best-on-val | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 42 | arc_challenge | PuDDing-style | 0.2995 | 0.3259 | 0.7267 | 0.7208 | 5 | 1.000 | 21.00 |
| 42 | arc_challenge | IG-style | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 42 | arc_challenge | layerwise_hidden_router | 0.2961 | 0.3063 | 0.7184 | 0.6774 | 42 | 1.000 | 21.00 |
| 42 | arc_challenge | Raw-SetBCE best-on-val | 0.3012 | 0.3038 | 0.7308 | 0.6717 | 21 | 1.000 | 21.00 |
| 42 | arc_challenge | OPAL-SetBCE best-on-val | 0.2824 | 0.3020 | 0.6853 | 0.6679 | 2 | 1.000 | 21.00 |
| 13 | piqa | Full | 0.7541 | 0.7590 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 13 | piqa | Static ends_heavy | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 13 | piqa | Static best-on-val | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 13 | piqa | PuDDing-style | 0.6828 | 0.6850 | 0.9055 | 0.9025 | 11 | 1.000 | 21.00 |
| 13 | piqa | IG-style | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 13 | piqa | layerwise_hidden_router | 0.7029 | 0.7013 | 0.9322 | 0.9240 | 69 | 1.000 | 21.00 |
| 13 | piqa | Raw-SetBCE best-on-val | 0.7144 | 0.7203 | 0.9473 | 0.9491 | 14 | 1.000 | 21.00 |
| 13 | piqa | OPAL-SetBCE best-on-val | 0.7149 | 0.7214 | 0.9481 | 0.9505 | 2 | 1.000 | 21.00 |
| 13 | openbookqa | Full | 0.3180 | 0.4080 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 13 | openbookqa | Static ends_heavy | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 13 | openbookqa | Static best-on-val | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 13 | openbookqa | PuDDing-style | 0.2720 | 0.3760 | 0.8553 | 0.9216 | 15 | 1.000 | 21.00 |
| 13 | openbookqa | IG-style | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 13 | openbookqa | layerwise_hidden_router | 0.2540 | 0.3520 | 0.7987 | 0.8627 | 95 | 1.000 | 21.00 |
| 13 | openbookqa | Raw-SetBCE best-on-val | 0.2560 | 0.3480 | 0.8050 | 0.8529 | 57 | 1.000 | 21.00 |
| 13 | openbookqa | OPAL-SetBCE best-on-val | 0.2560 | 0.3600 | 0.8050 | 0.8824 | 5 | 1.000 | 21.00 |
| 13 | winogrande | Full | 0.6385 | NA | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 13 | winogrande | Static ends_heavy | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 13 | winogrande | Static best-on-val | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 13 | winogrande | PuDDing-style | 0.5099 | NA | 0.7985 | 0.7985 | 15 | 1.000 | 21.00 |
| 13 | winogrande | IG-style | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 13 | winogrande | layerwise_hidden_router | 0.4964 | NA | 0.7775 | 0.7775 | 93 | 1.000 | 21.00 |
| 13 | winogrande | Raw-SetBCE best-on-val | 0.5028 | NA | 0.7874 | 0.7874 | 77 | 1.000 | 21.00 |
| 13 | winogrande | OPAL-SetBCE best-on-val | 0.5162 | NA | 0.8084 | 0.8084 | 2 | 1.000 | 21.00 |
| 13 | hellaswag | Full | 0.5029 | 0.6790 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 13 | hellaswag | Static ends_heavy | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 13 | hellaswag | Static best-on-val | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 13 | hellaswag | PuDDing-style | 0.3958 | 0.5243 | 0.7871 | 0.7721 | 10 | 1.000 | 21.00 |
| 13 | hellaswag | IG-style | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 13 | hellaswag | layerwise_hidden_router | 0.3836 | 0.5034 | 0.7628 | 0.7413 | 93 | 1.000 | 21.00 |
| 13 | hellaswag | Raw-SetBCE best-on-val | 0.3845 | 0.5041 | 0.7646 | 0.7423 | 17 | 1.000 | 21.00 |
| 13 | hellaswag | OPAL-SetBCE best-on-val | 0.3859 | 0.5052 | 0.7673 | 0.7440 | 2 | 1.000 | 21.00 |
| 13 | arc_easy | Full | 0.7538 | 0.7205 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 13 | arc_easy | Static ends_heavy | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 13 | arc_easy | Static best-on-val | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 13 | arc_easy | PuDDing-style | 0.6263 | 0.5564 | 0.8308 | 0.7722 | 8 | 1.000 | 21.00 |
| 13 | arc_easy | IG-style | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 13 | arc_easy | layerwise_hidden_router | 0.6221 | 0.5429 | 0.8252 | 0.7535 | 57 | 1.000 | 21.00 |
| 13 | arc_easy | Raw-SetBCE best-on-val | 0.6406 | 0.5450 | 0.8498 | 0.7564 | 16 | 1.000 | 21.00 |
| 13 | arc_easy | OPAL-SetBCE best-on-val | 0.6389 | 0.5513 | 0.8476 | 0.7652 | 2 | 1.000 | 21.00 |
| 13 | arc_challenge | Full | 0.4121 | 0.4522 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 13 | arc_challenge | Static ends_heavy | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 13 | arc_challenge | Static best-on-val | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 13 | arc_challenge | PuDDing-style | 0.2952 | 0.3183 | 0.7164 | 0.7038 | 6 | 1.000 | 21.00 |
| 13 | arc_challenge | IG-style | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 13 | arc_challenge | layerwise_hidden_router | 0.2901 | 0.3072 | 0.7039 | 0.6792 | 43 | 1.000 | 21.00 |
| 13 | arc_challenge | Raw-SetBCE best-on-val | 0.2892 | 0.3072 | 0.7019 | 0.6792 | 13 | 1.000 | 21.00 |
| 13 | arc_challenge | OPAL-SetBCE best-on-val | 0.2927 | 0.3123 | 0.7101 | 0.6906 | 2 | 1.000 | 21.00 |
| 3407 | piqa | Full | 0.7541 | 0.7590 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 3407 | piqa | Static ends_heavy | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 3407 | piqa | Static best-on-val | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 3407 | piqa | PuDDing-style | 0.7106 | 0.7095 | 0.9423 | 0.9348 | 9 | 1.000 | 21.00 |
| 3407 | piqa | IG-style | 0.7176 | 0.7220 | 0.9517 | 0.9513 | 1 | 1.000 | 21.00 |
| 3407 | piqa | layerwise_hidden_router | 0.6931 | 0.7067 | 0.9192 | 0.9312 | 98 | 1.000 | 21.00 |
| 3407 | piqa | Raw-SetBCE best-on-val | 0.7046 | 0.7155 | 0.9343 | 0.9427 | 25 | 1.000 | 21.00 |
| 3407 | piqa | OPAL-SetBCE best-on-val | 0.7111 | 0.7133 | 0.9430 | 0.9398 | 1 | 1.000 | 21.00 |
| 3407 | openbookqa | Full | 0.3180 | 0.4080 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 3407 | openbookqa | Static ends_heavy | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 3407 | openbookqa | Static best-on-val | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 3407 | openbookqa | PuDDing-style | 0.2800 | 0.3760 | 0.8805 | 0.9216 | 16 | 1.000 | 21.00 |
| 3407 | openbookqa | IG-style | 0.2900 | 0.3820 | 0.9119 | 0.9363 | 1 | 1.000 | 21.00 |
| 3407 | openbookqa | layerwise_hidden_router | 0.2420 | 0.3440 | 0.7610 | 0.8431 | 146 | 1.000 | 21.00 |
| 3407 | openbookqa | Raw-SetBCE best-on-val | 0.2680 | 0.3660 | 0.8428 | 0.8971 | 73 | 1.000 | 21.00 |
| 3407 | openbookqa | OPAL-SetBCE best-on-val | 0.2480 | 0.3620 | 0.7799 | 0.8873 | 2 | 1.000 | 21.00 |
| 3407 | winogrande | Full | 0.6385 | NA | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 3407 | winogrande | Static ends_heavy | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 3407 | winogrande | Static best-on-val | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 3407 | winogrande | PuDDing-style | 0.5320 | NA | 0.8331 | 0.8331 | 15 | 1.000 | 21.00 |
| 3407 | winogrande | IG-style | 0.5320 | NA | 0.8331 | 0.8331 | 1 | 1.000 | 21.00 |
| 3407 | winogrande | layerwise_hidden_router | 0.5241 | NA | 0.8208 | 0.8208 | 196 | 1.000 | 21.00 |
| 3407 | winogrande | Raw-SetBCE best-on-val | 0.5185 | NA | 0.8121 | 0.8121 | 97 | 1.000 | 21.00 |
| 3407 | winogrande | OPAL-SetBCE best-on-val | 0.5272 | NA | 0.8257 | 0.8257 | 2 | 1.000 | 21.00 |
| 3407 | hellaswag | Full | 0.5029 | 0.6790 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 3407 | hellaswag | Static ends_heavy | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 3407 | hellaswag | Static best-on-val | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 3407 | hellaswag | PuDDing-style | 0.3974 | 0.5266 | 0.7903 | 0.7755 | 10 | 1.000 | 21.00 |
| 3407 | hellaswag | IG-style | 0.3979 | 0.5274 | 0.7913 | 0.7767 | 1 | 1.000 | 21.00 |
| 3407 | hellaswag | layerwise_hidden_router | 0.3837 | 0.5035 | 0.7630 | 0.7415 | 117 | 1.000 | 21.00 |
| 3407 | hellaswag | Raw-SetBCE best-on-val | 0.3845 | 0.5036 | 0.7646 | 0.7416 | 39 | 1.000 | 21.00 |
| 3407 | hellaswag | OPAL-SetBCE best-on-val | 0.3822 | 0.4989 | 0.7600 | 0.7347 | 1 | 1.000 | 21.00 |
| 3407 | arc_easy | Full | 0.7538 | 0.7205 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 3407 | arc_easy | Static ends_heavy | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 3407 | arc_easy | Static best-on-val | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 3407 | arc_easy | PuDDing-style | 0.6372 | 0.5711 | 0.8453 | 0.7926 | 8 | 1.000 | 21.00 |
| 3407 | arc_easy | IG-style | 0.6406 | 0.5745 | 0.8498 | 0.7973 | 1 | 1.000 | 21.00 |
| 3407 | arc_easy | layerwise_hidden_router | 0.6225 | 0.5253 | 0.8258 | 0.7290 | 67 | 1.000 | 21.00 |
| 3407 | arc_easy | Raw-SetBCE best-on-val | 0.6376 | 0.5375 | 0.8459 | 0.7459 | 29 | 1.000 | 21.00 |
| 3407 | arc_easy | OPAL-SetBCE best-on-val | 0.6376 | 0.5379 | 0.8459 | 0.7465 | 1 | 1.000 | 21.00 |
| 3407 | arc_challenge | Full | 0.4121 | 0.4522 | 1.0000 | 1.0000 | 1 | 1.000 | 28.00 |
| 3407 | arc_challenge | Static ends_heavy | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 3407 | arc_challenge | Static best-on-val | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 3407 | arc_challenge | PuDDing-style | 0.3003 | 0.3268 | 0.7288 | 0.7226 | 7 | 1.000 | 21.00 |
| 3407 | arc_challenge | IG-style | 0.3003 | 0.3251 | 0.7288 | 0.7189 | 1 | 1.000 | 21.00 |
| 3407 | arc_challenge | layerwise_hidden_router | 0.2892 | 0.3038 | 0.7019 | 0.6717 | 52 | 1.000 | 21.00 |
| 3407 | arc_challenge | Raw-SetBCE best-on-val | 0.2995 | 0.3046 | 0.7267 | 0.6736 | 23 | 1.000 | 21.00 |
| 3407 | arc_challenge | OPAL-SetBCE best-on-val | 0.2790 | 0.3046 | 0.6770 | 0.6736 | 1 | 1.000 | 21.00 |

## Three-Seed Mean/Std

| task | method | acc mean | acc std | acc_norm mean | acc_norm std | retention_acc mean | retention_acc_norm mean | unique_masks mean | exact_skip_count_rate mean | average_kept_layers mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| piqa | Full | 0.7541 | 0.0000 | 0.7590 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 28.00 |
| piqa | Static ends_heavy | 0.7176 | 0.0000 | 0.7220 | 0.0000 | 0.9517 | 0.9513 | 1.0000 | 1.000 | 21.00 |
| piqa | Static best-on-val | 0.7176 | 0.0000 | 0.7220 | 0.0000 | 0.9517 | 0.9513 | 1.0000 | 1.000 | 21.00 |
| piqa | PuDDing-style | 0.7026 | 0.0172 | 0.7029 | 0.0157 | 0.9317 | 0.9262 | 8.6667 | 1.000 | 21.00 |
| piqa | IG-style | 0.7176 | 0.0000 | 0.7220 | 0.0000 | 0.9517 | 0.9513 | 1.0000 | 1.000 | 21.00 |
| piqa | layerwise_hidden_router | 0.7004 | 0.0064 | 0.7031 | 0.0031 | 0.9288 | 0.9264 | 78.6667 | 1.000 | 21.00 |
| piqa | Raw-SetBCE best-on-val | 0.7102 | 0.0051 | 0.7174 | 0.0026 | 0.9418 | 0.9453 | 21.6667 | 1.000 | 21.00 |
| piqa | OPAL-SetBCE best-on-val | 0.7127 | 0.0020 | 0.7155 | 0.0052 | 0.9452 | 0.9427 | 1.6667 | 1.000 | 21.00 |
| openbookqa | Full | 0.3180 | 0.0000 | 0.4080 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 28.00 |
| openbookqa | Static ends_heavy | 0.2900 | 0.0000 | 0.3820 | 0.0000 | 0.9119 | 0.9363 | 1.0000 | 1.000 | 21.00 |
| openbookqa | Static best-on-val | 0.2900 | 0.0000 | 0.3820 | 0.0000 | 0.9119 | 0.9363 | 1.0000 | 1.000 | 21.00 |
| openbookqa | PuDDing-style | 0.2760 | 0.0040 | 0.3727 | 0.0058 | 0.8679 | 0.9134 | 15.3333 | 1.000 | 21.00 |
| openbookqa | IG-style | 0.2900 | 0.0000 | 0.3820 | 0.0000 | 0.9119 | 0.9363 | 1.0000 | 1.000 | 21.00 |
| openbookqa | layerwise_hidden_router | 0.2453 | 0.0076 | 0.3467 | 0.0046 | 0.7715 | 0.8497 | 117.0000 | 1.000 | 21.00 |
| openbookqa | Raw-SetBCE best-on-val | 0.2593 | 0.0076 | 0.3573 | 0.0090 | 0.8155 | 0.8758 | 65.0000 | 1.000 | 21.00 |
| openbookqa | OPAL-SetBCE best-on-val | 0.2567 | 0.0090 | 0.3593 | 0.0031 | 0.8071 | 0.8807 | 4.0000 | 1.000 | 21.00 |
| winogrande | Full | 0.6385 | 0.0000 | 0.6385 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 28.00 |
| winogrande | Static ends_heavy | 0.5320 | 0.0000 | 0.5320 | 0.0000 | 0.8331 | 0.8331 | 1.0000 | 1.000 | 21.00 |
| winogrande | Static best-on-val | 0.5320 | 0.0000 | 0.5320 | 0.0000 | 0.8331 | 0.8331 | 1.0000 | 1.000 | 21.00 |
| winogrande | PuDDing-style | 0.5222 | 0.0113 | 0.5222 | 0.0113 | 0.8179 | 0.8179 | 15.3333 | 1.000 | 21.00 |
| winogrande | IG-style | 0.5320 | 0.0000 | 0.5320 | 0.0000 | 0.8331 | 0.8331 | 1.0000 | 1.000 | 21.00 |
| winogrande | layerwise_hidden_router | 0.5159 | 0.0169 | 0.5159 | 0.0169 | 0.8080 | 0.8080 | 135.0000 | 1.000 | 21.00 |
| winogrande | Raw-SetBCE best-on-val | 0.5117 | 0.0081 | 0.5117 | 0.0081 | 0.8014 | 0.8014 | 94.0000 | 1.000 | 21.00 |
| winogrande | OPAL-SetBCE best-on-val | 0.5272 | 0.0110 | 0.5272 | 0.0110 | 0.8257 | 0.8257 | 3.0000 | 1.000 | 21.00 |
| hellaswag | Full | 0.5029 | 0.0000 | 0.6790 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 28.00 |
| hellaswag | Static ends_heavy | 0.3979 | 0.0000 | 0.5274 | 0.0000 | 0.7913 | 0.7767 | 1.0000 | 1.000 | 21.00 |
| hellaswag | Static best-on-val | 0.3979 | 0.0000 | 0.5274 | 0.0000 | 0.7913 | 0.7767 | 1.0000 | 1.000 | 21.00 |
| hellaswag | PuDDing-style | 0.3970 | 0.0011 | 0.5260 | 0.0015 | 0.7895 | 0.7746 | 9.3333 | 1.000 | 21.00 |
| hellaswag | IG-style | 0.3979 | 0.0000 | 0.5274 | 0.0000 | 0.7913 | 0.7767 | 1.0000 | 1.000 | 21.00 |
| hellaswag | layerwise_hidden_router | 0.3845 | 0.0015 | 0.5052 | 0.0031 | 0.7646 | 0.7440 | 110.0000 | 1.000 | 21.00 |
| hellaswag | Raw-SetBCE best-on-val | 0.3847 | 0.0003 | 0.5032 | 0.0011 | 0.7649 | 0.7410 | 29.3333 | 1.000 | 21.00 |
| hellaswag | OPAL-SetBCE best-on-val | 0.3835 | 0.0021 | 0.5018 | 0.0032 | 0.7625 | 0.7389 | 1.6667 | 1.000 | 21.00 |
| arc_easy | Full | 0.7538 | 0.0000 | 0.7205 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 28.00 |
| arc_easy | Static ends_heavy | 0.6406 | 0.0000 | 0.5745 | 0.0000 | 0.8498 | 0.7973 | 1.0000 | 1.000 | 21.00 |
| arc_easy | Static best-on-val | 0.6406 | 0.0000 | 0.5745 | 0.0000 | 0.8498 | 0.7973 | 1.0000 | 1.000 | 21.00 |
| arc_easy | PuDDing-style | 0.6344 | 0.0072 | 0.5668 | 0.0090 | 0.8416 | 0.7866 | 7.0000 | 1.000 | 21.00 |
| arc_easy | IG-style | 0.6406 | 0.0000 | 0.5745 | 0.0000 | 0.8498 | 0.7973 | 1.0000 | 1.000 | 21.00 |
| arc_easy | layerwise_hidden_router | 0.6253 | 0.0052 | 0.5363 | 0.0097 | 0.8295 | 0.7444 | 58.6667 | 1.000 | 21.00 |
| arc_easy | Raw-SetBCE best-on-val | 0.6406 | 0.0029 | 0.5426 | 0.0045 | 0.8498 | 0.7531 | 23.6667 | 1.000 | 21.00 |
| arc_easy | OPAL-SetBCE best-on-val | 0.6390 | 0.0015 | 0.5435 | 0.0070 | 0.8478 | 0.7543 | 1.6667 | 1.000 | 21.00 |
| arc_challenge | Full | 0.4121 | 0.0000 | 0.4522 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 28.00 |
| arc_challenge | Static ends_heavy | 0.3003 | 0.0000 | 0.3251 | 0.0000 | 0.7288 | 0.7189 | 1.0000 | 1.000 | 21.00 |
| arc_challenge | Static best-on-val | 0.3003 | 0.0000 | 0.3251 | 0.0000 | 0.7288 | 0.7189 | 1.0000 | 1.000 | 21.00 |
| arc_challenge | PuDDing-style | 0.2984 | 0.0027 | 0.3237 | 0.0047 | 0.7239 | 0.7157 | 6.0000 | 1.000 | 21.00 |
| arc_challenge | IG-style | 0.3003 | 0.0000 | 0.3251 | 0.0000 | 0.7288 | 0.7189 | 1.0000 | 1.000 | 21.00 |
| arc_challenge | layerwise_hidden_router | 0.2918 | 0.0037 | 0.3057 | 0.0018 | 0.7081 | 0.6761 | 45.6667 | 1.000 | 21.00 |
| arc_challenge | Raw-SetBCE best-on-val | 0.2966 | 0.0065 | 0.3052 | 0.0018 | 0.7198 | 0.6748 | 19.0000 | 1.000 | 21.00 |
| arc_challenge | OPAL-SetBCE best-on-val | 0.2847 | 0.0071 | 0.3063 | 0.0053 | 0.6908 | 0.6774 | 1.6667 | 1.000 | 21.00 |

## Six-Task Average

| method | acc mean | acc std over seeds | acc_norm mean | acc_norm std over seeds | retention_acc mean | retention_acc_norm mean | unique_masks mean | exact_skip_count_rate mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.5632 | 0.0000 | 0.6095 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| Static ends_heavy | 0.4797 | 0.0000 | 0.5105 | 0.0000 | 0.8444 | 0.8356 | 1.0000 | 1.000 |
| Static best-on-val | 0.4797 | 0.0000 | 0.5105 | 0.0000 | 0.8444 | 0.8356 | 1.0000 | 1.000 |
| PuDDing-style | 0.4718 | 0.0070 | 0.5024 | 0.0065 | 0.8288 | 0.8224 | 10.2778 | 1.000 |
| IG-style | 0.4797 | 0.0000 | 0.5105 | 0.0000 | 0.8444 | 0.8356 | 1.0000 | 1.000 |
| layerwise_hidden_router | 0.4605 | 0.0033 | 0.4855 | 0.0023 | 0.8018 | 0.7914 | 90.8333 | 1.000 |
| Raw-SetBCE best-on-val | 0.4672 | 0.0023 | 0.4896 | 0.0015 | 0.8155 | 0.7986 | 42.1111 | 1.000 |
| OPAL-SetBCE best-on-val | 0.4673 | 0.0031 | 0.4923 | 0.0019 | 0.8132 | 0.8033 | 2.2778 | 1.000 |

## Required Judgments

- OPAL best-on-val beats `Raw-SetBCE best-on-val` by six-task mean acc_norm-or-acc: `True`
- OPAL best-on-val beats `layerwise_hidden_router` by six-task mean acc_norm-or-acc: `True`
- OPAL best-on-val beats `PuDDing-style` by six-task mean acc_norm-or-acc: `False`
- OPAL best-on-val beats `IG-style` by six-task mean acc_norm-or-acc: `False`
- OPAL best-on-val beats `Static best-on-val` by six-task mean acc_norm-or-acc: `False`
- OPAL best-on-val beats `Static ends_heavy` by six-task mean acc_norm-or-acc: `False`
- OPAL best-on-val unique_masks mean per seed: `[3.0, 2.5, 1.3333]`
- PuDDing-style candidate distributions by seed: `[(42, {'ends_heavy': 17305, 'random_diverse_v1': 47, 'random_diverse_v7': 185, 'last_k': 64, 'random_diverse_v5': 53, 'random_diverse_v6': 23, 'random_diverse_v3': 42, 'random_diverse_v9': 5, 'random_diverse_v10': 36, 'random_diverse_v4': 47, 'first_k': 47, 'uniform': 45, 'middle_heavy': 3, 'random_diverse_v8': 12, 'random_diverse_v2': 4, 'random_diverse_seed42': 7}), (13, {'ends_heavy': 15745, 'random_diverse_v1': 724, 'random_diverse_seed42': 590, 'last_k': 507, 'random_diverse_v4': 75, 'random_diverse_v8': 94, 'random_diverse_v9': 35, 'random_diverse_v3': 34, 'random_diverse_v6': 21, 'random_diverse_v2': 7, 'uniform': 24, 'random_diverse_v7': 11, 'middle_heavy': 23, 'first_k': 30, 'random_diverse_v5': 5}), (3407, {'ends_heavy': 17286, 'last_k': 183, 'random_diverse_v8': 66, 'random_diverse_v9': 60, 'random_diverse_v3': 41, 'random_diverse_v4': 12, 'random_diverse_v6': 28, 'random_diverse_v5': 56, 'random_diverse_v7': 17, 'random_diverse_seed42': 48, 'random_diverse_v2': 23, 'uniform': 25, 'random_diverse_v10': 27, 'first_k': 14, 'middle_heavy': 33, 'random_diverse_v1': 6})]`
- PuDDing-style degenerates to ends_heavy only: `False`
- IG-style candidate distributions by seed: `[(42, {'ends_heavy': 17925}), (13, {'ends_heavy': 17925}), (3407, {'ends_heavy': 17925})]`
- IG-style degenerates to ends_heavy only: `True`
- Caveat: OPAL best-on-val unique_masks is very low on at least one seed; keep the static-like checkpoint caveat.
