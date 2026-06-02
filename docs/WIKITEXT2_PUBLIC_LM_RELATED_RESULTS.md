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

## Original WikiText Bad Result

| method | type | NLL ↓ | PPL ↓ | Delta_NLL ↓ | Delta_PPL ↓ | unique masks |
|---|---|---:|---:|---:|---:|---:|
| Full | full | 2.2154 | 9.1649 | 0.0000 | 0.0000 | 1 |
| Static uniform | static | 4.0673 | 58.3987 | 1.8519 | 49.2338 | 1 |
| Static ends_heavy | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 |
| Static best-on-val C6 | static | 2.8539 | 17.3546 | 0.6385 | 8.1897 | 1 |
| Raw-SetBCE | ablation | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 |
| OPAL-SetBCE | ours | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 |

Raw-SetBCE is ahead of OPAL-SetBCE by 0.0054 NLL / 0.0855 PPL. OPAL is not currently the best learned method on WikiText-2.

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

## Why OPAL Did Not Win Yet

- OPAL beats Static best-on-val C6, so the mask application and learned routing path are not completely broken.
- OPAL loses narrowly to Raw-SetBCE and also loses to the stronger-access layerwise hidden router, which suggests `prefix_hk_raw_attn` is not automatically the best signal for public LM windows.
- OPAL produces more unique masks than Raw, but WikiText-2 PPL rewards mask quality, not mask diversity.
- The raw embedding baseline may be a lower-variance content prior for contiguous LM windows, while H^k may add noisy teacher-prefix features when the training set is only m=2000 windows.
- PuDDing-style and IG-style both collapse to the static `ends_heavy` candidate on all 289 test windows, so OPAL beating them mostly says OPAL beats candidate-library selection, not that it beats all adaptive related-work styles.

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
| Raw-SetBCE | ablation | 2.7618 | 15.8283 | 0.5464 | 6.6634 | 8 | 1.000 |
| OPAL-SetBCE | ours | 2.7672 | 15.9138 | 0.5518 | 6.7489 | 15 | 1.000 |

Dynamic ranking among learned/adaptive methods:

| rank | method | note |
|---:|---|---|
| 1 | layerwise_hidden_router | stronger-access in-framework baseline; not a full Dr.LLM reproduction |
| 2 | Raw-SetBCE | same final Delta_NLL set labels, raw prefix only |
| 3 | OPAL-SetBCE | ours; beats static/PuDDing/IG but not Raw/layerwise |
| 4 | PuDDing-style / IG-style | both select `ends_heavy` for every test window |

Selected candidate distributions:

- PuDDing-style: `{"ends_heavy": 289}`
- IG-style: `{"ends_heavy": 289}`

## OPAL Tuning Table

| setting | status | note |
|---|---|---|
| prefix tokens 256 | complete | Current main run; OPAL beats static but loses Raw narrowly. |
| prefix tokens 512 | pending | Run only if related baselines do not rescue the story. |
| validation checkpoint selection | pending | Next priority after prefix 512. |
| static prior / swap q=1/2 | pending | Use only after the simpler checks. |

## Final Judgment

- current decision: **appendix only**
- reason: OPAL beats Static best-on-val C6, PuDDing-style, and IG-style, but loses to Raw-SetBCE and the stronger-access layerwise_hidden_router.
- recommended wording: WikiText-2 is a public LM sanity / partial generalization result, not a main OPAL superiority claim.
- do not write: "OPAL is best on WikiText-2." The table does not support that.
- safe write: "On WikiText-2, OPAL improves over fixed/candidate-library skipping baselines but remains slightly behind a raw-prefix set router and a stronger-access layerwise hidden router."

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
