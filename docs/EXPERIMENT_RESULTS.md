# OPAL Experiment Notes

Updated: 2026-05-29

## Current Claim

This project is now framed as a general LLM layer-skipping / acceleration method, not a recommendation-only method. The primary quality metrics are:

```text
Delta_NLL lower is better
Delta_PPL lower is better
KL(full || skip) lower is better
```

`NDCG@10` and `retention_NDCG@10` are retained as PlanRec downstream sanity checks.

## Implemented Routers

The current risk router family predicts per-layer skip risk from prompt-only inputs and skips the lowest-risk layers under a fixed skip-rate budget.

| router_input | description |
|---|---|
| `raw_embedding` | raw prompt embedding mean baseline |
| `prefix_hk` + `--risk_pooling last` | last prompt token from early hidden state `H^k` |
| `prefix_hk` + `--risk_pooling mean` | mean pooled early hidden state `H^k` |
| `prefix_hk_raw_attn` | layer-query cross-attention over raw token embeddings and `H^k` sequence |
| `prefix_hk_raw_attn_hk_last_resid` | attention router plus residual `H^k` last-token risk head |
| `prefix_hk_raw_attn_raw_hk_last_resid` | attention router plus residual raw-last and `H^k` last-token risk heads |

The residual variants compute:

```text
attn_hk_last_resid:
  risk = risk_attn + risk_hk_last

attn_raw_hk_last_resid:
  risk = risk_attn + risk_raw_last + risk_hk_last
```

All router variants are prompt-only. They do not see target items, answers, generated continuations, or validation labels at inference.

## m2000 Three-Seed Result Snapshot

Setting:

```text
dataset: Office_Products
model: MiniOneRec Office checkpoint
skip_rate: 0.25
num_layers: 28
kept_layers: 21
skipped_layers: 7
risk objective: Delta_NLL
risk labels: random m2000 per seed
seeds: 42 / 13 / 3407
prefix_depth: 4
```

Three-seed averages:

| method | mean Delta_NLL | mean Delta_PPL | mean KL | mean NDCG@10 | mean retention | mean unique_masks |
|---|---:|---:|---:|---:|---:|---:|
| `prefix_hk_last` | -0.245278 | -5.851168 | 1.729122 | 0.1454 | 0.8297 | 70.7 |
| `raw_input_risk` | -0.161006 | -3.981497 | 1.775815 | 0.1443 | 0.8234 | 40.0 |
| `prefix_hk_raw_attn` | -0.147499 | -3.696668 | 1.705995 | 0.1373 | 0.7835 | 11.3 |
| `prefix_hk_mean` | -0.125229 | -3.150761 | 1.841142 | 0.1461 | 0.8338 | 228.0 |
| `prefix_hk_raw_fusion` | -0.084150 | -2.207629 | 1.992584 | 0.1363 | 0.7780 | 420.3 |
| `prefix_hk_raw_last` | 0.150101 | 4.741947 | 2.118066 | 0.1343 | 0.7663 | 74.0 |
| static `ends_heavy_k21` | 0.055907 | 1.589131 | 1.676606 | 0.1479 | 0.8443 | 1.0 |
| static `uniform_k21` | 0.389392 | 13.157743 | 1.666112 | 0.1456 | 0.8310 | 1.0 |
| static `shortgpt_k21` | 1.835030 | 145.520144 | 1.959648 | 0.1271 | 0.7254 | 1.0 |

Reading:

- Dynamic risk-router methods beat static baselines on `Delta_NLL` / `Delta_PPL` under the same 25% skip budget.
- `prefix_hk_last` is currently the strongest averaged likelihood-preservation variant.
- `prefix_hk_raw_attn` has the best dynamic KL average and a compact mask set, but it is not yet the best averaged `Delta_NLL` variant.
- The next check is whether larger label sets and residual heads make attention behave like the expected superset.

## m10000 Seed-42 Partial Result

Setting:

```text
dataset: Office_Products
model: MiniOneRec Office checkpoint
skip_rate: 0.25
num_layers: 28
kept_layers: 21
skipped_layers: 7
risk objective: Delta_NLL
risk labels: random m10000
seed: 42
prefix_depth: 4
eval_max_batches: 500
```

This is a partial result from the large-label run. It already includes full, static baselines, raw input risk, `H^k` pooled variants, the layer-query attention router, and both residual-attention variants.

| method | Delta_NLL | Delta_PPL | KL | NDCG@10 | retention | unique_masks |
|---|---:|---:|---:|---:|---:|---:|
| `raw_input_risk` | -0.409425 | -10.659394 | 1.741506 | 0.1158 | 0.8349 | 194 |
| `prefix_hk_last` | -0.407977 | -10.628873 | 1.697471 | 0.1195 | 0.8619 | 351 |
| `prefix_hk_mean` | -0.362123 | -9.638899 | 1.771152 | 0.1156 | 0.8339 | 560 |
| `prefix_hk_raw_attn_raw_hk_last_resid` | -0.219982 | -6.265108 | 1.873614 | 0.1111 | 0.8013 | 693 |
| `prefix_hk_raw_attn` | -0.201028 | -5.777893 | 1.896394 | 0.1126 | 0.8119 | 255 |
| `prefix_hk_raw_attn_hk_last_resid` | -0.100685 | -3.038918 | 2.089884 | 0.1049 | 0.7569 | 506 |
| static `ends_heavy_k21` | 0.001659 | 0.052694 | 1.670297 | 0.1160 | 0.8369 | 1 |
| static `uniform_k21` | 0.334089 | 12.585228 | 1.693851 | 0.1159 | 0.8361 | 1 |
| static `shortgpt_k21` | 1.759649 | 152.621410 | 1.926826 | 0.1000 | 0.7211 | 1 |
| static `first_k_k21` | 7.486680 | 56573.737068 | 8.093979 | 0.0869 | 0.6266 | 1 |
| static `last_k_k21` | 8.736554 | 197515.798719 | 8.834929 | 0.0010 | 0.0075 | 1 |
| static `middle_heavy_k21` | 11.244041 | 2424677.329996 | 10.958963 | 0.0027 | 0.0196 | 1 |

Reading:

- Increasing labels from m2000 to m10000 substantially improves the supervised dynamic routers on `Delta_NLL` / `Delta_PPL`.
- `raw_input_risk` and `prefix_hk_last` are essentially tied on `Delta_NLL` / `Delta_PPL`; `prefix_hk_last` is slightly better on KL and downstream `NDCG@10`.
- `prefix_hk_mean` is also strong, but weaker than `prefix_hk_last` on all three primary quality metrics.
- `prefix_hk_raw_attn` improves over static baselines on `Delta_NLL` / `Delta_PPL`, but it still does not beat `prefix_hk_last` or `raw_input_risk`.
- `prefix_hk_raw_attn_raw_hk_last_resid` improves over plain attention, while `prefix_hk_raw_attn_hk_last_resid` hurts. Simple additive residual fusion is therefore not yet the right attention-supernet formulation.
- Static `ends_heavy_k21` and `uniform_k21` remain competitive on KL, but they are much worse than the supervised dynamic routers on `Delta_NLL` / `Delta_PPL`.

Current seed-42 paper-facing interpretation:

```text
With m10000 one-layer-drop supervision, prompt-only risk routing becomes much
stronger. The strongest seed-42 dynamic methods are raw-input risk prediction
and prefix-H^k last-token risk prediction, with prefix-H^k last showing better
KL and downstream retention. Layer-query attention remains useful as an
ablation, but it is not yet the best main method without a better fusion or
training recipe.
```

## Large-Label Follow-Up

The large follow-up runner uses GPU IDs `0,1,2,3,4,5,6,7`, longer training, and robust accelerate port retry.

Default run:

```bash
cd /workspace/PriorDynamicPruning
git fetch origin codex/opal-llm-experiments
git pull --ff-only origin codex/opal-llm-experiments
git rev-parse --short HEAD

bash ./run_final_opal_attn_large_gpu01234567.sh 2>&1 | tee /tmp/final_opal_attn_large_m10000_gpu01234567.log
```

The default is:

```text
OPAL_LABEL_MAX_SAMPLES=10000
OPAL_EPOCHS=40
OPAL_SEEDS="42 13 3407"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

m5000:

```bash
cd /workspace/PriorDynamicPruning
OPAL_LABEL_MAX_SAMPLES=5000 \
bash ./run_final_opal_attn_large_gpu01234567.sh 2>&1 | tee /tmp/final_opal_attn_large_m5000_gpu01234567.log
```

Full train labels for the current Office split:

```bash
cd /workspace/PriorDynamicPruning
OPAL_LABEL_MAX_SAMPLES=38924 \
OPAL_EPOCHS=40 \
bash ./run_final_opal_attn_large_gpu01234567.sh 2>&1 | tee /tmp/final_opal_attn_large_full_gpu01234567.log
```

The runner reuses completed risk-label jsonl files, checkpoints, and final eval JSON files when present.

To inspect partial results during a run:

```bash
cd /workspace/PriorDynamicPruning
export OUTPUT_DIR=/workspace/PriorDynamicPruning/results/planrec_experiments

python3 summarize_planrec_results.py \
  --output_dir "$OUTPUT_DIR" \
  --table_name summary_final_opal_attn_large_partial.csv \
  --run_name_contains final_opal_attn_large_delta \
  --exclude_debug_sanity

cat "$OUTPUT_DIR/tables/quality_retention.md"
```

## 2026-05-29 Final-KL Greedy Set Supervision

Purpose:

```text
Test whether prefix_hk_raw_attn was unstable because one-layer-drop labels
optimize a first-order proxy, while deployment skips 7 layers jointly.
```

Incremental change:

```text
old label:
  risk_l = KL or Delta_NLL after dropping layer l alone

new label:
  skip_mask = 7 skipped layers selected by forward greedy search that minimizes
              final KL(full logits || skip-mask logits)
```

Constraints:

```text
no soft gate
no Gumbel
no STE
no inference-time search
no beam2 / topM / swap / lookahead in the first pass
```

Fixed experiment:

```text
dataset: Office_Products
seed: 42
label samples: m500
skip_rate: 0.25
protected_head: 4
protected_tail: 2
allowed layers: [4, ..., num_layers - 3]
search: forward greedy
objective: final_KL
compared routers: raw_input_risk vs prefix_hk_raw_attn
training loss: BCEWithLogits(-pred_risk, greedy_skip_mask)
inference: one router forward, skip the 7 lowest predicted-risk layers
```

Implemented files:

| file | purpose |
|---|---|
| `build_final_kl_greedy_set_labels.py` | builds `skip_mask` labels via final-KL forward greedy search |
| `train_opal_risk_router.py` | auto-detects `skip_mask` labels and trains pure BCE skip-set supervision |
| `run_final_kl_greedy_set_supervision_gpu01234567.sh` | one-shot server runner for m500 seed42 raw vs attention |

Server command:

```bash
cd /workspace/PriorDynamicPruning
git fetch origin codex/opal-llm-experiments
git pull --ff-only origin codex/opal-llm-experiments
git rev-parse --short HEAD

bash ./run_final_kl_greedy_set_supervision_gpu01234567.sh 2>&1 | tee /tmp/final_kl_greedy_set_m500_seed42_gpu01234567.log
```

Inspect results:

```bash
cd /workspace/PriorDynamicPruning
export OUTPUT_DIR=/workspace/PriorDynamicPruning/results/planrec_experiments
export RUN_GROUP=final_kl_greedy_set_m500_seed42

python3 summarize_planrec_results.py \
  --output_dir "$OUTPUT_DIR" \
  --table_name summary_${RUN_GROUP}.csv \
  --run_name_contains "$RUN_GROUP" \
  --exclude_debug_sanity

cat "$OUTPUT_DIR/tables/quality_retention.md"
```

Result status:

```text
Pending server run. Fill this section after the m500 greedy-set experiment
produces raw_input_risk and prefix_hk_raw_attn eval JSON files.
```
