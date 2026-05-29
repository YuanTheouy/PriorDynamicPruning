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
