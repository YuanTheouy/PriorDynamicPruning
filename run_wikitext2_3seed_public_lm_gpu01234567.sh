#!/usr/bin/env bash
set -euo pipefail

# Fixed WikiText-2 3-seed public LM supplement runner.
# Default seeds intentionally exclude seed42 because the required seed42
# artifacts were already produced. This script only orchestrates existing
# runners; it does not change methods, losses, routers, or hyperparameters.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"

export WIKITEXT_MODEL_PATH="${WIKITEXT_MODEL_PATH:-/workspace/ckpts/Qwen2.5-1.5B}"
export WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
export WIKITEXT_LABEL_SAMPLES="${WIKITEXT_LABEL_SAMPLES:-2000}"
export WIKITEXT_EVAL_WINDOWS="${WIKITEXT_EVAL_WINDOWS:-512}"
export WIKITEXT_EPOCHS="${WIKITEXT_EPOCHS:-40}"
export WIKITEXT_SEQ_LEN="${WIKITEXT_SEQ_LEN:-1024}"
export WIKITEXT_ROUTER_PREFIX_TOKENS="${WIKITEXT_ROUTER_PREFIX_TOKENS:-256}"
export WIKITEXT_SKIP_RATE="${WIKITEXT_SKIP_RATE:-0.25}"
export WIKITEXT_SKIP_COUNT="${WIKITEXT_SKIP_COUNT:-7}"
export WIKITEXT_PROTECTED_HEAD="${WIKITEXT_PROTECTED_HEAD:-4}"
export WIKITEXT_PROTECTED_TAIL="${WIKITEXT_PROTECTED_TAIL:-2}"
export WIKITEXT_THREE_SEED_SEEDS="${WIKITEXT_THREE_SEED_SEEDS:-13 3407}"
export WIKITEXT_THREE_SEED_BASE_PORT="${WIKITEXT_THREE_SEED_BASE_PORT:-58800}"
export WIKITEXT_VALCKPT_PARALLEL_WORKERS="${WIKITEXT_VALCKPT_PARALLEL_WORKERS:-8}"

cd "$REPO_DIR"

echo "=== WikiText-2 fixed 3-seed supplement ==="
echo "Seeds to run: ${WIKITEXT_THREE_SEED_SEEDS}"
echo "Seed42 is not included by default because it is already complete."
echo "Model: ${WIKITEXT_MODEL_PATH}"
echo "Dataset: ${WIKITEXT_DATASET_DISK_PATH}"

seed_index=0
for seed in $WIKITEXT_THREE_SEED_SEEDS; do
  base="$((WIKITEXT_THREE_SEED_BASE_PORT + seed_index * 400))"
  report_dir="results/wikitext2_public_lm_sanity/three_seed_reports/seed${seed}"
  mkdir -p "$report_dir"

  export WIKITEXT_SEED="$seed"

  echo "=== seed ${seed}: Full/static/Raw final/OPAL final ==="
  WIKITEXT_BASE_PORT="$base" \
  WIKITEXT_REPORT_MD="${report_dir}/public_sanity_seed${seed}.md" \
  bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh

  echo "=== seed ${seed}: PuDDing/IG/layerwise ==="
  WIKITEXT_BASE_PORT="$((base + 100))" \
  WIKITEXT_REPORT_MD="${report_dir}/public_sanity_seed${seed}.md" \
  WIKITEXT_RELATED_REPORT_MD="${report_dir}/related_seed${seed}.md" \
  WIKITEXT_RUN_PUDDING=1 \
  WIKITEXT_RUN_IG=1 \
  WIKITEXT_RUN_LAYERWISE=1 \
  bash ./run_wikitext2_related_baselines_gpu01234567.sh

  echo "=== seed ${seed}: Raw best-on-val ==="
  WIKITEXT_BASE_PORT="$((base + 200))" \
  WIKITEXT_VALCKPT_PARALLEL_WORKERS="$WIKITEXT_VALCKPT_PARALLEL_WORKERS" \
  bash ./run_wikitext2_raw_val_ckpt_select_gpu01234567.sh

  echo "=== seed ${seed}: OPAL best-on-val ==="
  WIKITEXT_BASE_PORT="$((base + 300))" \
  WIKITEXT_VALCKPT_PARALLEL_WORKERS="$WIKITEXT_VALCKPT_PARALLEL_WORKERS" \
  bash ./run_wikitext2_opal_val_ckpt_select_gpu01234567.sh

  seed_index="$((seed_index + 1))"
done

echo "=== Regenerate 3-seed WikiText-2 public LM report ==="
python3 ./summarize_wikitext2_3seed_public_lm_results.py

echo "=== Done: WikiText-2 fixed 3-seed supplement ==="
