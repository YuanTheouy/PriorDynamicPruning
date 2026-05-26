#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${REPO_ROOT}/transformers/src:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}
CATEGORY=${CATEGORY:-Office_Products}
OUTPUT_DIR=${OUTPUT_DIR:-./results/planrec_experiments}
DATA_SPLIT=${DATA_SPLIT:-test}
RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-}
BATCH_SIZE=${BATCH_SIZE:-4}
TOP_K_LAYERS=${TOP_K_LAYERS:-21}
TOP_K_ITEMS=${TOP_K_ITEMS:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_GPUS=${NUM_GPUS:-8}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16}
WARMUP_BATCHES=${WARMUP_BATCHES:-0}
TIMED_BATCHES=${TIMED_BATCHES:-0}
MAX_BATCHES=${MAX_BATCHES:-25}
BUDGETS=${BUDGETS:-14,18,21,24,28}
MASK_LIBRARY=${MASK_LIBRARY:-${OUTPUT_DIR}/raw_json/${CATEGORY}_masks.json}
MASK_STRATEGIES=${MASK_STRATEGIES:-uniform,shortgpt,first_k,last_k,ends_heavy,middle_heavy}
INPUT_GUIDED_FEATURE_SET=${INPUT_GUIDED_FEATURE_SET:-length_hash}
INPUT_GUIDED_NUM_BINS=${INPUT_GUIDED_NUM_BINS:-16}

if [[ "$DATA_SPLIT" != "test" && -z "$RUN_NAME_SUFFIX" ]]; then
  RUN_NAME_SUFFIX="_${DATA_SPLIT}"
fi

test_file=$(ls ./data/Amazon/${DATA_SPLIT}/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing MODEL_PATH: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$test_file" || ! -f "$info_file" ]]; then
  echo "Missing ${DATA_SPLIT}/info files for CATEGORY=$CATEGORY"
  exit 1
fi

accelerate launch --num_processes "$NUM_GPUS" ./eval_planrec_opal.py \
  --method oracle \
  --teacher_model "$MODEL_PATH" \
  --test_file "$test_file" \
  --info_file "$info_file" \
  --category "$CATEGORY" \
  --batch_size "$BATCH_SIZE" \
  --top_k_layers "$TOP_K_LAYERS" \
  --top_k_items "$TOP_K_ITEMS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --budgets "$BUDGETS" \
  --mask_strategies "$MASK_STRATEGIES" \
  --save_mask_library "$MASK_LIBRARY" \
  --input_guided_feature_set "$INPUT_GUIDED_FEATURE_SET" \
  --input_guided_num_bins "$INPUT_GUIDED_NUM_BINS" \
  --save_oracle_candidates \
  --seed "$SEED" \
  --warmup_batches "$WARMUP_BATCHES" \
  --timed_batches "$TIMED_BATCHES" \
  --max_batches "$MAX_BATCHES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "${CATEGORY}${RUN_NAME_SUFFIX}_oracle_masks_k${TOP_K_LAYERS}_seed${SEED}"
