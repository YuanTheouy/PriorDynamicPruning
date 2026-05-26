#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${REPO_ROOT}/transformers/src:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}
CATEGORY=${CATEGORY:-Office_Products}
OUTPUT_DIR=${OUTPUT_DIR:-./results/planrec_experiments}
BATCH_SIZE=${BATCH_SIZE:-8}
TOP_K_LAYERS=${TOP_K_LAYERS:-21}
TOP_K_ITEMS=${TOP_K_ITEMS:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_GPUS=${NUM_GPUS:-8}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16}
WARMUP_BATCHES=${WARMUP_BATCHES:-1}
TIMED_BATCHES=${TIMED_BATCHES:-0}
MAX_BATCHES=${MAX_BATCHES:-0}
BUDGETS=${BUDGETS:-14,18,21,24,28}
GROUP_BY_MASK=${GROUP_BY_MASK:-1}
INPUT_GUIDED_SELECTOR=${INPUT_GUIDED_SELECTOR:-length_hash}
INPUT_GUIDED_CALIBRATION_JSON=${INPUT_GUIDED_CALIBRATION_JSON:-}
INPUT_GUIDED_FEATURE_SET=${INPUT_GUIDED_FEATURE_SET:-length_hash}
INPUT_GUIDED_NUM_BINS=${INPUT_GUIDED_NUM_BINS:-16}
COMPENSATION=${COMPENSATION:-none}
MAX_COMPENSATED_SKIPPED_LAYERS=${MAX_COMPENSATED_SKIPPED_LAYERS:-0}
COMP_RANK=${COMP_RANK:-0}

test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing MODEL_PATH: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$test_file" || ! -f "$info_file" ]]; then
  echo "Missing test/info files for CATEGORY=$CATEGORY"
  exit 1
fi

group_flag=()
if [[ "$GROUP_BY_MASK" == "1" ]]; then
  group_flag=(--group_by_mask)
fi

calibration_flag=()
if [[ -n "$INPUT_GUIDED_CALIBRATION_JSON" ]]; then
  calibration_flag=(--input_guided_calibration_json "$INPUT_GUIDED_CALIBRATION_JSON")
fi

accelerate launch --num_processes "$NUM_GPUS" ./eval_planrec_opal.py \
  --method input_guided \
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
  --input_guided_selector "$INPUT_GUIDED_SELECTOR" \
  --input_guided_feature_set "$INPUT_GUIDED_FEATURE_SET" \
  --input_guided_num_bins "$INPUT_GUIDED_NUM_BINS" \
  --compensation "$COMPENSATION" \
  --max_compensated_skipped_layers "$MAX_COMPENSATED_SKIPPED_LAYERS" \
  --comp_rank "$COMP_RANK" \
  "${calibration_flag[@]}" \
  "${group_flag[@]}" \
  --seed "$SEED" \
  --warmup_batches "$WARMUP_BATCHES" \
  --timed_batches "$TIMED_BATCHES" \
  --max_batches "$MAX_BATCHES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "${CATEGORY}_input_guided_${INPUT_GUIDED_SELECTOR}_${INPUT_GUIDED_FEATURE_SET}_k${TOP_K_LAYERS}_${COMPENSATION}_r${COMP_RANK}_seed${SEED}"
