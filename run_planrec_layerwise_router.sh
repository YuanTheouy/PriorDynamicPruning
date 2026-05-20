#!/bin/bash
set -euo pipefail

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
GROUP_BY_TEMPLATE=${GROUP_BY_TEMPLATE:-0}
TEMPLATE_LIBRARY=${TEMPLATE_LIBRARY:-${OUTPUT_DIR}/raw_json/${CATEGORY}_templates.json}
STUDENT_CKPT=${STUDENT_CKPT:-}
POLICY_CKPT=${POLICY_CKPT:-}

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

if [[ -z "$STUDENT_CKPT" ]]; then
  STUDENT_CKPT=$(ls -v ./student_ckpts/${CATEGORY}/student_epoch_*.pt 2>/dev/null | tail -1)
fi
if [[ -z "$POLICY_CKPT" ]]; then
  POLICY_CKPT=$(ls -v ./policy_ckpts/unfrozon/${CATEGORY}/policy_epoch_*.pt 2>/dev/null | tail -1)
fi

if [[ ! -f "$STUDENT_CKPT" || ! -f "$POLICY_CKPT" ]]; then
  echo "Missing STUDENT_CKPT or POLICY_CKPT"
  exit 1
fi

group_flag=()
if [[ "$GROUP_BY_TEMPLATE" == "1" ]]; then
  group_flag=(--group_by_template)
fi

accelerate launch --num_processes "$NUM_GPUS" ./eval_planrec_opal.py \
  --method layerwise_router \
  --teacher_model "$MODEL_PATH" \
  --student_ckpt "$STUDENT_CKPT" \
  --policy_ckpt "$POLICY_CKPT" \
  --test_file "$test_file" \
  --info_file "$info_file" \
  --category "$CATEGORY" \
  --batch_size "$BATCH_SIZE" \
  --top_k_layers "$TOP_K_LAYERS" \
  --top_k_items "$TOP_K_ITEMS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --budgets "$BUDGETS" \
  --save_template_library "$TEMPLATE_LIBRARY" \
  --dynamic_selection topk_mask \
  "${group_flag[@]}" \
  --seed "$SEED" \
  --warmup_batches "$WARMUP_BATCHES" \
  --timed_batches "$TIMED_BATCHES" \
  --max_batches "$MAX_BATCHES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "${CATEGORY}_layerwise_router_k${TOP_K_LAYERS}_seed${SEED}"
