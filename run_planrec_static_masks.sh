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
STRATEGIES=${STRATEGIES:-uniform,shortgpt,first_k,last_k,ends_heavy,middle_heavy}
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

IFS=',' read -r -a strategy_list <<< "$STRATEGIES"
for strategy in "${strategy_list[@]}"; do
  strategy=$(echo "$strategy" | xargs)
  [[ -z "$strategy" ]] && continue
  accelerate launch --num_processes "$NUM_GPUS" ./eval_planrec_opal.py \
    --method static \
    --teacher_model "$MODEL_PATH" \
    --test_file "$test_file" \
    --info_file "$info_file" \
    --category "$CATEGORY" \
    --batch_size "$BATCH_SIZE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --top_k_items "$TOP_K_ITEMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --precision "$PRECISION" \
    --static_strategy "$strategy" \
    --compensation "$COMPENSATION" \
    --max_compensated_skipped_layers "$MAX_COMPENSATED_SKIPPED_LAYERS" \
    --comp_rank "$COMP_RANK" \
    --seed "$SEED" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "${CATEGORY}_static_mask_${strategy}_k${TOP_K_LAYERS}_${COMPENSATION}_r${COMP_RANK}_seed${SEED}"
done

python3 ./summarize_planrec_results.py --output_dir "$OUTPUT_DIR"
