#!/usr/bin/env bash
set -euo pipefail

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export BASE_PORT="${BASE_PORT:-50000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="/workspace/PriorDynamicPruning/transformers/src:/workspace/PriorDynamicPruning:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MODEL_PATH="${MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}"
export CATEGORY="${CATEGORY:-Office_Products}"
export TRAIN_FILE="${TRAIN_FILE:-/workspace/PriorDynamicPruning/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv}"
export TEST_FILE="${TEST_FILE:-/workspace/PriorDynamicPruning/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-/workspace/PriorDynamicPruning/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"

export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/PriorDynamicPruning/results/planrec_experiments}"
export RISK_DIR="${RISK_DIR:-/workspace/PriorDynamicPruning/results/opal_risk_labels}"
export CKPT_ROOT="${CKPT_ROOT:-/workspace/PriorDynamicPruning/policy_ckpts/final_opal_attn}"

export PRECISION="${PRECISION:-bf16}"
export PREFIX_DEPTH="${PREFIX_DEPTH:-4}"
export SKIP_RATE="${SKIP_RATE:-0.25}"
export TOP_K_LAYERS="${TOP_K_LAYERS:-21}"
export TOP_K_ITEMS="${TOP_K_ITEMS:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export LABEL_MAX_SAMPLES="${LABEL_MAX_SAMPLES:-2000}"
export LABEL_BATCH_SIZE="${LABEL_BATCH_SIZE:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EPOCHS="${EPOCHS:-20}"
export LR="${LR:-1e-4}"
export RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0.1}"
export SKIP_SET_LOSS_WEIGHT="${SKIP_SET_LOSS_WEIGHT:-0.1}"
export HUBER_BETA="${HUBER_BETA:-1.0}"
export ROUTER_DIM="${ROUTER_DIM:-256}"
export ROUTER_HEADS="${ROUTER_HEADS:-4}"
export RECENT_TOKENS="${RECENT_TOKENS:-32}"
export RECENT_DECAY="${RECENT_DECAY:-0.85}"
export WARMUP_BATCHES="${WARMUP_BATCHES:-0}"
export TIMED_BATCHES="${TIMED_BATCHES:-0}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
export SUMMARY_TABLE="${SUMMARY_TABLE:-summary_final_opal_attn_delta_m2000.csv}"
export RUN_GROUP="${RUN_GROUP:-final_opal_attn_delta_m2000}"
export SEEDS="${SEEDS:-42 13 3407}"

cd "$REPO_DIR"
mkdir -p "$RISK_DIR" "$CKPT_ROOT" "$OUTPUT_DIR"

run_accelerate() {
  local port="$NEXT_PORT"
  export NEXT_PORT="$((NEXT_PORT + 1))"
  echo "=== accelerate port ${port}: $* ==="
  accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --main_process_port "$port" \
    --mixed_precision "$PRECISION" \
    --dynamo_backend no \
    "$@"
}

final_json_exists() {
  local run_name="$1"
  test -s "${OUTPUT_DIR}/raw_json/${run_name}.json"
}

build_labels_if_needed() {
  local seed="$1"
  local label_file="$2"
  if [ -s "$label_file" ] && [ "$(wc -l < "$label_file")" -ge "$LABEL_MAX_SAMPLES" ]; then
    echo "=== Reuse risk labels: ${label_file} ==="
    return
  fi
  echo "=== Build random Delta_NLL risk labels seed=${seed} ==="
  run_accelerate ./build_layer_risk_labels.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --max_samples "$LABEL_MAX_SAMPLES" \
    --sample_strategy random \
    --sample_seed "$seed" \
    --objective Delta_NLL \
    --output "$label_file" \
    --batch_size "$LABEL_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$seed"
}

train_router_if_needed() {
  local seed="$1"
  local label_file="$2"
  local variant="$3"
  local router_input="$4"
  local risk_pooling="$5"
  local out_dir="${CKPT_ROOT}/${RUN_GROUP}_seed${seed}/${variant}"
  local ckpt="${out_dir}/risk_router.pt"
  if [ -s "$ckpt" ]; then
    echo "=== Reuse checkpoint: ${ckpt} ==="
    return
  fi
  echo "=== Train ${variant} seed=${seed} ==="
  local args=(
    ./train_opal_risk_router.py
    --teacher_model "$MODEL_PATH"
    --train_file "$TRAIN_FILE"
    --info_file "$INFO_FILE"
    --category "$CATEGORY"
    --risk_label_file "$label_file"
    --router_input "$router_input"
    --prefix_depth "$PREFIX_DEPTH"
    --batch_size "$TRAIN_BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT"
    --skip_set_loss_weight "$SKIP_SET_LOSS_WEIGHT"
    --skip_rate "$SKIP_RATE"
    --top_k_layers "$TOP_K_LAYERS"
    --huber_beta "$HUBER_BETA"
    --router_dim "$ROUTER_DIM"
    --router_heads "$ROUTER_HEADS"
    --recent_tokens "$RECENT_TOKENS"
    --recent_decay "$RECENT_DECAY"
    --output_dir "$out_dir"
    --precision "$PRECISION"
    --loss_log_interval 50
    --seed "$seed"
  )
  if [ "$risk_pooling" != "none" ]; then
    args+=(--risk_pooling "$risk_pooling")
  fi
  run_accelerate "${args[@]}"
}

eval_full_if_needed() {
  local seed="$1"
  local run_name="${CATEGORY}_${RUN_GROUP}_seed${seed}_full"
  if final_json_exists "$run_name"; then
    echo "=== Reuse eval JSON: ${run_name} ==="
    return
  fi
  run_accelerate ./eval_planrec_opal.py \
    --method full \
    --teacher_model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --top_k_items "$TOP_K_ITEMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --precision "$PRECISION" \
    --seed "$seed" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$EVAL_MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name"
}

eval_static_if_needed() {
  local seed="$1"
  local strategy="$2"
  local run_name="${CATEGORY}_${RUN_GROUP}_seed${seed}_static_${strategy}_k${TOP_K_LAYERS}"
  if final_json_exists "$run_name"; then
    echo "=== Reuse eval JSON: ${run_name} ==="
    return
  fi
  run_accelerate ./eval_planrec_opal.py \
    --method static \
    --static_strategy "$strategy" \
    --teacher_model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --top_k_items "$TOP_K_ITEMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --precision "$PRECISION" \
    --seed "$seed" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$EVAL_MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name"
}

eval_router_if_needed() {
  local seed="$1"
  local variant="$2"
  local method="$3"
  local ckpt="${CKPT_ROOT}/${RUN_GROUP}_seed${seed}/${variant}/risk_router.pt"
  local run_name="${CATEGORY}_${RUN_GROUP}_seed${seed}_${variant}_skip${SKIP_RATE}"
  if final_json_exists "$run_name"; then
    echo "=== Reuse eval JSON: ${run_name} ==="
    return
  fi
  run_accelerate ./eval_planrec_opal.py \
    --method "$method" \
    --risk_router_ckpt "$ckpt" \
    --teacher_model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --top_k_items "$TOP_K_ITEMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --precision "$PRECISION" \
    --prefix_depth "$PREFIX_DEPTH" \
    --skip_rate "$SKIP_RATE" \
    --seed "$seed" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$EVAL_MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name"
}

for SEED in $SEEDS; do
  export SEED
  export LABEL_FILE="${RISK_DIR}/${CATEGORY}_risk_Delta_NLL_random_m${LABEL_MAX_SAMPLES}_seed${SEED}.jsonl"
  build_labels_if_needed "$SEED" "$LABEL_FILE"

  train_router_if_needed "$SEED" "$LABEL_FILE" raw_input_risk raw_embedding none
  train_router_if_needed "$SEED" "$LABEL_FILE" prefix_hk_mean prefix_hk mean
  train_router_if_needed "$SEED" "$LABEL_FILE" prefix_hk_last prefix_hk last
  train_router_if_needed "$SEED" "$LABEL_FILE" prefix_hk_raw_last prefix_hk_raw_last none
  train_router_if_needed "$SEED" "$LABEL_FILE" prefix_hk_raw_fusion prefix_hk_raw_fusion none
  train_router_if_needed "$SEED" "$LABEL_FILE" prefix_hk_raw_attn prefix_hk_raw_attn none

  eval_full_if_needed "$SEED"
  for STATIC_STRATEGY in ends_heavy uniform shortgpt first_k last_k middle_heavy; do
    eval_static_if_needed "$SEED" "$STATIC_STRATEGY"
  done

  eval_router_if_needed "$SEED" raw_input_risk raw_input_risk
  eval_router_if_needed "$SEED" prefix_hk_mean opal_risk
  eval_router_if_needed "$SEED" prefix_hk_last opal_risk
  eval_router_if_needed "$SEED" prefix_hk_raw_last opal_risk
  eval_router_if_needed "$SEED" prefix_hk_raw_fusion opal_risk
  eval_router_if_needed "$SEED" prefix_hk_raw_attn opal_risk
done

echo "=== Summarize final OPAL attention experiments ==="
python3 summarize_planrec_results.py \
  --output_dir "$OUTPUT_DIR" \
  --table_name "$SUMMARY_TABLE" \
  --run_name_contains "$RUN_GROUP" \
  --exclude_debug_sanity

cat "$OUTPUT_DIR/tables/quality_retention.md"
echo "=== Done final OPAL attention experiments ==="
