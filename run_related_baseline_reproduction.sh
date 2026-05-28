#!/usr/bin/env bash
set -euo pipefail

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
export BASE_PORT="${BASE_PORT:-51000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MODEL_PATH="${MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}"
export CATEGORY="${CATEGORY:-Office_Products}"
export TRAIN_FILE="${TRAIN_FILE:-${REPO_DIR}/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv}"
export TEST_FILE="${TEST_FILE:-${REPO_DIR}/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv}"
export INFO_FILE="${INFO_FILE:-${REPO_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/planrec_experiments}"
export RISK_DIR="${RISK_DIR:-${REPO_DIR}/results/opal_risk_labels}"
export RELATED_DIR="${RELATED_DIR:-${REPO_DIR}/policy_ckpts/related_baselines}"
export OPAL_CKPT_ROOT="${OPAL_CKPT_ROOT:-${REPO_DIR}/policy_ckpts/final_opal_attn}"

export PRECISION="${PRECISION:-bf16}"
export PREFIX_DEPTH="${PREFIX_DEPTH:-4}"
export SKIP_RATE="${SKIP_RATE:-0.25}"
export TOP_K_LAYERS="${TOP_K_LAYERS:-21}"
export TOP_K_ITEMS="${TOP_K_ITEMS:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export LABEL_MAX_SAMPLES="${LABEL_MAX_SAMPLES:-2000}"
export LABEL_BATCH_SIZE="${LABEL_BATCH_SIZE:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export LAYERWISE_TRAIN_BATCH_SIZE="${LAYERWISE_TRAIN_BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EPOCHS="${EPOCHS:-20}"
export LAYERWISE_EPOCHS="${LAYERWISE_EPOCHS:-5}"
export LR="${LR:-1e-4}"
export RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0.1}"
export SKIP_SET_LOSS_WEIGHT="${SKIP_SET_LOSS_WEIGHT:-0.1}"
export HUBER_BETA="${HUBER_BETA:-1.0}"
export WARMUP_BATCHES="${WARMUP_BATCHES:-0}"
export TIMED_BATCHES="${TIMED_BATCHES:-0}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
export CANDIDATE_COUNT="${CANDIDATE_COUNT:-16}"
export IG_CLUSTERS="${IG_CLUSTERS:-8}"
export SEED="${SEED:-42}"
export RUN_GROUP="${RUN_GROUP:-related_baselines_delta_m2000_seed${SEED}}"
export SUMMARY_TABLE="${SUMMARY_TABLE:-summary_${RUN_GROUP}.csv}"
export RAW_INPUT_CKPT="${RAW_INPUT_CKPT:-${OPAL_CKPT_ROOT}/final_opal_attn_delta_m2000_seed${SEED}/raw_input_risk/risk_router.pt}"
export OPAL_LQ_CKPT="${OPAL_LQ_CKPT:-${OPAL_CKPT_ROOT}/final_opal_attn_delta_m2000_seed${SEED}/prefix_hk_raw_attn/risk_router.pt}"

cd "$REPO_DIR"
mkdir -p "$RISK_DIR" "$RELATED_DIR" "$OUTPUT_DIR"

run_accelerate() {
  local port="$NEXT_PORT"
  while ! python3 -c 'import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
' "$port"; do
    echo "=== Skip occupied accelerate port ${port} ==="
    port="$((port + 1))"
  done
  export NEXT_PORT="$((port + 1))"
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

export LABEL_FILE="${RISK_DIR}/${CATEGORY}_risk_Delta_NLL_random_m${LABEL_MAX_SAMPLES}_seed${SEED}.jsonl"
export CANDIDATE_LABEL_FILE="${RISK_DIR}/${CATEGORY}_candidate_Delta_NLL_C${CANDIDATE_COUNT}_m${LABEL_MAX_SAMPLES}_seed${SEED}.jsonl"
export SEED_DIR="${RELATED_DIR}/${RUN_GROUP}"
export PUDDING_DIR="${SEED_DIR}/pudding_prompt_candidate_C${CANDIDATE_COUNT}"
export IG_DIR="${SEED_DIR}/ig_cluster"
export LAYERWISE_DIR="${SEED_DIR}/layerwise_hidden_router"

if [ -s "$LABEL_FILE" ] && [ "$(wc -l < "$LABEL_FILE")" -ge "$LABEL_MAX_SAMPLES" ]; then
  echo "=== Reuse risk labels: ${LABEL_FILE} ==="
else
  run_accelerate ./build_layer_risk_labels.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --max_samples "$LABEL_MAX_SAMPLES" \
    --sample_strategy random \
    --sample_seed "$SEED" \
    --objective Delta_NLL \
    --output "$LABEL_FILE" \
    --batch_size "$LABEL_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$SEED"
fi

if [ -s "$CANDIDATE_LABEL_FILE" ] && [ "$(wc -l < "$CANDIDATE_LABEL_FILE")" -ge "$LABEL_MAX_SAMPLES" ]; then
  echo "=== Reuse candidate labels: ${CANDIDATE_LABEL_FILE} ==="
else
  run_accelerate ./build_candidate_mask_labels.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$LABEL_FILE" \
    --use_risk_label_sample_ids \
    --candidate_count "$CANDIDATE_COUNT" \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --objective Delta_NLL \
    --output "$CANDIDATE_LABEL_FILE" \
    --batch_size "$LABEL_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$SEED"
fi

if [ -s "${PUDDING_DIR}/candidate_router.pt" ]; then
  echo "=== Reuse PuDDing-style router: ${PUDDING_DIR}/candidate_router.pt ==="
else
  run_accelerate ./train_prompt_candidate_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --candidate_label_file "$CANDIDATE_LABEL_FILE" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --huber_beta "$HUBER_BETA" \
    --output_dir "$PUDDING_DIR" \
    --precision "$PRECISION" \
    --seed "$SEED"
fi

if [ -s "${LAYERWISE_DIR}/layerwise_hidden_router.pt" ]; then
  echo "=== Reuse layerwise hidden router: ${LAYERWISE_DIR}/layerwise_hidden_router.pt ==="
else
  run_accelerate ./train_layerwise_hidden_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$LABEL_FILE" \
    --risk_pooling mean \
    --batch_size "$LAYERWISE_TRAIN_BATCH_SIZE" \
    --epochs "$LAYERWISE_EPOCHS" \
    --lr "$LR" \
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT" \
    --skip_set_loss_weight "$SKIP_SET_LOSS_WEIGHT" \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --huber_beta "$HUBER_BETA" \
    --output_dir "$LAYERWISE_DIR" \
    --precision "$PRECISION" \
    --seed "$SEED"
fi

eval_method_if_needed() {
  local run_name="$1"
  shift
  if final_json_exists "$run_name"; then
    echo "=== Reuse eval JSON: ${run_name} ==="
    return
  fi
  run_accelerate ./eval_planrec_opal.py \
    "$@" \
    --teacher_model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --top_k_items "$TOP_K_ITEMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --precision "$PRECISION" \
    --prefix_depth "$PREFIX_DEPTH" \
    --skip_rate "$SKIP_RATE" \
    --seed "$SEED" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$EVAL_MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name"
}

eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_pudding_prompt_candidate_C${CANDIDATE_COUNT}" \
  --method pudding_prompt_candidate \
  --candidate_router_ckpt "${PUDDING_DIR}/candidate_router.pt"

for K in $IG_CLUSTERS; do
  artifact="${IG_DIR}/ig_cluster_K${K}.pt"
  if [ -s "$artifact" ]; then
    echo "=== Reuse IG cluster artifact: ${artifact} ==="
  else
    run_accelerate ./build_ig_cluster_artifact.py \
      --teacher_model "$MODEL_PATH" \
      --train_file "$TRAIN_FILE" \
      --info_file "$INFO_FILE" \
      --category "$CATEGORY" \
      --risk_label_file "$LABEL_FILE" \
      --num_clusters "$K" \
      --normalize_embeddings \
      --skip_rate "$SKIP_RATE" \
      --top_k_layers "$TOP_K_LAYERS" \
      --output "$artifact" \
      --batch_size "$TRAIN_BATCH_SIZE" \
      --precision "$PRECISION" \
      --seed "$SEED"
  fi
  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_ig_cluster_K${K}" \
    --method ig_cluster_mask \
    --ig_cluster_artifact "$artifact"
done

eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_layerwise_hidden_router" \
  --method layerwise_hidden_router \
  --layerwise_router_ckpt "${LAYERWISE_DIR}/layerwise_hidden_router.pt"

eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_full" \
  --method full

for STATIC_STRATEGY in ends_heavy uniform; do
  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_static_${STATIC_STRATEGY}" \
    --method static \
    --static_strategy "$STATIC_STRATEGY"
done

if [ -s "$RAW_INPUT_CKPT" ]; then
  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_raw_input_risk" \
    --method raw_input_risk \
    --risk_router_ckpt "$RAW_INPUT_CKPT"
else
  echo "=== Skip raw_input_risk eval; checkpoint not found: ${RAW_INPUT_CKPT} ==="
fi

if [ -s "$OPAL_LQ_CKPT" ]; then
  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_opal_lq_prefix_hk_raw_attn" \
    --method opal_risk \
    --risk_router_ckpt "$OPAL_LQ_CKPT"
else
  echo "=== Skip OPAL-LQ eval; checkpoint not found: ${OPAL_LQ_CKPT} ==="
fi

python3 summarize_planrec_results.py \
  --output_dir "$OUTPUT_DIR" \
  --table_name "$SUMMARY_TABLE" \
  --run_name_contains "$RUN_GROUP" \
  --exclude_debug_sanity

cat "$OUTPUT_DIR/tables/quality_retention.md"
