#!/usr/bin/env bash
set -euo pipefail

# Exact K-subset CE experiment over final-KL greedy skip-set labels.
# Router inputs and inference stay unchanged: one-shot risk prediction, then skip the K lowest-risk allowed layers.
export REPO_DIR=/workspace/PriorDynamicPruning
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export BASE_PORT="${OPAL_BASE_PORT:-56000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="/workspace/PriorDynamicPruning/transformers/src:/workspace/PriorDynamicPruning:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

export MODEL_PATH=/workspace/ckpts/MiniOneRec/Office_ckpt
export CATEGORY=Office_Products
export TRAIN_FILE=/workspace/PriorDynamicPruning/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv
export TEST_FILE=/workspace/PriorDynamicPruning/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv
export INFO_FILE=/workspace/PriorDynamicPruning/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt

export OUTPUT_DIR=/workspace/PriorDynamicPruning/results/planrec_experiments
export LABEL_DIR=/workspace/PriorDynamicPruning/results/opal_greedy_set_labels
export BCE_CKPT_ROOT=/workspace/PriorDynamicPruning/policy_ckpts/final_kl_greedy_set
export EXACT_CKPT_ROOT=/workspace/PriorDynamicPruning/policy_ckpts/final_kl_greedy_set_exact_k_ce
export DIAG_DIR=/workspace/PriorDynamicPruning/results/opal_greedy_set_diagnostics

export SEED="${OPAL_SEED:-42}"
export PRECISION=bf16
export PREFIX_DEPTH=4
export SKIP_RATE=0.25
export TOP_K_LAYERS=21
export GREEDY_OBJECTIVE="${OPAL_GREEDY_OBJECTIVE:-final_KL}"
export TOP_K_ITEMS=50
export MAX_NEW_TOKENS=256
export LABEL_MAX_SAMPLES="${OPAL_LABEL_MAX_SAMPLES:-2000}"
export LABEL_BATCH_SIZE=1
export TRAIN_BATCH_SIZE="${OPAL_TRAIN_BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE=1
export EPOCHS="${OPAL_EPOCHS:-40}"
export LR="${OPAL_LR:-1e-4}"
export MAX_GRAD_NORM="${OPAL_MAX_GRAD_NORM:-1.0}"
export ROUTER_DIM="${OPAL_ROUTER_DIM:-256}"
export ROUTER_HEADS="${OPAL_ROUTER_HEADS:-4}"
export PROTECTED_HEAD=4
export PROTECTED_TAIL=2
export WARMUP_BATCHES=0
export TIMED_BATCHES=0
export EVAL_MAX_BATCHES="${OPAL_EVAL_MAX_BATCHES:-500}"
export OBJECTIVE_TAG
OBJECTIVE_TAG="$(printf "%s" "$GREEDY_OBJECTIVE" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
export BCE_RUN_GROUP="${OBJECTIVE_TAG}_greedy_set_m${LABEL_MAX_SAMPLES}_seed${SEED}"
export RUN_GROUP="${OBJECTIVE_TAG}_greedy_set_exact_k_ce_m${LABEL_MAX_SAMPLES}_seed${SEED}"
export SUMMARY_TABLE="summary_${RUN_GROUP}.csv"
export LABEL_FILE="${LABEL_DIR}/${CATEGORY}_${OBJECTIVE_TAG}_greedy_set_m${LABEL_MAX_SAMPLES}_seed${SEED}_head${PROTECTED_HEAD}_tail${PROTECTED_TAIL}.jsonl"

cd "$REPO_DIR"
mkdir -p "$LABEL_DIR" "$BCE_CKPT_ROOT" "$EXACT_CKPT_ROOT" "$OUTPUT_DIR" "$DIAG_DIR"

echo "=== Exact K-subset CE config ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LABEL_MAX_SAMPLES=${LABEL_MAX_SAMPLES}"
echo "SEED=${SEED}"
echo "SKIP_RATE=${SKIP_RATE}"
echo "GREEDY_OBJECTIVE=${GREEDY_OBJECTIVE}"
echo "PROTECTED_HEAD=${PROTECTED_HEAD}"
echo "PROTECTED_TAIL=${PROTECTED_TAIL}"
echo "RUN_GROUP=${RUN_GROUP}"

run_accelerate() {
  local port="$NEXT_PORT"
  while true; do
    while ! python3 -c 'import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
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
    local port_log
    port_log="$(mktemp "/tmp/opal_exact_k_ce_${port}_XXXX.log")"
    set +e
    accelerate launch \
      --num_processes "$NUM_GPUS" \
      --num_machines 1 \
      --main_process_port "$port" \
      --mixed_precision "$PRECISION" \
      --dynamo_backend no \
      "$@" 2>&1 | tee "$port_log"
    local status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -eq 0 ]; then
      rm -f "$port_log"
      return 0
    fi
    if grep -q "EADDRINUSE" "$port_log"; then
      echo "=== Port ${port} became occupied during launch; retry with next port ==="
      rm -f "$port_log"
      port="$((port + 1))"
      continue
    fi
    rm -f "$port_log"
    return "$status"
  done
}

final_json_exists() {
  local run_name="$1"
  test -s "${OUTPUT_DIR}/raw_json/${run_name}.json"
}

checkpoint_usable() {
  local ckpt="$1"
  local metrics
  metrics="$(dirname "$ckpt")/training_metrics.json"
  if [ ! -s "$ckpt" ]; then
    return 1
  fi
  python3 - "$metrics" <<'PY'
import json
import math
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    sys.exit(0)

def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True

try:
    data = json.load(open(path))
except Exception:
    sys.exit(1)
sys.exit(0 if finite_tree(data) else 1)
PY
}

if [ -s "$LABEL_FILE" ] && [ "$(wc -l < "$LABEL_FILE")" -ge "$LABEL_MAX_SAMPLES" ]; then
  echo "=== Reuse final-KL greedy labels: ${LABEL_FILE} ==="
else
  echo "=== Build final-KL greedy set labels ==="
  run_accelerate ./build_final_kl_greedy_set_labels.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --protected_head "$PROTECTED_HEAD" \
    --protected_tail "$PROTECTED_TAIL" \
    --objective "$GREEDY_OBJECTIVE" \
    --max_samples "$LABEL_MAX_SAMPLES" \
    --sample_strategy random \
    --sample_seed "$SEED" \
    --output "$LABEL_FILE" \
    --batch_size "$LABEL_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$SEED"
fi

train_bce_router_if_needed() {
  local variant="$1"
  local router_input="$2"
  local risk_pooling="${3:-mean}"
  local out_dir="${BCE_CKPT_ROOT}/${BCE_RUN_GROUP}/${variant}"
  local ckpt="${out_dir}/risk_router.pt"
  if checkpoint_usable "$ckpt"; then
    echo "=== Reuse BCE checkpoint: ${ckpt} ==="
    return
  fi
  echo "=== Train ${variant} with BCE skip-set supervision ==="
  run_accelerate ./train_opal_risk_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$LABEL_FILE" \
    --router_input "$router_input" \
    --risk_pooling "$risk_pooling" \
    --prefix_depth "$PREFIX_DEPTH" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --ranking_loss_weight 0 \
    --skip_set_loss_weight 0 \
    --set_loss_type bce \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --huber_beta 1.0 \
    --router_dim "$ROUTER_DIM" \
    --router_heads "$ROUTER_HEADS" \
    --output_dir "$out_dir" \
    --precision "$PRECISION" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --loss_log_interval 10 \
    --seed "$SEED"
}

train_exact_router_if_needed() {
  local variant="$1"
  local router_input="$2"
  local risk_pooling="${3:-mean}"
  local out_dir="${EXACT_CKPT_ROOT}/${RUN_GROUP}/${variant}"
  local ckpt="${out_dir}/risk_router.pt"
  if checkpoint_usable "$ckpt"; then
    echo "=== Reuse exact-k CE checkpoint: ${ckpt} ==="
    return
  fi
  echo "=== Train ${variant} with Exact K-subset CE ==="
  run_accelerate ./train_opal_risk_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$LABEL_FILE" \
    --router_input "$router_input" \
    --risk_pooling "$risk_pooling" \
    --prefix_depth "$PREFIX_DEPTH" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --ranking_loss_weight 0 \
    --skip_set_loss_weight 0 \
    --set_loss_type exact_k_ce \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --huber_beta 1.0 \
    --router_dim "$ROUTER_DIM" \
    --router_heads "$ROUTER_HEADS" \
    --output_dir "$out_dir" \
    --precision "$PRECISION" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --loss_log_interval 10 \
    --seed "$SEED"
}

eval_ckpt_if_needed() {
  local variant="$1"
  local method="$2"
  local ckpt="$3"
  local run_name="${CATEGORY}_${RUN_GROUP}_${variant}_skip${SKIP_RATE}"
  if [ ! -s "$ckpt" ]; then
    echo "Missing checkpoint for ${variant}: ${ckpt}" >&2
    return 1
  fi
  if final_json_exists "$run_name"; then
    echo "=== Reuse eval JSON: ${run_name} ==="
    return
  fi
  echo "=== Eval ${variant} ==="
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
    --seed "$SEED" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$EVAL_MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name"
}

overlap_diag_if_needed() {
  local variant="$1"
  local ckpt="$2"
  local prefix="${RUN_GROUP}_${variant}_train_overlap"
  local output_jsonl="${DIAG_DIR}/${prefix}.jsonl"
  local summary_json="${DIAG_DIR}/${prefix}.summary.json"
  if [ -s "$summary_json" ]; then
    echo "=== Reuse overlap summary: ${summary_json} ==="
    return
  fi
  echo "=== Train-label overlap diagnostic: ${variant} ==="
  run_accelerate ./diagnose_greedy_set_overlap.py \
    --teacher_model "$MODEL_PATH" \
    --risk_router_ckpt "$ckpt" \
    --risk_label_file "$LABEL_FILE" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --max_samples 0 \
    --sample_strategy first \
    --sample_seed "$SEED" \
    --output_jsonl "$output_jsonl" \
    --summary_json "$summary_json" \
    --batch_size 1 \
    --precision "$PRECISION" \
    --seed "$SEED"
}

train_bce_router_if_needed raw_input_risk raw_embedding mean
train_bce_router_if_needed prefix_hk_raw_attn prefix_hk_raw_attn mean

train_exact_router_if_needed raw_input_risk_exact_k_ce raw_embedding mean
train_exact_router_if_needed prefix_hk_last_exact_k_ce prefix_hk last
train_exact_router_if_needed prefix_hk_raw_attn_exact_k_ce prefix_hk_raw_attn mean

eval_ckpt_if_needed raw_input_risk_bce raw_input_risk "${BCE_CKPT_ROOT}/${BCE_RUN_GROUP}/raw_input_risk/risk_router.pt"
eval_ckpt_if_needed prefix_hk_raw_attn_bce opal_risk "${BCE_CKPT_ROOT}/${BCE_RUN_GROUP}/prefix_hk_raw_attn/risk_router.pt"
eval_ckpt_if_needed raw_input_risk_exact_k_ce raw_input_risk "${EXACT_CKPT_ROOT}/${RUN_GROUP}/raw_input_risk_exact_k_ce/risk_router.pt"
eval_ckpt_if_needed prefix_hk_last_exact_k_ce opal_risk "${EXACT_CKPT_ROOT}/${RUN_GROUP}/prefix_hk_last_exact_k_ce/risk_router.pt"
eval_ckpt_if_needed prefix_hk_raw_attn_exact_k_ce opal_risk "${EXACT_CKPT_ROOT}/${RUN_GROUP}/prefix_hk_raw_attn_exact_k_ce/risk_router.pt"

overlap_diag_if_needed raw_input_risk_bce "${BCE_CKPT_ROOT}/${BCE_RUN_GROUP}/raw_input_risk/risk_router.pt"
overlap_diag_if_needed prefix_hk_raw_attn_bce "${BCE_CKPT_ROOT}/${BCE_RUN_GROUP}/prefix_hk_raw_attn/risk_router.pt"
overlap_diag_if_needed raw_input_risk_exact_k_ce "${EXACT_CKPT_ROOT}/${RUN_GROUP}/raw_input_risk_exact_k_ce/risk_router.pt"
overlap_diag_if_needed prefix_hk_last_exact_k_ce "${EXACT_CKPT_ROOT}/${RUN_GROUP}/prefix_hk_last_exact_k_ce/risk_router.pt"
overlap_diag_if_needed prefix_hk_raw_attn_exact_k_ce "${EXACT_CKPT_ROOT}/${RUN_GROUP}/prefix_hk_raw_attn_exact_k_ce/risk_router.pt"

echo "=== Summarize Exact K-subset CE experiment ==="
python3 summarize_planrec_results.py \
  --output_dir "$OUTPUT_DIR" \
  --table_name "$SUMMARY_TABLE" \
  --run_name_contains "$RUN_GROUP" \
  --exclude_debug_sanity

cat "$OUTPUT_DIR/tables/quality_retention.md"

echo "=== Compact train-label overlap summaries ==="
python3 - <<'PY'
import json, os
diag_dir = os.environ["DIAG_DIR"]
run_group = os.environ["RUN_GROUP"]
for variant in [
    "raw_input_risk_bce",
    "prefix_hk_raw_attn_bce",
    "raw_input_risk_exact_k_ce",
    "prefix_hk_last_exact_k_ce",
    "prefix_hk_raw_attn_exact_k_ce",
]:
    path = f"{diag_dir}/{run_group}_{variant}_train_overlap.summary.json"
    if not os.path.exists(path):
        continue
    s = json.load(open(path))
    print("\n===", variant, "===")
    for key in [
        "num_samples",
        "exact_match_rate",
        "mean_overlap_ratio",
        "mean_overlap_count",
        "mean_hamming_count",
        "mean_pairwise_order_accuracy",
        "unique_predicted_masks",
        "unique_label_masks",
    ]:
        print(key, s.get(key))
PY

echo "=== Done Exact K-subset CE experiment ==="
