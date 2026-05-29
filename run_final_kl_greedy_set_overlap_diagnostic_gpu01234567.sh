#!/usr/bin/env bash
set -euo pipefail

# Train-label diagnostic: compare router top-7 skipped layers with final-KL greedy skip-set labels.
export REPO_DIR=/workspace/PriorDynamicPruning
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export BASE_PORT="${OPAL_BASE_PORT:-55000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="/workspace/PriorDynamicPruning/transformers/src:/workspace/PriorDynamicPruning:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

export MODEL_PATH=/workspace/ckpts/MiniOneRec/Office_ckpt
export CATEGORY=Office_Products
export TRAIN_FILE=/workspace/PriorDynamicPruning/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv
export INFO_FILE=/workspace/PriorDynamicPruning/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt

export SEED=42
export PRECISION=bf16
export PREFIX_DEPTH=4
export PROTECTED_HEAD=4
export PROTECTED_TAIL=2
export LABEL_MAX_SAMPLES="${OPAL_LABEL_MAX_SAMPLES:-2000}"
export DIAG_MAX_SAMPLES="${OPAL_DIAG_MAX_SAMPLES:-0}"
export DIAG_BATCH_SIZE=1
export RUN_GROUP="final_kl_greedy_set_m${LABEL_MAX_SAMPLES}_seed${SEED}"
export LABEL_FILE="/workspace/PriorDynamicPruning/results/opal_greedy_set_labels/${CATEGORY}_final_KL_greedy_set_m${LABEL_MAX_SAMPLES}_seed${SEED}_head${PROTECTED_HEAD}_tail${PROTECTED_TAIL}.jsonl"
export CKPT_ROOT="/workspace/PriorDynamicPruning/policy_ckpts/final_kl_greedy_set/${RUN_GROUP}"
export DIAG_DIR=/workspace/PriorDynamicPruning/results/opal_greedy_set_diagnostics

cd "$REPO_DIR"
mkdir -p "$DIAG_DIR"

echo "=== Final-KL greedy set overlap diagnostic config ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LABEL_FILE=${LABEL_FILE}"
echo "CKPT_ROOT=${CKPT_ROOT}"
echo "DIAG_MAX_SAMPLES=${DIAG_MAX_SAMPLES} (0 means all label rows)"

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
    port_log="$(mktemp "/tmp/opal_set_overlap_${port}_XXXX.log")"
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

run_diag() {
  local variant="$1"
  local ckpt="${CKPT_ROOT}/${variant}/risk_router.pt"
  if [ ! -s "$ckpt" ]; then
    echo "Missing checkpoint for ${variant}: ${ckpt}" >&2
    return 1
  fi
  if [ ! -s "$LABEL_FILE" ]; then
    echo "Missing greedy-set label file: ${LABEL_FILE}" >&2
    return 1
  fi
  local prefix="${RUN_GROUP}_${variant}_train_overlap"
  local output_jsonl="${DIAG_DIR}/${prefix}.jsonl"
  local summary_json="${DIAG_DIR}/${prefix}.summary.json"
  echo "=== Diagnose train-label overlap: ${variant} ==="
  run_accelerate ./diagnose_greedy_set_overlap.py \
    --teacher_model "$MODEL_PATH" \
    --risk_router_ckpt "$ckpt" \
    --risk_label_file "$LABEL_FILE" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --max_samples "$DIAG_MAX_SAMPLES" \
    --sample_strategy first \
    --sample_seed "$SEED" \
    --output_jsonl "$output_jsonl" \
    --summary_json "$summary_json" \
    --batch_size "$DIAG_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$SEED"
}

run_diag raw_input_risk
run_diag prefix_hk_raw_attn

echo "=== Raw train-overlap summary ==="
cat "${DIAG_DIR}/${RUN_GROUP}_raw_input_risk_train_overlap.summary.json"
echo
echo "=== Prefix Hk raw attention train-overlap summary ==="
cat "${DIAG_DIR}/${RUN_GROUP}_prefix_hk_raw_attn_train_overlap.summary.json"
echo
echo "=== Done final-KL greedy set overlap diagnostic ==="
