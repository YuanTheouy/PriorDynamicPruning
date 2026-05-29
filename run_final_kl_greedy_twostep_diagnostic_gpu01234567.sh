#!/usr/bin/env bash
set -euo pipefail

# Cheap diagnostic for combination effects before building full 7-layer greedy labels.
# For each sample, compare router top choices with final-KL greedy step 1 and step 2.
export REPO_DIR=/workspace/PriorDynamicPruning
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export BASE_PORT="${OPAL_BASE_PORT:-54000}"
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
export SKIP_RATE=0.25
export PROTECTED_HEAD=4
export PROTECTED_TAIL=2
export DIAG_MAX_SAMPLES="${OPAL_DIAG_MAX_SAMPLES:-500}"
export DIAG_SAMPLE_STRATEGY="${OPAL_DIAG_SAMPLE_STRATEGY:-random}"
export DIAG_SAMPLE_SEED="${OPAL_DIAG_SAMPLE_SEED:-42}"
export DIAG_BATCH_SIZE=1
export RUN_GROUP="final_kl_greedy_twostep_diag_m${DIAG_MAX_SAMPLES}_seed${SEED}"
export DIAG_DIR=/workspace/PriorDynamicPruning/results/opal_greedy_diagnostics

# Defaults point at the large one-layer Delta_NLL routers, because this diagnostic asks
# whether their top-ranked skip layers already diverge from true conditional final-KL choices.
export RAW_CKPT="${RAW_CKPT:-/workspace/PriorDynamicPruning/policy_ckpts/final_opal_attn_large/final_opal_attn_large_delta_m10000_seed42/raw_input_risk/risk_router.pt}"
export ATTN_CKPT="${ATTN_CKPT:-/workspace/PriorDynamicPruning/policy_ckpts/final_opal_attn_large/final_opal_attn_large_delta_m10000_seed42/prefix_hk_raw_attn/risk_router.pt}"

cd "$REPO_DIR"
mkdir -p "$DIAG_DIR"

echo "=== Final-KL two-step greedy/router diagnostic config ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DIAG_MAX_SAMPLES=${DIAG_MAX_SAMPLES}"
echo "DIAG_SAMPLE_STRATEGY=${DIAG_SAMPLE_STRATEGY}"
echo "DIAG_SAMPLE_SEED=${DIAG_SAMPLE_SEED}"
echo "SKIP_RATE=${SKIP_RATE}"
echo "PROTECTED_HEAD=${PROTECTED_HEAD}"
echo "PROTECTED_TAIL=${PROTECTED_TAIL}"
echo "RAW_CKPT=${RAW_CKPT}"
echo "ATTN_CKPT=${ATTN_CKPT}"

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
    port_log="$(mktemp "/tmp/opal_twostep_diag_${port}_XXXX.log")"
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
  local ckpt="$2"
  if [ ! -s "$ckpt" ]; then
    echo "Missing checkpoint for ${variant}: ${ckpt}" >&2
    return 1
  fi
  local prefix="${RUN_GROUP}_${variant}"
  local output_jsonl="${DIAG_DIR}/${prefix}.jsonl"
  local summary_json="${DIAG_DIR}/${prefix}.summary.json"
  echo "=== Diagnose ${variant} ==="
  run_accelerate ./diagnose_greedy_router_alignment.py \
    --teacher_model "$MODEL_PATH" \
    --risk_router_ckpt "$ckpt" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --skip_rate "$SKIP_RATE" \
    --protected_head "$PROTECTED_HEAD" \
    --protected_tail "$PROTECTED_TAIL" \
    --max_samples "$DIAG_MAX_SAMPLES" \
    --sample_strategy "$DIAG_SAMPLE_STRATEGY" \
    --sample_seed "$DIAG_SAMPLE_SEED" \
    --output_jsonl "$output_jsonl" \
    --summary_json "$summary_json" \
    --batch_size "$DIAG_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$SEED"
}

run_diag raw_input_risk "$RAW_CKPT"
run_diag prefix_hk_raw_attn "$ATTN_CKPT"

echo "=== Raw summary ==="
cat "${DIAG_DIR}/${RUN_GROUP}_raw_input_risk.summary.json"
echo
echo "=== Prefix Hk raw attention summary ==="
cat "${DIAG_DIR}/${RUN_GROUP}_prefix_hk_raw_attn.summary.json"
echo
echo "=== Done final-KL two-step greedy/router diagnostic ==="
