#!/usr/bin/env bash
set -euo pipefail

# OPAL-SetBCE cross-setting runner.
# Runs only:
#   prefix_hk_raw_attn + final Delta_NLL greedy set labels + BCE set supervision.
# It intentionally does not run raw, PrefixLast, Exact-K CE, or OPAL-SetAttn v1.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${OPAL_BASE_PORT:-56600}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/planrec_experiments}"
export LABEL_DIR="${LABEL_DIR:-${REPO_DIR}/results/opal_greedy_set_labels}"
export CKPT_ROOT="${CKPT_ROOT:-${REPO_DIR}/policy_ckpts/opal_setbce_cross_setting}"
export DIAG_DIR="${DIAG_DIR:-${REPO_DIR}/results/opal_greedy_set_diagnostics}"

export OPAL_TASK="${OPAL_TASK:-industrial25}"
export SEED="${OPAL_SEED:-42}"
export PRECISION="${PRECISION:-bf16}"
export PREFIX_DEPTH="${PREFIX_DEPTH:-4}"
export TOP_K_ITEMS="${TOP_K_ITEMS:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export LABEL_MAX_SAMPLES="${OPAL_LABEL_MAX_SAMPLES:-2000}"
export LABEL_BATCH_SIZE="${OPAL_LABEL_BATCH_SIZE:-1}"
export TRAIN_BATCH_SIZE="${OPAL_TRAIN_BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${OPAL_EVAL_BATCH_SIZE:-1}"
export EPOCHS="${OPAL_EPOCHS:-40}"
export LR="${OPAL_LR:-1e-4}"
export MAX_GRAD_NORM="${OPAL_MAX_GRAD_NORM:-1.0}"
export ROUTER_DIM="${OPAL_ROUTER_DIM:-256}"
export ROUTER_HEADS="${OPAL_ROUTER_HEADS:-4}"
export PROTECTED_HEAD="${PROTECTED_HEAD:-4}"
export PROTECTED_TAIL="${PROTECTED_TAIL:-2}"
export WARMUP_BATCHES="${WARMUP_BATCHES:-0}"
export TIMED_BATCHES="${TIMED_BATCHES:-0}"
export EVAL_MAX_BATCHES="${OPAL_EVAL_MAX_BATCHES:-500}"
export GREEDY_OBJECTIVE=Delta_NLL
export OBJECTIVE_TAG=delta_nll

skip_tag() {
  printf "%s" "$1" | tr "." "p"
}

configure_task() {
  case "$OPAL_TASK" in
    office25)
      export MODEL_PATH="${OFFICE_MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}"
      export CATEGORY="Office_Products"
      export TRAIN_FILE="${REPO_DIR}/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv"
      export TEST_FILE="${REPO_DIR}/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv"
      export INFO_FILE="${REPO_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"
      export SKIP_RATE="${OPAL_SKIP_RATE:-0.25}"
      export TOP_K_LAYERS="${OPAL_TOP_K_LAYERS:-21}"
      ;;
    industrial25)
      export MODEL_PATH="${INDUSTRIAL_MODEL_PATH:-/workspace/ckpts/MiniOneRec/Industrial_ckpt}"
      export CATEGORY="Industrial_and_Scientific"
      export TRAIN_FILE="${REPO_DIR}/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
      export TEST_FILE="${REPO_DIR}/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
      export INFO_FILE="${REPO_DIR}/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
      export SKIP_RATE="${OPAL_SKIP_RATE:-0.25}"
      export TOP_K_LAYERS="${OPAL_TOP_K_LAYERS:-21}"
      ;;
    office36)
      export MODEL_PATH="${OFFICE_MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}"
      export CATEGORY="Office_Products"
      export TRAIN_FILE="${REPO_DIR}/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv"
      export TEST_FILE="${REPO_DIR}/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv"
      export INFO_FILE="${REPO_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"
      export SKIP_RATE="${OPAL_SKIP_RATE:-0.357}"
      export TOP_K_LAYERS="${OPAL_TOP_K_LAYERS:-18}"
      ;;
    *)
      echo "Unknown OPAL_TASK=${OPAL_TASK}; expected office25, industrial25, or office36." >&2
      exit 1
      ;;
  esac
}

configure_task
export SKIP_TAG
SKIP_TAG="$(skip_tag "$SKIP_RATE")"
export RUN_GROUP="opal_setbce_${OPAL_TASK}_${OBJECTIVE_TAG}_greedy_set_m${LABEL_MAX_SAMPLES}_seed${SEED}"
export SUMMARY_TABLE="summary_${RUN_GROUP}.csv"
export LABEL_FILE="${LABEL_DIR}/${CATEGORY}_${OPAL_TASK}_${OBJECTIVE_TAG}_greedy_set_m${LABEL_MAX_SAMPLES}_seed${SEED}_k${TOP_K_LAYERS}_skip${SKIP_TAG}_head${PROTECTED_HEAD}_tail${PROTECTED_TAIL}.jsonl"
export ROUTER_DIR="${CKPT_ROOT}/${RUN_GROUP}/prefix_hk_raw_attn_bce"
export ROUTER_CKPT="${ROUTER_DIR}/risk_router.pt"

cd "$REPO_DIR"
mkdir -p "$LABEL_DIR" "$CKPT_ROOT" "$OUTPUT_DIR" "$DIAG_DIR"

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
    port_log="$(mktemp "/tmp/opal_setbce_${port}_XXXX.log")"
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
    echo "=== Failed command log retained at ${port_log} ===" >&2
    return "$status"
  done
}

line_count_at_least() {
  local path="$1"
  local min_rows="$2"
  test -s "$path" && [ "$(wc -l < "$path")" -ge "$min_rows" ]
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

final_json_exists() {
  local run_name="$1"
  test -s "${OUTPUT_DIR}/raw_json/${run_name}.json"
}

echo "=== OPAL-SetBCE cross-setting config ==="
echo "REPO_DIR=${REPO_DIR}"
echo "OPAL_TASK=${OPAL_TASK}"
echo "CATEGORY=${CATEGORY}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "TEST_FILE=${TEST_FILE}"
echo "INFO_FILE=${INFO_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "SEED=${SEED}"
echo "SKIP_RATE=${SKIP_RATE}"
echo "TOP_K_LAYERS=${TOP_K_LAYERS}"
echo "LABEL_MAX_SAMPLES=${LABEL_MAX_SAMPLES}"
echo "EPOCHS=${EPOCHS}"
echo "EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES}"
echo "LABEL_FILE=${LABEL_FILE}"
echo "ROUTER_CKPT=${ROUTER_CKPT}"
echo "RUN_GROUP=${RUN_GROUP}"

if line_count_at_least "$LABEL_FILE" "$LABEL_MAX_SAMPLES"; then
  echo "=== Reuse Delta_NLL greedy set labels: ${LABEL_FILE} ==="
else
  echo "=== Build Delta_NLL greedy set labels ==="
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

if checkpoint_usable "$ROUTER_CKPT"; then
  echo "=== Reuse OPAL-SetBCE checkpoint: ${ROUTER_CKPT} ==="
else
  echo "=== Train OPAL-SetBCE prefix_hk_raw_attn + BCE ==="
  run_accelerate ./train_opal_risk_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$LABEL_FILE" \
    --router_input prefix_hk_raw_attn \
    --prefix_depth "$PREFIX_DEPTH" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --ranking_loss_weight 0 \
    --skip_set_loss_weight 0 \
    --set_loss_type bce \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --router_dim "$ROUTER_DIM" \
    --router_heads "$ROUTER_HEADS" \
    --output_dir "$ROUTER_DIR" \
    --precision "$PRECISION" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --loss_log_interval 10 \
    --seed "$SEED"
fi

EVAL_RUN_NAME="${CATEGORY}_${RUN_GROUP}_prefix_hk_raw_attn_bce_skip${SKIP_RATE}"
if final_json_exists "$EVAL_RUN_NAME"; then
  echo "=== Reuse eval JSON: ${EVAL_RUN_NAME} ==="
else
  echo "=== Eval OPAL-SetBCE ==="
  run_accelerate ./eval_planrec_opal.py \
    --method opal_risk \
    --risk_router_ckpt "$ROUTER_CKPT" \
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
    --run_name "$EVAL_RUN_NAME"
fi

OVERLAP_PREFIX="${RUN_GROUP}_prefix_hk_raw_attn_bce_train_overlap"
OVERLAP_JSONL="${DIAG_DIR}/${OVERLAP_PREFIX}.jsonl"
OVERLAP_SUMMARY="${DIAG_DIR}/${OVERLAP_PREFIX}.summary.json"
if [ -s "$OVERLAP_SUMMARY" ]; then
  echo "=== Reuse overlap summary: ${OVERLAP_SUMMARY} ==="
else
  echo "=== Train-label overlap diagnostic: OPAL-SetBCE ==="
  run_accelerate ./diagnose_greedy_set_overlap.py \
    --teacher_model "$MODEL_PATH" \
    --risk_router_ckpt "$ROUTER_CKPT" \
    --risk_label_file "$LABEL_FILE" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --max_samples 0 \
    --sample_strategy first \
    --sample_seed "$SEED" \
    --output_jsonl "$OVERLAP_JSONL" \
    --summary_json "$OVERLAP_SUMMARY" \
    --batch_size 1 \
    --precision "$PRECISION" \
    --seed "$SEED"
fi

echo "=== Summarize OPAL-SetBCE cross-setting ==="
python3 summarize_planrec_results.py \
  --output_dir "$OUTPUT_DIR" \
  --table_name "$SUMMARY_TABLE" \
  --run_name_contains "$RUN_GROUP" \
  --exclude_debug_sanity

echo "=== Compact result rows ==="
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
summary_table = output_dir / "tables" / os.environ["SUMMARY_TABLE"]
if not summary_table.exists():
    print(f"missing summary table: {summary_table}")
else:
    for row in csv.DictReader(summary_table.open()):
        print(json.dumps({
            "task": os.environ["OPAL_TASK"],
            "run_name": row.get("run_name"),
            "Delta_NLL": row.get("Delta_NLL"),
            "Delta_PPL": row.get("Delta_PPL"),
            "KL_full_to_skip": row.get("KL_full_to_skip"),
            "NDCG@10": row.get("NDCG@10"),
            "retention_NDCG@10": row.get("retention_NDCG@10"),
            "risk_router_input": row.get("risk_router_input"),
        }, ensure_ascii=False))
PY

echo "=== Overlap summary ==="
cat "$OVERLAP_SUMMARY"

echo "=== Training metrics first/best/last ==="
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["ROUTER_DIR"]) / "training_metrics.json"
data = json.load(path.open())
history = data.get("history", [])
if not history:
    print(json.dumps({"error": "empty training history", "path": str(path)}, ensure_ascii=False))
else:
    first = history[0]
    last = history[-1]
    best = min(history, key=lambda row: float(row.get("loss", float("inf"))))
    print(json.dumps({
        "task": os.environ["OPAL_TASK"],
        "router_input": data.get("metadata", {}).get("router_input"),
        "set_loss_type": data.get("metadata", {}).get("set_loss_type"),
        "epochs": len(history),
        "first": {
            "epoch": first.get("epoch"),
            "loss": first.get("loss"),
            "skip_set": first.get("skip_set"),
        },
        "best": {
            "epoch": best.get("epoch"),
            "loss": best.get("loss"),
            "skip_set": best.get("skip_set"),
        },
        "last": {
            "epoch": last.get("epoch"),
            "loss": last.get("loss"),
            "skip_set": last.get("skip_set"),
        },
    }, ensure_ascii=False))
PY

echo "=== Summary csv rows ==="
cat "${OUTPUT_DIR}/tables/${SUMMARY_TABLE}"

echo "=== Done: OPAL-SetBCE ${OPAL_TASK} seed${SEED} ==="
