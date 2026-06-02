#!/usr/bin/env bash
set -euo pipefail

# WikiText-2 OPAL validation-checkpoint selection.
# This only trains/evaluates OPAL-SetBCE checkpoints. It does not rerun Full/static/Raw.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${WIKITEXT_BASE_PORT:-58600}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${HOME}/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export WIKITEXT_MODEL_PATH="${WIKITEXT_MODEL_PATH:-}"
export WIKITEXT_LABEL_SAMPLES="${WIKITEXT_LABEL_SAMPLES:-2000}"
export WIKITEXT_EVAL_WINDOWS="${WIKITEXT_EVAL_WINDOWS:-512}"
export WIKITEXT_EPOCHS="${WIKITEXT_EPOCHS:-40}"
export WIKITEXT_SEQ_LEN="${WIKITEXT_SEQ_LEN:-1024}"
export WIKITEXT_ROUTER_PREFIX_TOKENS="${WIKITEXT_ROUTER_PREFIX_TOKENS:-256}"
export WIKITEXT_PREFIX_DEPTH="${WIKITEXT_PREFIX_DEPTH:-4}"
export WIKITEXT_SEED="${WIKITEXT_SEED:-42}"
export WIKITEXT_SKIP_RATE="${WIKITEXT_SKIP_RATE:-0.25}"
export WIKITEXT_SKIP_COUNT="${WIKITEXT_SKIP_COUNT:-0}"
export WIKITEXT_PROTECTED_HEAD="${WIKITEXT_PROTECTED_HEAD:-4}"
export WIKITEXT_PROTECTED_TAIL="${WIKITEXT_PROTECTED_TAIL:-2}"
export WIKITEXT_PRECISION="${WIKITEXT_PRECISION:-bf16}"
export WIKITEXT_TRAIN_BATCH_SIZE="${WIKITEXT_TRAIN_BATCH_SIZE:-4}"
export WIKITEXT_EVAL_BATCH_SIZE="${WIKITEXT_EVAL_BATCH_SIZE:-1}"
export WIKITEXT_LR="${WIKITEXT_LR:-1e-4}"
export WIKITEXT_ROUTER_DIM="${WIKITEXT_ROUTER_DIM:-256}"
export WIKITEXT_ROUTER_HEADS="${WIKITEXT_ROUTER_HEADS:-4}"
export WIKITEXT_MAX_GRAD_NORM="${WIKITEXT_MAX_GRAD_NORM:-1.0}"
export WIKITEXT_MASK_IMPL_TAG="${WIKITEXT_MASK_IMPL_TAG:-maskcfg}"
export WIKITEXT_VALCKPT_PARALLEL_WORKERS="${WIKITEXT_VALCKPT_PARALLEL_WORKERS:-$NUM_GPUS}"
export WIKITEXT_VALCKPT_OMP_NUM_THREADS="${WIKITEXT_VALCKPT_OMP_NUM_THREADS:-2}"
export WIKITEXT_DATASET_CACHE_DIR="${WIKITEXT_DATASET_CACHE_DIR:-}"
if [ -z "${WIKITEXT_DATASET_DISK_PATH:-}" ] && [ -d "/workspace/datasets/wikitext/wikitext-2-raw-v1" ]; then
  export WIKITEXT_DATASET_DISK_PATH="/workspace/datasets/wikitext/wikitext-2-raw-v1"
else
  export WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-}"
fi

if [ -z "$WIKITEXT_MODEL_PATH" ]; then
  echo "WIKITEXT_MODEL_PATH is required." >&2
  exit 2
fi

cd "$REPO_DIR"

model_tag="$(basename "$WIKITEXT_MODEL_PATH" | tr ' ./:' '____')"
skip_tag="$(printf "%s" "$WIKITEXT_SKIP_RATE" | tr "." "p")"
export WIKITEXT_LABEL_RUN_ID="${WIKITEXT_LABEL_RUN_ID:-wikitext2_${model_tag}_${WIKITEXT_MASK_IMPL_TAG}_seq${WIKITEXT_SEQ_LEN}_pref${WIKITEXT_ROUTER_PREFIX_TOKENS}_m${WIKITEXT_LABEL_SAMPLES}_seed${WIKITEXT_SEED}_skip${skip_tag}}"
prefix_depth_tag=""
if [ "$WIKITEXT_PREFIX_DEPTH" != "4" ]; then
  prefix_depth_tag="_h${WIKITEXT_PREFIX_DEPTH}"
fi
export WIKITEXT_RUN_ID="${WIKITEXT_RUN_ID:-${WIKITEXT_LABEL_RUN_ID}${prefix_depth_tag}_valckpt}"
export WIKITEXT_RESULT_ROOT="${WIKITEXT_RESULT_ROOT:-${REPO_DIR}/results/wikitext2_public_lm_sanity}"
export WIKITEXT_LABEL_FILE="${WIKITEXT_RESULT_ROOT}/labels/${WIKITEXT_LABEL_RUN_ID}_delta_nll_greedy_set_labels.jsonl"
export WIKITEXT_VALCKPT_ROOT="${WIKITEXT_VALCKPT_ROOT:-${REPO_DIR}/policy_ckpts/wikitext2_public_lm_val_ckpt/${WIKITEXT_RUN_ID}}"
export WIKITEXT_VALCKPT_DIR="${WIKITEXT_VALCKPT_ROOT}/prefix_hk_raw_attn_bce"
export WIKITEXT_EPOCH_CKPT_DIR="${WIKITEXT_VALCKPT_DIR}/epoch_checkpoints"
export WIKITEXT_VAL_METRIC_DIR="${WIKITEXT_RESULT_ROOT}/val_ckpt_metrics/${WIKITEXT_RUN_ID}"
export WIKITEXT_BEST_JSON="${WIKITEXT_VAL_METRIC_DIR}/best_validation_checkpoint.json"

mkdir -p "$WIKITEXT_EPOCH_CKPT_DIR" "$WIKITEXT_VAL_METRIC_DIR"

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
    port_log="$(mktemp "/tmp/wikitext2_valckpt_${port}_XXXX.log")"
    set +e
    accelerate launch \
      --num_processes "$NUM_GPUS" \
      --num_machines 1 \
      --main_process_port "$port" \
      --mixed_precision "$WIKITEXT_PRECISION" \
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

json_usable() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, math, os, sys
path = sys.argv[1]
if not os.path.exists(path) or os.path.getsize(path) == 0:
    sys.exit(1)
try:
    payload = json.load(open(path))
except Exception:
    sys.exit(1)
sys.exit(0 if math.isfinite(float(payload.get("nll", "nan"))) else 1)
PY
}

make_visible_gpu_array() {
  local visible_gpu_csv="${CUDA_VISIBLE_DEVICES:-}"
  if [ -z "$visible_gpu_csv" ]; then
    visible_gpu_csv="$(seq -s, 0 "$((NUM_GPUS - 1))")"
  fi
  visible_gpu_csv="${visible_gpu_csv// /}"
  IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$visible_gpu_csv"
}

eval_epoch_checkpoint_single_gpu() {
  local epoch="$1"
  local gpu_id="$2"
  local epoch_tag
  epoch_tag="$(printf "%03d" "$epoch")"
  local ckpt="${WIKITEXT_EPOCH_CKPT_DIR}/risk_router_epoch${epoch_tag}.pt"
  local out="${WIKITEXT_VAL_METRIC_DIR}/opal_epoch${epoch_tag}_validation.json"
  if json_usable "$out"; then
    echo "=== Reuse validation metric: ${out} ==="
    return 0
  fi
  if [ ! -s "$ckpt" ]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    return 2
  fi
  echo "=== GPU ${gpu_id}: evaluate epoch ${epoch_tag} ==="
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  OMP_NUM_THREADS="$WIKITEXT_VALCKPT_OMP_NUM_THREADS" \
  MKL_NUM_THREADS="$WIKITEXT_VALCKPT_OMP_NUM_THREADS" \
  python3 ./eval_wikitext_opal_ppl.py eval \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split validation \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method router \
    --method_label "OPAL-SetBCE epoch${epoch} validation" \
    --risk_router_ckpt "$ckpt" \
    --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_opal_epoch${epoch_tag}_validation" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --output_json "$out" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
}

eval_epoch_checkpoint_worker() {
  local worker_idx="$1"
  local worker_count="$2"
  local gpu_id="$3"
  local epoch
  for epoch in $(seq 1 "$WIKITEXT_EPOCHS"); do
    if [ "$(( (epoch - 1) % worker_count ))" -ne "$worker_idx" ]; then
      continue
    fi
    eval_epoch_checkpoint_single_gpu "$epoch" "$gpu_id"
  done
}

eval_all_epoch_checkpoints_parallel() {
  make_visible_gpu_array
  if [ "${#VISIBLE_GPU_IDS[@]}" -eq 0 ]; then
    echo "No visible GPUs found from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}." >&2
    return 2
  fi
  local worker_count="$WIKITEXT_VALCKPT_PARALLEL_WORKERS"
  if [ "$worker_count" -gt "${#VISIBLE_GPU_IDS[@]}" ]; then
    worker_count="${#VISIBLE_GPU_IDS[@]}"
  fi
  if [ "$worker_count" -gt "$WIKITEXT_EPOCHS" ]; then
    worker_count="$WIKITEXT_EPOCHS"
  fi
  if [ "$worker_count" -lt 1 ]; then
    echo "WIKITEXT_VALCKPT_PARALLEL_WORKERS must be >= 1." >&2
    return 2
  fi

  echo "=== Parallel validation sweep: ${worker_count} single-GPU workers over ${WIKITEXT_EPOCHS} checkpoints ==="
  echo "=== Visible GPUs: ${VISIBLE_GPU_IDS[*]} ==="
  local -a worker_pids=()
  local -a worker_logs=()
  local worker_idx
  for worker_idx in $(seq 0 "$((worker_count - 1))"); do
    local gpu_id="${VISIBLE_GPU_IDS[$worker_idx]}"
    local log_path="${WIKITEXT_VAL_METRIC_DIR}/validation_worker_gpu${gpu_id}.log"
    worker_logs+=("$log_path")
    echo "=== Start validation worker ${worker_idx} on GPU ${gpu_id}; log ${log_path} ==="
    (
      set -euo pipefail
      eval_epoch_checkpoint_worker "$worker_idx" "$worker_count" "$gpu_id"
    ) > "$log_path" 2>&1 &
    worker_pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done

  if [ "$status" -ne 0 ]; then
    echo "=== At least one validation worker failed. Log tails follow. ===" >&2
    local log_path
    for log_path in "${worker_logs[@]}"; do
      echo "--- ${log_path} ---" >&2
      tail -n 80 "$log_path" >&2 || true
    done
    return "$status"
  fi

  local missing=0
  local epoch
  for epoch in $(seq 1 "$WIKITEXT_EPOCHS"); do
    local out="${WIKITEXT_VAL_METRIC_DIR}/opal_epoch$(printf "%03d" "$epoch")_validation.json"
    if ! json_usable "$out"; then
      echo "Missing or unusable validation metric after parallel sweep: ${out}" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    return 2
  fi
  echo "=== Parallel validation sweep complete ==="
}

if [ ! -s "$WIKITEXT_LABEL_FILE" ]; then
  echo "Missing greedy labels: ${WIKITEXT_LABEL_FILE}" >&2
  echo "Run run_wikitext2_public_lm_sanity_gpu01234567.sh once for this label run first." >&2
  exit 2
fi

echo "=== WikiText-2 OPAL validation-checkpoint selection ==="
echo "WIKITEXT_RUN_ID=${WIKITEXT_RUN_ID}"
echo "WIKITEXT_LABEL_RUN_ID=${WIKITEXT_LABEL_RUN_ID}"
echo "WIKITEXT_PREFIX_DEPTH=${WIKITEXT_PREFIX_DEPTH}"
echo "WIKITEXT_LABEL_FILE=${WIKITEXT_LABEL_FILE}"
echo "WIKITEXT_EPOCH_CKPT_DIR=${WIKITEXT_EPOCH_CKPT_DIR}"
echo "WIKITEXT_VAL_METRIC_DIR=${WIKITEXT_VAL_METRIC_DIR}"

last_ckpt="${WIKITEXT_EPOCH_CKPT_DIR}/risk_router_epoch$(printf "%03d" "$WIKITEXT_EPOCHS").pt"
if [ -s "$last_ckpt" ]; then
  echo "=== Reuse epoch checkpoints through ${last_ckpt} ==="
else
  echo "=== Train OPAL-SetBCE with per-epoch checkpoints ==="
  run_accelerate ./eval_wikitext_opal_ppl.py train_router \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split train \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --risk_label_file "$WIKITEXT_LABEL_FILE" \
    --router_input prefix_hk_raw_attn \
    --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --batch_size "$WIKITEXT_TRAIN_BATCH_SIZE" \
    --epochs "$WIKITEXT_EPOCHS" \
    --lr "$WIKITEXT_LR" \
    --router_dim "$WIKITEXT_ROUTER_DIM" \
    --router_heads "$WIKITEXT_ROUTER_HEADS" \
    --max_grad_norm "$WIKITEXT_MAX_GRAD_NORM" \
    --output_dir "$WIKITEXT_VALCKPT_DIR" \
    --save_epoch_checkpoints \
    --epoch_checkpoint_dir "$WIKITEXT_EPOCH_CKPT_DIR" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

echo "=== Evaluate every epoch checkpoint on validation ==="
eval_all_epoch_checkpoints_parallel

echo "=== Select best validation checkpoint ==="
python3 - "$WIKITEXT_VAL_METRIC_DIR" "$WIKITEXT_EPOCH_CKPT_DIR" "$WIKITEXT_BEST_JSON" <<'PY'
import glob, json, os, sys
metric_dir, ckpt_dir, best_json = sys.argv[1:]
rows = []
for path in sorted(glob.glob(os.path.join(metric_dir, "opal_epoch*_validation.json"))):
    payload = json.load(open(path))
    epoch = int(os.path.basename(path).split("epoch", 1)[1].split("_", 1)[0])
    rows.append({
        "epoch": epoch,
        "nll": float(payload["nll"]),
        "ppl": float(payload["ppl"]),
        "metric_json": path,
        "checkpoint": os.path.join(ckpt_dir, f"risk_router_epoch{epoch:03d}.pt"),
    })
if not rows:
    raise SystemExit("No validation metrics found.")
best = min(rows, key=lambda row: (row["nll"], row["epoch"]))
payload = {"best": best, "rows": rows}
json.dump(payload, open(best_json, "w"), indent=2)
print(json.dumps(best, indent=2))
PY

best_ckpt="$(python3 - "$WIKITEXT_BEST_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best"]["checkpoint"])
PY
)"
best_epoch="$(python3 - "$WIKITEXT_BEST_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best"]["epoch"])
PY
)"
test_json="${WIKITEXT_VAL_METRIC_DIR}/opal_best_val_epoch$(printf "%03d" "$best_epoch")_test.json"
if json_usable "$test_json"; then
  echo "=== Reuse best-checkpoint test metric: ${test_json} ==="
else
  echo "=== Evaluate best validation checkpoint on test: epoch ${best_epoch} ==="
  run_accelerate ./eval_wikitext_opal_ppl.py eval \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split test \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method router \
    --method_label "OPAL-SetBCE best-on-val epoch${best_epoch}" \
    --risk_router_ckpt "$best_ckpt" \
    --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_opal_best_val_epoch$(printf "%03d" "$best_epoch")_test" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --output_json "$test_json" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

echo "=== Compact validation-checkpoint result ==="
python3 - "$WIKITEXT_BEST_JSON" "$test_json" <<'PY'
import json, sys
best = json.load(open(sys.argv[1]))["best"]
test = json.load(open(sys.argv[2]))
print(json.dumps({
    "best_epoch": best["epoch"],
    "validation_NLL": best["nll"],
    "validation_PPL": best["ppl"],
    "test_NLL": test["nll"],
    "test_PPL": test["ppl"],
    "test_eval_tokens": test["eval_tokens"],
    "unique_masks": test.get("unique_masks"),
    "exact_skip_count_rate": test.get("exact_skip_count_rate"),
}, ensure_ascii=False))
PY

echo "=== Done: WikiText-2 OPAL validation-checkpoint selection ${WIKITEXT_RUN_ID} ==="
