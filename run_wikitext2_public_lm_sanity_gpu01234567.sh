#!/usr/bin/env bash
set -euo pipefail

# WikiText-2 public LM sanity runner for OPAL-SetBCE.
# This intentionally runs only WikiText-2 PPL:
#   Full, static uniform, static ends_heavy, static best-on-val C6,
#   Raw-SetBCE, and OPAL-SetBCE.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${WIKITEXT_BASE_PORT:-58000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${HOME}/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export WIKITEXT_MODEL_PATH="${WIKITEXT_MODEL_PATH:-}"
export WIKITEXT_LABEL_SAMPLES="${WIKITEXT_LABEL_SAMPLES:-2000}"
export WIKITEXT_EVAL_WINDOWS="${WIKITEXT_EVAL_WINDOWS:-512}"
export WIKITEXT_EPOCHS="${WIKITEXT_EPOCHS:-40}"
export WIKITEXT_SEQ_LEN="${WIKITEXT_SEQ_LEN:-1024}"
export WIKITEXT_ROUTER_PREFIX_TOKENS="${WIKITEXT_ROUTER_PREFIX_TOKENS:-$((WIKITEXT_SEQ_LEN / 4))}"
export WIKITEXT_SEED="${WIKITEXT_SEED:-42}"
export WIKITEXT_SKIP_RATE="${WIKITEXT_SKIP_RATE:-0.25}"
export WIKITEXT_SKIP_COUNT="${WIKITEXT_SKIP_COUNT:-0}"
export WIKITEXT_PROTECTED_HEAD="${WIKITEXT_PROTECTED_HEAD:-4}"
export WIKITEXT_PROTECTED_TAIL="${WIKITEXT_PROTECTED_TAIL:-2}"
export WIKITEXT_PREFIX_DEPTH="${WIKITEXT_PREFIX_DEPTH:-4}"
export WIKITEXT_PRECISION="${WIKITEXT_PRECISION:-bf16}"
export WIKITEXT_LABEL_BATCH_SIZE="${WIKITEXT_LABEL_BATCH_SIZE:-1}"
export WIKITEXT_CANDIDATE_BATCH_SIZE="${WIKITEXT_CANDIDATE_BATCH_SIZE:-1}"
export WIKITEXT_TRAIN_BATCH_SIZE="${WIKITEXT_TRAIN_BATCH_SIZE:-4}"
export WIKITEXT_EVAL_BATCH_SIZE="${WIKITEXT_EVAL_BATCH_SIZE:-1}"
export WIKITEXT_LR="${WIKITEXT_LR:-1e-4}"
export WIKITEXT_ROUTER_DIM="${WIKITEXT_ROUTER_DIM:-256}"
export WIKITEXT_ROUTER_HEADS="${WIKITEXT_ROUTER_HEADS:-4}"
export WIKITEXT_MAX_GRAD_NORM="${WIKITEXT_MAX_GRAD_NORM:-1.0}"
export WIKITEXT_DATASET_CACHE_DIR="${WIKITEXT_DATASET_CACHE_DIR:-}"
export WIKITEXT_MASK_IMPL_TAG="${WIKITEXT_MASK_IMPL_TAG:-maskcfg}"
if [ -z "${WIKITEXT_DATASET_DISK_PATH:-}" ] && [ -d "/workspace/datasets/wikitext/wikitext-2-raw-v1" ]; then
  export WIKITEXT_DATASET_DISK_PATH="/workspace/datasets/wikitext/wikitext-2-raw-v1"
else
  export WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-}"
fi
export WIKITEXT_RUN_RAW="${WIKITEXT_RUN_RAW:-1}"

if [ -z "$WIKITEXT_MODEL_PATH" ]; then
  cat >&2 <<'EOF'
WIKITEXT_MODEL_PATH is required.

First inspect server checkpoints in tmux:

cd /workspace/PriorDynamicPruning
find /workspace/ckpts -maxdepth 4 -type d \
  | grep -Ei 'qwen|llama|mistral|MiniOneRec' \
  | head -80

Prefer a Qwen2/Qwen2.5 base 1.5B or 3B checkpoint.
EOF
  exit 2
fi

if [ "$WIKITEXT_ROUTER_PREFIX_TOKENS" -ge "$WIKITEXT_SEQ_LEN" ]; then
  echo "WIKITEXT_ROUTER_PREFIX_TOKENS must be smaller than WIKITEXT_SEQ_LEN." >&2
  exit 2
fi

cd "$REPO_DIR"

model_tag="$(basename "$WIKITEXT_MODEL_PATH" | tr ' ./:' '____')"
skip_tag="$(printf "%s" "$WIKITEXT_SKIP_RATE" | tr "." "p")"
skip_budget_tag=""
if [ "$WIKITEXT_SKIP_COUNT" != "0" ]; then
  skip_budget_tag="_K${WIKITEXT_SKIP_COUNT}"
fi
export WIKITEXT_LABEL_RUN_ID="${WIKITEXT_LABEL_RUN_ID:-wikitext2_${model_tag}_${WIKITEXT_MASK_IMPL_TAG}_seq${WIKITEXT_SEQ_LEN}_pref${WIKITEXT_ROUTER_PREFIX_TOKENS}_m${WIKITEXT_LABEL_SAMPLES}_seed${WIKITEXT_SEED}_skip${skip_tag}${skip_budget_tag}}"
prefix_depth_tag=""
if [ "$WIKITEXT_PREFIX_DEPTH" != "4" ]; then
  prefix_depth_tag="_h${WIKITEXT_PREFIX_DEPTH}"
fi
export WIKITEXT_RUN_ID="${WIKITEXT_RUN_ID:-${WIKITEXT_LABEL_RUN_ID}${prefix_depth_tag}}"
export WIKITEXT_RESULT_ROOT="${WIKITEXT_RESULT_ROOT:-${REPO_DIR}/results/wikitext2_public_lm_sanity}"
export WIKITEXT_LABEL_DIR="${WIKITEXT_LABEL_DIR:-${WIKITEXT_RESULT_ROOT}/labels}"
export WIKITEXT_METRIC_DIR="${WIKITEXT_METRIC_DIR:-${WIKITEXT_RESULT_ROOT}/metrics/${WIKITEXT_RUN_ID}}"
export WIKITEXT_DIAG_DIR="${WIKITEXT_DIAG_DIR:-${WIKITEXT_RESULT_ROOT}/diagnostics/${WIKITEXT_RUN_ID}}"
export WIKITEXT_CKPT_ROOT="${WIKITEXT_CKPT_ROOT:-${REPO_DIR}/policy_ckpts/wikitext2_public_lm_sanity/${WIKITEXT_RUN_ID}}"
export WIKITEXT_REPORT_MD="${WIKITEXT_REPORT_MD:-${REPO_DIR}/docs/WIKITEXT2_PUBLIC_LM_SANITY_RESULTS.md}"
export WIKITEXT_LABEL_FILE="${WIKITEXT_LABEL_DIR}/${WIKITEXT_LABEL_RUN_ID}_delta_nll_greedy_set_labels.jsonl"
export RAW_ROUTER_DIR="${WIKITEXT_CKPT_ROOT}/raw_embedding_bce"
export OPAL_ROUTER_DIR="${WIKITEXT_CKPT_ROOT}/prefix_hk_raw_attn_bce"
export RAW_ROUTER_CKPT="${RAW_ROUTER_DIR}/risk_router.pt"
export OPAL_ROUTER_CKPT="${OPAL_ROUTER_DIR}/risk_router.pt"

mkdir -p "$WIKITEXT_LABEL_DIR" "$WIKITEXT_METRIC_DIR" "$WIKITEXT_DIAG_DIR" "$WIKITEXT_CKPT_ROOT" "$(dirname "$WIKITEXT_REPORT_MD")"

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
    port_log="$(mktemp "/tmp/wikitext2_opal_${port}_XXXX.log")"
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

line_count_at_least() {
  local path="$1"
  local min_rows="$2"
  test -s "$path" && [ "$(wc -l < "$path")" -ge "$min_rows" ]
}

json_usable() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import math
import os
import sys

path = sys.argv[1]
if not os.path.exists(path) or os.path.getsize(path) == 0:
    sys.exit(1)
try:
    payload = json.load(open(path))
except Exception:
    sys.exit(1)
for key in ("nll", "ppl", "eval_tokens"):
    if key not in payload:
        sys.exit(1)
if not math.isfinite(float(payload["nll"])) or not math.isfinite(float(payload["ppl"])):
    sys.exit(1)
sys.exit(0)
PY
}

checkpoint_usable() {
  local ckpt="$1"
  local metrics
  metrics="$(dirname "$ckpt")/training_metrics.json"
  python3 - "$ckpt" "$metrics" <<'PY'
import json
import math
import os
import sys

ckpt, metrics = sys.argv[1], sys.argv[2]
if not os.path.exists(ckpt) or os.path.getsize(ckpt) == 0:
    sys.exit(1)
if not os.path.exists(metrics) or os.path.getsize(metrics) == 0:
    sys.exit(1)
try:
    payload = json.load(open(metrics))
except Exception:
    sys.exit(1)
history = payload.get("history") or []
if not history:
    sys.exit(1)
for row in history:
    loss = float(row.get("loss", "nan"))
    if not math.isfinite(loss):
        sys.exit(1)
sys.exit(0)
PY
}

eval_json() {
  local output_json="$1"
  shift
  if json_usable "$output_json"; then
    echo "=== Reuse eval JSON: ${output_json} ==="
  else
    run_accelerate ./eval_wikitext_opal_ppl.py eval "$@" --output_json "$output_json"
  fi
}

echo "=== WikiText-2 public LM sanity config ==="
echo "REPO_DIR=${REPO_DIR}"
echo "WIKITEXT_MODEL_PATH=${WIKITEXT_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "WIKITEXT_RUN_ID=${WIKITEXT_RUN_ID}"
echo "WIKITEXT_LABEL_RUN_ID=${WIKITEXT_LABEL_RUN_ID}"
echo "WIKITEXT_LABEL_SAMPLES=${WIKITEXT_LABEL_SAMPLES}"
echo "WIKITEXT_EVAL_WINDOWS=${WIKITEXT_EVAL_WINDOWS}"
echo "WIKITEXT_EPOCHS=${WIKITEXT_EPOCHS}"
echo "WIKITEXT_SEQ_LEN=${WIKITEXT_SEQ_LEN}"
echo "WIKITEXT_ROUTER_PREFIX_TOKENS=${WIKITEXT_ROUTER_PREFIX_TOKENS}"
echo "WIKITEXT_PREFIX_DEPTH=${WIKITEXT_PREFIX_DEPTH}"
echo "WIKITEXT_SKIP_RATE=${WIKITEXT_SKIP_RATE}"
echo "WIKITEXT_SKIP_COUNT=${WIKITEXT_SKIP_COUNT}"
echo "WIKITEXT_PROTECTED_HEAD=${WIKITEXT_PROTECTED_HEAD}"
echo "WIKITEXT_PROTECTED_TAIL=${WIKITEXT_PROTECTED_TAIL}"
echo "WIKITEXT_DATASET_DISK_PATH=${WIKITEXT_DATASET_DISK_PATH}"
echo "WIKITEXT_DATASET_CACHE_DIR=${WIKITEXT_DATASET_CACHE_DIR}"
echo "WIKITEXT_MASK_IMPL_TAG=${WIKITEXT_MASK_IMPL_TAG}"
echo "WIKITEXT_LABEL_FILE=${WIKITEXT_LABEL_FILE}"
echo "RAW_ROUTER_CKPT=${RAW_ROUTER_CKPT}"
echo "OPAL_ROUTER_CKPT=${OPAL_ROUTER_CKPT}"
echo "WIKITEXT_REPORT_MD=${WIKITEXT_REPORT_MD}"

if line_count_at_least "$WIKITEXT_LABEL_FILE" "$WIKITEXT_LABEL_SAMPLES"; then
  echo "=== Reuse WikiText-2 Delta_NLL greedy set labels: ${WIKITEXT_LABEL_FILE} ==="
else
  echo "=== Build WikiText-2 Delta_NLL greedy set labels ==="
  run_accelerate ./build_wikitext_greedy_set_labels.py \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split train \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --label_samples "$WIKITEXT_LABEL_SAMPLES" \
    --sample_strategy random \
    --sample_seed "$WIKITEXT_SEED" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --output "$WIKITEXT_LABEL_FILE" \
    --batch_size "$WIKITEXT_LABEL_BATCH_SIZE" \
    --candidate_batch_size "$WIKITEXT_CANDIDATE_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

if [ "$WIKITEXT_RUN_RAW" = "1" ]; then
  if checkpoint_usable "$RAW_ROUTER_CKPT"; then
    echo "=== Reuse Raw-SetBCE checkpoint: ${RAW_ROUTER_CKPT} ==="
  else
    echo "=== Train Raw-SetBCE: raw_embedding + BCE ==="
    run_accelerate ./eval_wikitext_opal_ppl.py train_router \
      --teacher_model "$WIKITEXT_MODEL_PATH" \
      --split train \
      --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
      --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
      --seq_len "$WIKITEXT_SEQ_LEN" \
      --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
      --risk_label_file "$WIKITEXT_LABEL_FILE" \
      --router_input raw_embedding \
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
      --output_dir "$RAW_ROUTER_DIR" \
      --precision "$WIKITEXT_PRECISION" \
      --seed "$WIKITEXT_SEED"
  fi
fi

if checkpoint_usable "$OPAL_ROUTER_CKPT"; then
  echo "=== Reuse OPAL-SetBCE checkpoint: ${OPAL_ROUTER_CKPT} ==="
else
  echo "=== Train OPAL-SetBCE: prefix_hk_raw_attn + BCE ==="
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
    --output_dir "$OPAL_ROUTER_DIR" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

FULL_TEST_JSON="${WIKITEXT_METRIC_DIR}/full_test.json"
UNIFORM_TEST_JSON="${WIKITEXT_METRIC_DIR}/static_uniform_test.json"
ENDS_TEST_JSON="${WIKITEXT_METRIC_DIR}/static_ends_heavy_test.json"
BEST_TEST_JSON="${WIKITEXT_METRIC_DIR}/static_best_on_val_c6_test.json"
RAW_TEST_JSON="${WIKITEXT_METRIC_DIR}/raw_setbce_test.json"
OPAL_TEST_JSON="${WIKITEXT_METRIC_DIR}/opal_setbce_test.json"

eval_json "$FULL_TEST_JSON" \
  --teacher_model "$WIKITEXT_MODEL_PATH" \
  --split test \
  --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
  --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --method full \
  --method_label Full \
  --run_name "${WIKITEXT_RUN_ID}_full_test" \
  --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
  --precision "$WIKITEXT_PRECISION" \
  --seed "$WIKITEXT_SEED"

echo "=== Select Static best-on-val C6 on validation split ==="
for strategy in uniform ends_heavy first_k last_k middle_heavy random_diverse_seed42; do
  val_json="${WIKITEXT_METRIC_DIR}/static_${strategy}_validation.json"
  eval_json "$val_json" \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split validation \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method static \
    --method_label "Static ${strategy} validation" \
    --static_strategy "$strategy" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_static_${strategy}_validation" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
done

BEST_STATIC_STRATEGY="$(python3 - "$WIKITEXT_METRIC_DIR" <<'PY'
import json
import os
import sys

metric_dir = sys.argv[1]
strategies = ["uniform", "ends_heavy", "first_k", "last_k", "middle_heavy", "random_diverse_seed42"]
rows = []
for strategy in strategies:
    path = os.path.join(metric_dir, f"static_{strategy}_validation.json")
    payload = json.load(open(path))
    rows.append((float(payload["nll"]), strategy))
rows.sort()
print(rows[0][1])
PY
)"
BEST_STATIC_VAL_JSON="${WIKITEXT_METRIC_DIR}/static_${BEST_STATIC_STRATEGY}_validation.json"
echo "=== Static best-on-val C6 selected: ${BEST_STATIC_STRATEGY} ==="

eval_json "$UNIFORM_TEST_JSON" \
  --teacher_model "$WIKITEXT_MODEL_PATH" \
  --split test \
  --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
  --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --method static \
  --method_label "Static uniform" \
  --static_strategy uniform \
  --skip_rate "$WIKITEXT_SKIP_RATE" \
  --skip_count "$WIKITEXT_SKIP_COUNT" \
  --protected_head "$WIKITEXT_PROTECTED_HEAD" \
  --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
  --run_name "${WIKITEXT_RUN_ID}_static_uniform_test" \
  --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
  --precision "$WIKITEXT_PRECISION" \
  --seed "$WIKITEXT_SEED"

python3 - "$FULL_TEST_JSON" "$UNIFORM_TEST_JSON" <<'PY'
import json
import math
import sys

full = json.load(open(sys.argv[1]))
static = json.load(open(sys.argv[2]))
skip_count = int(static.get("skip_count") or 0)
delta = abs(float(static["nll"]) - float(full["nll"]))
if skip_count > 0 and delta <= 1e-12:
    print(
        "Layer-mask sanity failed: Static uniform NLL is exactly equal to Full NLL "
        f"with skip_count={skip_count}. This usually means the model forward ignored the skip mask.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"Layer-mask sanity ok: static_uniform_delta_nll={float(static['nll']) - float(full['nll']):.12g}")
PY

eval_json "$ENDS_TEST_JSON" \
  --teacher_model "$WIKITEXT_MODEL_PATH" \
  --split test \
  --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
  --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --method static \
  --method_label "Static ends_heavy" \
  --static_strategy ends_heavy \
  --skip_rate "$WIKITEXT_SKIP_RATE" \
  --skip_count "$WIKITEXT_SKIP_COUNT" \
  --protected_head "$WIKITEXT_PROTECTED_HEAD" \
  --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
  --run_name "${WIKITEXT_RUN_ID}_static_ends_heavy_test" \
  --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
  --precision "$WIKITEXT_PRECISION" \
  --seed "$WIKITEXT_SEED"

eval_json "$BEST_TEST_JSON" \
  --teacher_model "$WIKITEXT_MODEL_PATH" \
  --split test \
  --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
  --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --method static \
  --method_label "Static best-on-val C6" \
  --static_strategy "$BEST_STATIC_STRATEGY" \
  --skip_rate "$WIKITEXT_SKIP_RATE" \
  --skip_count "$WIKITEXT_SKIP_COUNT" \
  --protected_head "$WIKITEXT_PROTECTED_HEAD" \
  --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
  --run_name "${WIKITEXT_RUN_ID}_static_best_on_val_c6_test" \
  --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
  --precision "$WIKITEXT_PRECISION" \
  --seed "$WIKITEXT_SEED"

if [ "$WIKITEXT_RUN_RAW" = "1" ]; then
  eval_json "$RAW_TEST_JSON" \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split test \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method router \
    --method_label "Raw-SetBCE" \
    --risk_router_ckpt "$RAW_ROUTER_CKPT" \
    --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_raw_setbce_test" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

eval_json "$OPAL_TEST_JSON" \
  --teacher_model "$WIKITEXT_MODEL_PATH" \
  --split test \
  --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
  --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --method router \
  --method_label "OPAL-SetBCE" \
  --risk_router_ckpt "$OPAL_ROUTER_CKPT" \
  --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
  --skip_rate "$WIKITEXT_SKIP_RATE" \
  --skip_count "$WIKITEXT_SKIP_COUNT" \
  --protected_head "$WIKITEXT_PROTECTED_HEAD" \
  --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
  --run_name "${WIKITEXT_RUN_ID}_opal_setbce_test" \
  --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
  --precision "$WIKITEXT_PRECISION" \
  --seed "$WIKITEXT_SEED"

RAW_OVERLAP_SUMMARY=""
if [ "$WIKITEXT_RUN_RAW" = "1" ]; then
  RAW_OVERLAP_JSONL="${WIKITEXT_DIAG_DIR}/raw_setbce_train_overlap.jsonl"
  RAW_OVERLAP_SUMMARY="${WIKITEXT_DIAG_DIR}/raw_setbce_train_overlap.summary.json"
  if [ -s "$RAW_OVERLAP_SUMMARY" ]; then
    echo "=== Reuse Raw overlap summary: ${RAW_OVERLAP_SUMMARY} ==="
  else
    run_accelerate ./eval_wikitext_opal_ppl.py diagnose_overlap \
      --teacher_model "$WIKITEXT_MODEL_PATH" \
      --split train \
      --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
      --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
      --seq_len "$WIKITEXT_SEQ_LEN" \
      --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
      --risk_label_file "$WIKITEXT_LABEL_FILE" \
      --risk_router_ckpt "$RAW_ROUTER_CKPT" \
      --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
      --protected_head "$WIKITEXT_PROTECTED_HEAD" \
      --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
      --max_samples 0 \
      --sample_strategy first \
      --sample_seed "$WIKITEXT_SEED" \
      --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
      --output_jsonl "$RAW_OVERLAP_JSONL" \
      --summary_json "$RAW_OVERLAP_SUMMARY" \
      --precision "$WIKITEXT_PRECISION" \
      --seed "$WIKITEXT_SEED"
  fi
fi

OPAL_OVERLAP_JSONL="${WIKITEXT_DIAG_DIR}/opal_setbce_train_overlap.jsonl"
OPAL_OVERLAP_SUMMARY="${WIKITEXT_DIAG_DIR}/opal_setbce_train_overlap.summary.json"
if [ -s "$OPAL_OVERLAP_SUMMARY" ]; then
  echo "=== Reuse OPAL overlap summary: ${OPAL_OVERLAP_SUMMARY} ==="
else
  run_accelerate ./eval_wikitext_opal_ppl.py diagnose_overlap \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split train \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --risk_label_file "$WIKITEXT_LABEL_FILE" \
    --risk_router_ckpt "$OPAL_ROUTER_CKPT" \
    --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --max_samples 0 \
    --sample_strategy first \
    --sample_seed "$WIKITEXT_SEED" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --output_jsonl "$OPAL_OVERLAP_JSONL" \
    --summary_json "$OPAL_OVERLAP_SUMMARY" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

REPORT_ARGS=(
  write_report
  --metric "Full=${FULL_TEST_JSON}"
  --metric "Static uniform=${UNIFORM_TEST_JSON}"
  --metric "Static ends_heavy=${ENDS_TEST_JSON}"
  --metric "Static best-on-val C6=${BEST_TEST_JSON}"
  --metric "OPAL-SetBCE=${OPAL_TEST_JSON}"
  --label_metadata "${WIKITEXT_LABEL_FILE}.metadata.json"
  --static_best_strategy "$BEST_STATIC_STRATEGY"
  --static_best_val_json "$BEST_STATIC_VAL_JSON"
  --opal_training_metrics "${OPAL_ROUTER_DIR}/training_metrics.json"
  --opal_overlap_summary "$OPAL_OVERLAP_SUMMARY"
  --output_md "$WIKITEXT_REPORT_MD"
  --date "$(date +%F)"
)

if [ "$WIKITEXT_RUN_RAW" = "1" ]; then
  REPORT_ARGS+=(--metric "Raw-SetBCE=${RAW_TEST_JSON}")
  REPORT_ARGS+=(--raw_training_metrics "${RAW_ROUTER_DIR}/training_metrics.json")
  REPORT_ARGS+=(--raw_overlap_summary "$RAW_OVERLAP_SUMMARY")
fi

echo "=== Write WikiText-2 public LM sanity report ==="
python3 ./eval_wikitext_opal_ppl.py "${REPORT_ARGS[@]}"

echo "=== Compact final metrics ==="
python3 - "$FULL_TEST_JSON" "$UNIFORM_TEST_JSON" "$ENDS_TEST_JSON" "$BEST_TEST_JSON" "$RAW_TEST_JSON" "$OPAL_TEST_JSON" <<'PY'
import json
import os
import sys

names = ["Full", "Static uniform", "Static ends_heavy", "Static best-on-val C6", "Raw-SetBCE", "OPAL-SetBCE"]
paths = sys.argv[1:]
full = json.load(open(paths[0]))
for name, path in zip(names, paths):
    if not os.path.exists(path):
        continue
    row = json.load(open(path))
    print(json.dumps({
        "method": name,
        "NLL": row["nll"],
        "PPL": row["ppl"],
        "Delta_NLL": row["nll"] - full["nll"],
        "Delta_PPL": row["ppl"] - full["ppl"],
        "eval_tokens": row["eval_tokens"],
        "unique_masks": row.get("unique_masks"),
        "average_kept_layers": row.get("average_kept_layers"),
        "exact_skip_count_rate": row.get("exact_skip_count_rate"),
    }, ensure_ascii=False))
PY

echo "=== Done: WikiText-2 public LM sanity ${WIKITEXT_RUN_ID} ==="
