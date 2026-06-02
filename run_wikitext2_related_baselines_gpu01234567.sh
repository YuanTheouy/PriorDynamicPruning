#!/usr/bin/env bash
set -euo pipefail

# WikiText-2 public LM related-baseline rescue runner.
# Runs matched PuDDing-style, IG-style, and layerwise-hidden baselines
# against the existing OPAL WikiText-2 public LM sanity setup.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${WIKITEXT_BASE_PORT:-58200}"
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
export WIKITEXT_CANDIDATE_BATCH_SIZE="${WIKITEXT_CANDIDATE_BATCH_SIZE:-4}"
export WIKITEXT_TRAIN_BATCH_SIZE="${WIKITEXT_TRAIN_BATCH_SIZE:-4}"
export WIKITEXT_EVAL_BATCH_SIZE="${WIKITEXT_EVAL_BATCH_SIZE:-1}"
export WIKITEXT_LR="${WIKITEXT_LR:-1e-4}"
export WIKITEXT_ROUTER_DIM="${WIKITEXT_ROUTER_DIM:-256}"
export WIKITEXT_ROUTER_HEADS="${WIKITEXT_ROUTER_HEADS:-4}"
export WIKITEXT_MAX_GRAD_NORM="${WIKITEXT_MAX_GRAD_NORM:-1.0}"
export WIKITEXT_DATASET_CACHE_DIR="${WIKITEXT_DATASET_CACHE_DIR:-}"
export WIKITEXT_MASK_IMPL_TAG="${WIKITEXT_MASK_IMPL_TAG:-maskcfg}"
export WIKITEXT_RUN_PUDDING="${WIKITEXT_RUN_PUDDING:-1}"
export WIKITEXT_RUN_IG="${WIKITEXT_RUN_IG:-1}"
export WIKITEXT_RUN_LAYERWISE="${WIKITEXT_RUN_LAYERWISE:-1}"
export WIKITEXT_IG_CLUSTERS="${WIKITEXT_IG_CLUSTERS:-8}"
export WIKITEXT_IG_KMEANS_ITERS="${WIKITEXT_IG_KMEANS_ITERS:-30}"
if [ -z "${WIKITEXT_DATASET_DISK_PATH:-}" ] && [ -d "/workspace/datasets/wikitext/wikitext-2-raw-v1" ]; then
  export WIKITEXT_DATASET_DISK_PATH="/workspace/datasets/wikitext/wikitext-2-raw-v1"
else
  export WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-}"
fi

if [ -z "$WIKITEXT_MODEL_PATH" ]; then
  cat >&2 <<'EOF'
WIKITEXT_MODEL_PATH is required, for example:

WIKITEXT_MODEL_PATH=/workspace/ckpts/Qwen2.5-1.5B \
bash ./run_wikitext2_related_baselines_gpu01234567.sh
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
export WIKITEXT_LABEL_RUN_ID="${WIKITEXT_LABEL_RUN_ID:-wikitext2_${model_tag}_${WIKITEXT_MASK_IMPL_TAG}_seq${WIKITEXT_SEQ_LEN}_pref${WIKITEXT_ROUTER_PREFIX_TOKENS}_m${WIKITEXT_LABEL_SAMPLES}_seed${WIKITEXT_SEED}_skip${skip_tag}}"
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
export WIKITEXT_RELATED_LABEL_DIR="${WIKITEXT_RELATED_LABEL_DIR:-${WIKITEXT_RESULT_ROOT}/related_labels}"
export WIKITEXT_RELATED_METRIC_DIR="${WIKITEXT_RELATED_METRIC_DIR:-${WIKITEXT_RESULT_ROOT}/related_metrics/${WIKITEXT_RUN_ID}}"
export WIKITEXT_RELATED_CKPT_ROOT="${WIKITEXT_RELATED_CKPT_ROOT:-${REPO_DIR}/policy_ckpts/wikitext2_public_lm_related/${WIKITEXT_RUN_ID}}"
export WIKITEXT_REPORT_MD="${WIKITEXT_REPORT_MD:-${REPO_DIR}/docs/WIKITEXT2_PUBLIC_LM_SANITY_RESULTS.md}"
export WIKITEXT_RELATED_REPORT_MD="${WIKITEXT_RELATED_REPORT_MD:-${REPO_DIR}/docs/WIKITEXT2_PUBLIC_LM_RELATED_RESULTS.md}"
export WIKITEXT_LABEL_FILE="${WIKITEXT_LABEL_DIR}/${WIKITEXT_LABEL_RUN_ID}_delta_nll_greedy_set_labels.jsonl"
export WIKITEXT_CANDIDATE_LABEL_FILE="${WIKITEXT_RELATED_LABEL_DIR}/${WIKITEXT_LABEL_RUN_ID}_c16_candidate_delta_nll.jsonl"

export RAW_ROUTER_DIR="${WIKITEXT_CKPT_ROOT}/raw_embedding_bce"
export OPAL_ROUTER_DIR="${WIKITEXT_CKPT_ROOT}/prefix_hk_raw_attn_bce"
export RAW_ROUTER_CKPT="${RAW_ROUTER_DIR}/risk_router.pt"
export OPAL_ROUTER_CKPT="${OPAL_ROUTER_DIR}/risk_router.pt"
export PUDDING_ROUTER_DIR="${WIKITEXT_RELATED_CKPT_ROOT}/pudding_candidate_quality"
export PUDDING_ROUTER_CKPT="${PUDDING_ROUTER_DIR}/candidate_router.pt"
export IG_ARTIFACT="${WIKITEXT_RELATED_CKPT_ROOT}/ig_prefix_k${WIKITEXT_IG_CLUSTERS}.pt"
export IG_SUMMARY="${WIKITEXT_RELATED_CKPT_ROOT}/ig_prefix_k${WIKITEXT_IG_CLUSTERS}.summary.json"
export LAYERWISE_ROUTER_DIR="${WIKITEXT_RELATED_CKPT_ROOT}/layerwise_hidden_bce"
export LAYERWISE_ROUTER_CKPT="${LAYERWISE_ROUTER_DIR}/risk_router.pt"

mkdir -p \
  "$WIKITEXT_LABEL_DIR" \
  "$WIKITEXT_METRIC_DIR" \
  "$WIKITEXT_DIAG_DIR" \
  "$WIKITEXT_CKPT_ROOT" \
  "$WIKITEXT_RELATED_LABEL_DIR" \
  "$WIKITEXT_RELATED_METRIC_DIR" \
  "$WIKITEXT_RELATED_CKPT_ROOT" \
  "$(dirname "$WIKITEXT_RELATED_REPORT_MD")"

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
    port_log="$(mktemp "/tmp/wikitext2_related_${port}_XXXX.log")"
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
payload = json.load(open(metrics))
history = payload.get("history") or []
if not history:
    sys.exit(1)
for row in history:
    if not math.isfinite(float(row.get("loss", "nan"))):
        sys.exit(1)
sys.exit(0)
PY
}

candidate_checkpoint_usable() {
  checkpoint_usable "$1"
}

artifact_usable() {
  local path="$1"
  test -s "$path"
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

FULL_TEST_JSON="${WIKITEXT_METRIC_DIR}/full_test.json"
UNIFORM_TEST_JSON="${WIKITEXT_METRIC_DIR}/static_uniform_test.json"
ENDS_TEST_JSON="${WIKITEXT_METRIC_DIR}/static_ends_heavy_test.json"
BEST_TEST_JSON="${WIKITEXT_METRIC_DIR}/static_best_on_val_c6_test.json"
RAW_TEST_JSON="${WIKITEXT_METRIC_DIR}/raw_setbce_test.json"
OPAL_TEST_JSON="${WIKITEXT_METRIC_DIR}/opal_setbce_test.json"
PUDDING_TEST_JSON="${WIKITEXT_RELATED_METRIC_DIR}/pudding_style_test.json"
IG_TEST_JSON="${WIKITEXT_RELATED_METRIC_DIR}/ig_style_test.json"
LAYERWISE_TEST_JSON="${WIKITEXT_RELATED_METRIC_DIR}/layerwise_hidden_router_test.json"

echo "=== WikiText-2 related-baseline rescue config ==="
echo "REPO_DIR=${REPO_DIR}"
echo "WIKITEXT_MODEL_PATH=${WIKITEXT_MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "WIKITEXT_RUN_ID=${WIKITEXT_RUN_ID}"
echo "WIKITEXT_LABEL_RUN_ID=${WIKITEXT_LABEL_RUN_ID}"
echo "WIKITEXT_LABEL_FILE=${WIKITEXT_LABEL_FILE}"
echo "WIKITEXT_CANDIDATE_LABEL_FILE=${WIKITEXT_CANDIDATE_LABEL_FILE}"
echo "WIKITEXT_PREFIX_DEPTH=${WIKITEXT_PREFIX_DEPTH}"
echo "WIKITEXT_RELATED_REPORT_MD=${WIKITEXT_RELATED_REPORT_MD}"
echo "WIKITEXT_RUN_PUDDING=${WIKITEXT_RUN_PUDDING}"
echo "WIKITEXT_RUN_IG=${WIKITEXT_RUN_IG}"
echo "WIKITEXT_RUN_LAYERWISE=${WIKITEXT_RUN_LAYERWISE}"

if ! json_usable "$FULL_TEST_JSON" || ! json_usable "$BEST_TEST_JSON" || ! json_usable "$RAW_TEST_JSON" || ! json_usable "$OPAL_TEST_JSON"; then
  echo "=== Public sanity metrics are missing; run the base WikiText-2 sanity first ==="
  WIKITEXT_BASE_PORT="$NEXT_PORT" bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
  NEXT_PORT="$((NEXT_PORT + 40))"
fi

if line_count_at_least "$WIKITEXT_CANDIDATE_LABEL_FILE" "$WIKITEXT_LABEL_SAMPLES"; then
  echo "=== Reuse C16 candidate labels: ${WIKITEXT_CANDIDATE_LABEL_FILE} ==="
else
  if ! line_count_at_least "$WIKITEXT_LABEL_FILE" "$WIKITEXT_LABEL_SAMPLES"; then
    echo "=== Greedy labels missing; run the base WikiText-2 sanity first ==="
    WIKITEXT_BASE_PORT="$NEXT_PORT" bash ./run_wikitext2_public_lm_sanity_gpu01234567.sh
    NEXT_PORT="$((NEXT_PORT + 40))"
  fi
  echo "=== Build WikiText-2 C16 candidate Delta_NLL labels ==="
  run_accelerate ./eval_wikitext_opal_ppl.py build_candidate_labels \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split train \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --reference_label_file "$WIKITEXT_LABEL_FILE" \
    --label_samples "$WIKITEXT_LABEL_SAMPLES" \
    --sample_strategy random \
    --sample_seed "$WIKITEXT_SEED" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --batch_size "$WIKITEXT_LABEL_BATCH_SIZE" \
    --candidate_batch_size "$WIKITEXT_CANDIDATE_BATCH_SIZE" \
    --output "$WIKITEXT_CANDIDATE_LABEL_FILE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

if [ "$WIKITEXT_RUN_PUDDING" = "1" ]; then
  if candidate_checkpoint_usable "$PUDDING_ROUTER_CKPT"; then
    echo "=== Reuse PuDDing-style candidate router: ${PUDDING_ROUTER_CKPT} ==="
  else
    echo "=== Train PuDDing-style candidate quality router ==="
    run_accelerate ./eval_wikitext_opal_ppl.py train_candidate_router \
      --teacher_model "$WIKITEXT_MODEL_PATH" \
      --split train \
      --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
      --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
      --seq_len "$WIKITEXT_SEQ_LEN" \
      --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
      --candidate_label_file "$WIKITEXT_CANDIDATE_LABEL_FILE" \
      --skip_rate "$WIKITEXT_SKIP_RATE" \
      --skip_count "$WIKITEXT_SKIP_COUNT" \
      --protected_head "$WIKITEXT_PROTECTED_HEAD" \
      --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
      --batch_size "$WIKITEXT_TRAIN_BATCH_SIZE" \
      --epochs "$WIKITEXT_EPOCHS" \
      --lr "$WIKITEXT_LR" \
      --max_grad_norm "$WIKITEXT_MAX_GRAD_NORM" \
      --output_dir "$PUDDING_ROUTER_DIR" \
      --precision "$WIKITEXT_PRECISION" \
      --seed "$WIKITEXT_SEED"
  fi
  eval_json "$PUDDING_TEST_JSON" \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split test \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method candidate_router \
    --method_label "PuDDing-style" \
    --candidate_router_ckpt "$PUDDING_ROUTER_CKPT" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_pudding_style_test" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

if [ "$WIKITEXT_RUN_IG" = "1" ]; then
  if artifact_usable "$IG_ARTIFACT" && test -s "$IG_SUMMARY"; then
    echo "=== Reuse IG-style artifact: ${IG_ARTIFACT} ==="
  else
    echo "=== Build IG-style prefix cluster mask artifact ==="
    run_accelerate ./eval_wikitext_opal_ppl.py build_ig_artifact \
      --teacher_model "$WIKITEXT_MODEL_PATH" \
      --split train \
      --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
      --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
      --seq_len "$WIKITEXT_SEQ_LEN" \
      --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
      --candidate_label_file "$WIKITEXT_CANDIDATE_LABEL_FILE" \
      --skip_rate "$WIKITEXT_SKIP_RATE" \
      --skip_count "$WIKITEXT_SKIP_COUNT" \
      --protected_head "$WIKITEXT_PROTECTED_HEAD" \
      --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
      --batch_size "$WIKITEXT_TRAIN_BATCH_SIZE" \
      --clusters "$WIKITEXT_IG_CLUSTERS" \
      --kmeans_iters "$WIKITEXT_IG_KMEANS_ITERS" \
      --output_artifact "$IG_ARTIFACT" \
      --output_summary "$IG_SUMMARY" \
      --precision "$WIKITEXT_PRECISION" \
      --seed "$WIKITEXT_SEED"
  fi
  eval_json "$IG_TEST_JSON" \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split test \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method ig \
    --method_label "IG-style" \
    --ig_artifact "$IG_ARTIFACT" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_ig_style_test" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

if [ "$WIKITEXT_RUN_LAYERWISE" = "1" ]; then
  if checkpoint_usable "$LAYERWISE_ROUTER_CKPT"; then
    echo "=== Reuse layerwise_hidden_router checkpoint: ${LAYERWISE_ROUTER_CKPT} ==="
  else
    echo "=== Train layerwise_hidden_router: stronger-access BCE baseline ==="
    run_accelerate ./eval_wikitext_opal_ppl.py train_router \
      --teacher_model "$WIKITEXT_MODEL_PATH" \
      --split train \
      --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
      --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
      --seq_len "$WIKITEXT_SEQ_LEN" \
      --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
      --risk_label_file "$WIKITEXT_LABEL_FILE" \
      --router_input layerwise_hidden \
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
      --output_dir "$LAYERWISE_ROUTER_DIR" \
      --precision "$WIKITEXT_PRECISION" \
      --seed "$WIKITEXT_SEED"
  fi
  eval_json "$LAYERWISE_TEST_JSON" \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split test \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method router \
    --method_label "layerwise_hidden_router" \
    --risk_router_ckpt "$LAYERWISE_ROUTER_CKPT" \
    --prefix_depth "$WIKITEXT_PREFIX_DEPTH" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "${WIKITEXT_RUN_ID}_layerwise_hidden_router_test" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED"
fi

REPORT_ARGS=(
  write_related_report
  --metric "Full=${FULL_TEST_JSON}"
  --metric "Static uniform=${UNIFORM_TEST_JSON}"
  --metric "Static ends_heavy=${ENDS_TEST_JSON}"
  --metric "Static best-on-val C6=${BEST_TEST_JSON}"
  --metric "Raw-SetBCE=${RAW_TEST_JSON}"
  --metric "OPAL-SetBCE=${OPAL_TEST_JSON}"
  --training_metric "Raw-SetBCE=${RAW_ROUTER_DIR}/training_metrics.json"
  --training_metric "OPAL-SetBCE=${OPAL_ROUTER_DIR}/training_metrics.json"
  --candidate_label_metadata "${WIKITEXT_CANDIDATE_LABEL_FILE}.metadata.json"
  --output_md "$WIKITEXT_RELATED_REPORT_MD"
  --date "$(date +%F)"
)

if [ "$WIKITEXT_RUN_PUDDING" = "1" ]; then
  REPORT_ARGS+=(--metric "PuDDing-style=${PUDDING_TEST_JSON}")
  REPORT_ARGS+=(--training_metric "PuDDing-style=${PUDDING_ROUTER_DIR}/training_metrics.json")
fi
if [ "$WIKITEXT_RUN_IG" = "1" ]; then
  REPORT_ARGS+=(--metric "IG-style=${IG_TEST_JSON}")
  REPORT_ARGS+=(--ig_summary "$IG_SUMMARY")
fi
if [ "$WIKITEXT_RUN_LAYERWISE" = "1" ]; then
  REPORT_ARGS+=(--metric "layerwise_hidden_router=${LAYERWISE_TEST_JSON}")
  REPORT_ARGS+=(--training_metric "layerwise_hidden_router=${LAYERWISE_ROUTER_DIR}/training_metrics.json")
fi
if [ -s "${WIKITEXT_DIAG_DIR}/raw_setbce_train_overlap.summary.json" ]; then
  REPORT_ARGS+=(--overlap_summary "Raw-SetBCE=${WIKITEXT_DIAG_DIR}/raw_setbce_train_overlap.summary.json")
fi
if [ -s "${WIKITEXT_DIAG_DIR}/opal_setbce_train_overlap.summary.json" ]; then
  REPORT_ARGS+=(--overlap_summary "OPAL-SetBCE=${WIKITEXT_DIAG_DIR}/opal_setbce_train_overlap.summary.json")
fi

echo "=== Write WikiText-2 related-baseline report ==="
python3 ./eval_wikitext_opal_ppl.py "${REPORT_ARGS[@]}"

echo "=== Compact final related metrics ==="
python3 - "$FULL_TEST_JSON" "$UNIFORM_TEST_JSON" "$ENDS_TEST_JSON" "$BEST_TEST_JSON" "$PUDDING_TEST_JSON" "$IG_TEST_JSON" "$LAYERWISE_TEST_JSON" "$RAW_TEST_JSON" "$OPAL_TEST_JSON" <<'PY'
import json
import os
import sys

names = [
    "Full",
    "Static uniform",
    "Static ends_heavy",
    "Static best-on-val C6",
    "PuDDing-style",
    "IG-style",
    "layerwise_hidden_router",
    "Raw-SetBCE",
    "OPAL-SetBCE",
]
paths = sys.argv[1:]
full = json.load(open(paths[0]))
for name, path in zip(names, paths):
    if not os.path.exists(path):
        print(json.dumps({"method": name, "status": "missing"}))
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
        "exact_skip_count_rate": row.get("exact_skip_count_rate"),
        "selected_candidate_distribution": row.get("selected_candidate_distribution", {}),
    }, ensure_ascii=False))
PY

echo "=== Done: WikiText-2 related-baseline rescue ${WIKITEXT_RUN_ID} ==="
