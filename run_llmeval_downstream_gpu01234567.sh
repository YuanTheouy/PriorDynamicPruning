#!/usr/bin/env bash
set -euo pipefail

# Unattended downstream evaluation for OPAL layer skipping via
# lm-evaluation-harness. This consumes existing WikiText-trained routers/artifacts
# and does not rerun WikiText PPL, label building, or router training.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${LLMEVAL_BASE_PORT:-59000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${HOME}/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The public lm-eval datasets are small, but Hugging Face may route parquet
# downloads through Xet/CAS, which is fragile from the server network. These
# defaults keep unattended runs on the regular HF/mirror path unless the caller
# explicitly overrides them.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

export LLMEVAL_MODEL_PATH="${LLMEVAL_MODEL_PATH:-/workspace/ckpts/Qwen2.5-1.5B}"
export LLMEVAL_TASKS="${LLMEVAL_TASKS:-piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge}"
export LLMEVAL_SEEDS="${LLMEVAL_SEEDS:-42 13 3407}"
export LLMEVAL_MAX_LENGTH="${LLMEVAL_MAX_LENGTH:-1024}"
export LLMEVAL_ROUTER_PREFIX_TOKENS="${LLMEVAL_ROUTER_PREFIX_TOKENS:-256}"
export LLMEVAL_BATCH_SIZE="${LLMEVAL_BATCH_SIZE:-8}"
export LLMEVAL_DTYPE="${LLMEVAL_DTYPE:-bfloat16}"
export LLMEVAL_SKIP_RATE="${LLMEVAL_SKIP_RATE:-0.25}"
export LLMEVAL_SKIP_COUNT="${LLMEVAL_SKIP_COUNT:-7}"
export LLMEVAL_PROTECTED_HEAD="${LLMEVAL_PROTECTED_HEAD:-4}"
export LLMEVAL_PROTECTED_TAIL="${LLMEVAL_PROTECTED_TAIL:-2}"
export LLMEVAL_PREFIX_DEPTH="${LLMEVAL_PREFIX_DEPTH:-4}"
export LLMEVAL_LABEL_SAMPLES="${LLMEVAL_LABEL_SAMPLES:-2000}"
export LLMEVAL_MASK_IMPL_TAG="${LLMEVAL_MASK_IMPL_TAG:-maskcfg}"
export LLMEVAL_OUTPUT_ROOT="${LLMEVAL_OUTPUT_ROOT:-${REPO_DIR}/results/llmeval_downstream}"
export LLMEVAL_REPORT_MD="${LLMEVAL_REPORT_MD:-${REPO_DIR}/docs/LLMEVAL_DOWNSTREAM_RESULTS.md}"
export LLMEVAL_LIMIT="${LLMEVAL_LIMIT:-0}"
export LLMEVAL_RUNNER_BACKEND="${LLMEVAL_RUNNER_BACKEND:-single_gpu_pool}"
export LLMEVAL_PREFETCH_LIMIT="${LLMEVAL_PREFETCH_LIMIT:-1}"
export LLMEVAL_COMPENSATION_MODE="${LLMEVAL_COMPENSATION_MODE:-none}"
export LLMEVAL_COMPENSATION_RANK="${LLMEVAL_COMPENSATION_RANK:-0}"
export LLMEVAL_COMPENSATION_STATIC_GATE="${LLMEVAL_COMPENSATION_STATIC_GATE:-1.0}"

cd "$REPO_DIR"

model_tag="$(basename "$LLMEVAL_MODEL_PATH" | tr ' ./:' '____')"
skip_tag="$(printf "%s" "$LLMEVAL_SKIP_RATE" | tr "." "p")"
export LLMEVAL_RUN_ID="${LLMEVAL_RUN_ID:-llmeval_${model_tag}_${LLMEVAL_MASK_IMPL_TAG}_len${LLMEVAL_MAX_LENGTH}_pref${LLMEVAL_ROUTER_PREFIX_TOKENS}_m${LLMEVAL_LABEL_SAMPLES}_skip${skip_tag}}"
export LLMEVAL_METRIC_ROOT="${LLMEVAL_OUTPUT_ROOT}/metrics"
export LLMEVAL_LOG_ROOT="${LLMEVAL_OUTPUT_ROOT}/logs/${LLMEVAL_RUN_ID}"
mkdir -p "$LLMEVAL_METRIC_ROOT" "$LLMEVAL_LOG_ROOT" "$(dirname "$LLMEVAL_REPORT_MD")"

if ! python3 -c 'import lm_eval' >/dev/null 2>&1; then
  echo "=== Install lm-evaluation-harness into current venv ==="
  python3 -m pip install -U "lm_eval[hf]"
fi

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
    port_log="${LLMEVAL_LOG_ROOT}/accelerate_${port}.log"
    set +e
    accelerate launch \
      --num_processes "$NUM_GPUS" \
      --num_machines 1 \
      --main_process_port "$port" \
      --mixed_precision bf16 \
      --dynamo_backend no \
      "$@" 2>&1 | tee "$port_log"
    local status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -eq 0 ]; then
      return 0
    fi
    if grep -q "EADDRINUSE" "$port_log"; then
      echo "=== Port ${port} became occupied during launch; retry with next port ==="
      port="$((port + 1))"
      continue
    fi
    echo "=== Failed command log retained at ${port_log} ===" >&2
    return "$status"
  done
}

json_has_tasks() {
  local path="$1"
  local task_csv="$2"
  python3 - "$path" "$task_csv" <<'PY'
import json, math, os, sys
path, task_csv = sys.argv[1], sys.argv[2]
if not os.path.exists(path) or os.path.getsize(path) == 0:
    sys.exit(1)
try:
    payload = json.load(open(path))
except Exception:
    sys.exit(1)
expected = [x.strip() for x in task_csv.split(",") if x.strip()]
rows = payload.get("task_metrics") or []
seen = {str(row.get("task")) for row in rows}
if not all(task in seen for task in expected):
    sys.exit(1)
for key in ("average_acc", "average_acc_norm"):
    if key not in payload or not math.isfinite(float(payload[key])):
        sys.exit(1)
sys.exit(0)
PY
}

best_epoch_from_artifacts() {
  local best_json="$1"
  local metric_dir="$2"
  local prefix="$3"
  local ckpt_dir="$4"
  python3 - "$best_json" "$metric_dir" "$prefix" "$ckpt_dir" <<'PY'
import glob
import json
import os
import re
import sys

best_json, metric_dir, prefix, ckpt_dir = sys.argv[1:5]

def emit(epoch: int) -> None:
    print(f"{int(epoch):03d}")
    raise SystemExit(0)

if os.path.exists(best_json) and os.path.getsize(best_json) > 0:
    payload = json.load(open(best_json))
    epoch = payload.get("best_epoch")
    if epoch is None:
        epoch = (payload.get("best") or {}).get("epoch")
    if epoch is None:
        epoch = payload.get("epoch")
    if epoch is not None:
        emit(epoch)

patterns = [
    os.path.join(metric_dir, f"{prefix}_best_val_epoch*_test.json"),
    os.path.join(metric_dir, f"{prefix}_best_val*_epoch*_test.json"),
]
metric_matches = []
for pattern in patterns:
    for path in glob.glob(pattern):
        m = re.search(r"_epoch(\d+)_test\.json$", os.path.basename(path))
        if m:
            metric_matches.append((("minuniq" in os.path.basename(path)), os.path.basename(path), int(m.group(1))))
if metric_matches:
    metric_matches.sort()
    emit(metric_matches[0][2])

ckpt_matches = []
for path in glob.glob(os.path.join(ckpt_dir, "risk_router_epoch*.pt")):
    m = re.search(r"risk_router_epoch(\d+)\.pt$", os.path.basename(path))
    if m:
        ckpt_matches.append(int(m.group(1)))
if ckpt_matches:
    # Last-epoch fallback only: the validation summary/test JSON should normally
    # exist, but this keeps unattended downstream runs alive on partially written
    # server artifacts.
    emit(max(ckpt_matches))

raise SystemExit(
    "Missing best checkpoint artifacts. Checked:\n"
    f"  best_json={best_json}\n"
    f"  metric_dir={metric_dir}\n"
    f"  ckpt_dir={ckpt_dir}"
)
PY
}

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "Missing required artifact: ${path}" >&2
    exit 2
  fi
}

declare -a JOB_SEEDS=()
declare -a JOB_SLUGS=()
declare -a JOB_METHODS=()
declare -a JOB_LABELS=()
declare -a JOB_EXTRAS=()

append_eval_job() {
  local seed="$1"
  local slug="$2"
  local method="$3"
  local label="$4"
  shift 4
  local extra_args=""
  if [ "$#" -gt 0 ]; then
    printf -v extra_args '%q ' "$@"
  fi
  JOB_SEEDS+=("$seed")
  JOB_SLUGS+=("$slug")
  JOB_METHODS+=("$method")
  JOB_LABELS+=("$label")
  JOB_EXTRAS+=("$extra_args")
}

eval_method() {
  local seed="$1"
  local slug="$2"
  local method="$3"
  local label="$4"
  shift 4
  local seed_dir="${LLMEVAL_METRIC_ROOT}/seed${seed}"
  local out="${seed_dir}/${slug}.json"
  mkdir -p "$seed_dir"
  if json_has_tasks "$out" "$LLMEVAL_TASKS"; then
    echo "=== Reuse downstream metric: ${out} ==="
    return 0
  fi
  if [ "$LLMEVAL_RUNNER_BACKEND" = "single_gpu_pool" ]; then
    append_eval_job "$seed" "$slug" "$method" "$label" "$@"
    echo "=== Queue downstream metric: seed=${seed} slug=${slug} ==="
    return 0
  fi
  local run_name="${LLMEVAL_RUN_ID}_seed${seed}_${slug}"
  run_accelerate ./eval_lm_eval_harness_opal.py eval \
    --model "$LLMEVAL_MODEL_PATH" \
    --tasks "$LLMEVAL_TASKS" \
    --method "$method" \
    --method_label "$label" \
    --skip_rate "$LLMEVAL_SKIP_RATE" \
    --skip_count "$LLMEVAL_SKIP_COUNT" \
    --protected_head "$LLMEVAL_PROTECTED_HEAD" \
    --protected_tail "$LLMEVAL_PROTECTED_TAIL" \
    --router_prefix_tokens "$LLMEVAL_ROUTER_PREFIX_TOKENS" \
    --max_length "$LLMEVAL_MAX_LENGTH" \
    --batch_size "$LLMEVAL_BATCH_SIZE" \
    --dtype "$LLMEVAL_DTYPE" \
    --compensation_mode "$LLMEVAL_COMPENSATION_MODE" \
    --compensation_rank "$LLMEVAL_COMPENSATION_RANK" \
    --compensation_static_gate "$LLMEVAL_COMPENSATION_STATIC_GATE" \
    --prefix_depth "$LLMEVAL_PREFIX_DEPTH" \
    --seed "$seed" \
    --limit "$LLMEVAL_LIMIT" \
    --run_name "$run_name" \
    --output_dir "${LLMEVAL_OUTPUT_ROOT}/rows/${run_name}" \
    --output_json "$out" \
    "$@"
}

run_direct_eval_on_gpu() {
  local gpu="$1"
  local seed="$2"
  local slug="$3"
  local method="$4"
  local label="$5"
  local extra_args="$6"
  local limit_override="${7:-$LLMEVAL_LIMIT}"
  local seed_dir="${LLMEVAL_METRIC_ROOT}/seed${seed}"
  local out="${seed_dir}/${slug}.json"
  local run_name="${LLMEVAL_RUN_ID}_seed${seed}_${slug}"
  local log_path="${LLMEVAL_LOG_ROOT}/gpu${gpu}_seed${seed}_${slug}.log"
  mkdir -p "$seed_dir"
  if json_has_tasks "$out" "$LLMEVAL_TASKS"; then
    echo "=== GPU ${gpu}: reuse downstream metric ${out} ==="
    return 0
  fi
  set --
  if [ -n "$extra_args" ]; then
    eval "set -- ${extra_args}"
  fi
  echo "=== GPU ${gpu}: run seed=${seed} slug=${slug} method=${method} ==="
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" NUM_GPUS=1 python3 ./eval_lm_eval_harness_opal.py eval \
    --model "$LLMEVAL_MODEL_PATH" \
    --tasks "$LLMEVAL_TASKS" \
    --method "$method" \
    --method_label "$label" \
    --skip_rate "$LLMEVAL_SKIP_RATE" \
    --skip_count "$LLMEVAL_SKIP_COUNT" \
    --protected_head "$LLMEVAL_PROTECTED_HEAD" \
    --protected_tail "$LLMEVAL_PROTECTED_TAIL" \
    --router_prefix_tokens "$LLMEVAL_ROUTER_PREFIX_TOKENS" \
    --max_length "$LLMEVAL_MAX_LENGTH" \
    --batch_size "$LLMEVAL_BATCH_SIZE" \
    --dtype "$LLMEVAL_DTYPE" \
    --compensation_mode "$LLMEVAL_COMPENSATION_MODE" \
    --compensation_rank "$LLMEVAL_COMPENSATION_RANK" \
    --compensation_static_gate "$LLMEVAL_COMPENSATION_STATIC_GATE" \
    --prefix_depth "$LLMEVAL_PREFIX_DEPTH" \
    --seed "$seed" \
    --limit "$limit_override" \
    --run_name "$run_name" \
    --output_dir "${LLMEVAL_OUTPUT_ROOT}/rows/${run_name}" \
    --output_json "$out" \
    "$@" 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  if [ "$status" -ne 0 ]; then
    echo "=== GPU ${gpu}: failed seed=${seed} slug=${slug}; log ${log_path} ===" >&2
    return "$status"
  fi
}

run_prefetch_if_needed() {
  if [ "$LLMEVAL_PREFETCH_LIMIT" = "0" ]; then
    return 0
  fi
  local prefetch_dir="${LLMEVAL_OUTPUT_ROOT}/prefetch"
  local prefetch_json="${prefetch_dir}/${LLMEVAL_RUN_ID}_full_limit${LLMEVAL_PREFETCH_LIMIT}.json"
  local prefetch_log="${LLMEVAL_LOG_ROOT}/prefetch_gpu0_limit${LLMEVAL_PREFETCH_LIMIT}.log"
  mkdir -p "$prefetch_dir"
  if json_has_tasks "$prefetch_json" "$LLMEVAL_TASKS"; then
    echo "=== Reuse downstream dataset prefetch: ${prefetch_json} ==="
    return 0
  fi
  echo "=== Prefetch lm-eval datasets with dummy model limit=${LLMEVAL_PREFETCH_LIMIT} ==="
  set +e
  CUDA_VISIBLE_DEVICES="" python3 -m lm_eval \
    --model dummy \
    --tasks "$LLMEVAL_TASKS" \
    --num_fewshot 0 \
    --batch_size 1 \
    --limit "$LLMEVAL_PREFETCH_LIMIT" \
    --output_path "${prefetch_dir}/dummy_prefetch_limit${LLMEVAL_PREFETCH_LIMIT}" \
    2>&1 | tee "$prefetch_log"
  local status="${PIPESTATUS[0]}"
  set -e
  if [ "$status" -ne 0 ]; then
    echo "=== Prefetch failed; log ${prefetch_log} ===" >&2
    return "$status"
  fi
  python3 - "$prefetch_json" "$LLMEVAL_TASKS" "$LLMEVAL_PREFETCH_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

out, task_csv, limit = sys.argv[1:4]
tasks = [x.strip() for x in task_csv.split(",") if x.strip()]
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps({
    "prefetch": "lm_eval_dummy",
    "tasks": tasks,
    "task_metrics": [{"task": task, "acc": 0.0, "acc_norm": 0.0} for task in tasks],
    "average_acc": 0.0,
    "average_acc_norm": 0.0,
    "limit": limit,
}, indent=2) + "\n")
PY
}

run_single_gpu_pool() {
  local visible_csv="$CUDA_VISIBLE_DEVICES"
  local old_ifs="$IFS"
  IFS=','
  read -r -a gpu_list <<< "$visible_csv"
  IFS="$old_ifs"
  local worker_count="$NUM_GPUS"
  if [ "$worker_count" -gt "${#gpu_list[@]}" ]; then
    worker_count="${#gpu_list[@]}"
  fi
  if [ "$worker_count" -lt 1 ]; then
    echo "No GPUs available in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    return 2
  fi
  local job_count="${#JOB_SEEDS[@]}"
  if [ "$job_count" -eq 0 ]; then
    echo "=== No downstream jobs queued ==="
    return 0
  fi
  echo "=== Run ${job_count} downstream jobs with ${worker_count} single-GPU workers ==="
  run_prefetch_if_needed

  worker() {
    local worker_idx="$1"
    local gpu="$2"
    local stride="$3"
    local idx
    for ((idx=worker_idx; idx<job_count; idx+=stride)); do
      run_direct_eval_on_gpu \
        "$gpu" \
        "${JOB_SEEDS[$idx]}" \
        "${JOB_SLUGS[$idx]}" \
        "${JOB_METHODS[$idx]}" \
        "${JOB_LABELS[$idx]}" \
        "${JOB_EXTRAS[$idx]}"
    done
  }

  local pids=()
  local worker_idx
  for ((worker_idx=0; worker_idx<worker_count; worker_idx++)); do
    worker "$worker_idx" "${gpu_list[$worker_idx]}" "$worker_count" &
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "=== At least one downstream worker failed; see ${LLMEVAL_LOG_ROOT}/gpu*_seed*.log ===" >&2
    return 1
  fi
}

echo "=== OPAL downstream lm-eval-style run ==="
echo "REPO_DIR=${REPO_DIR}"
echo "LLMEVAL_MODEL_PATH=${LLMEVAL_MODEL_PATH}"
echo "LLMEVAL_TASKS=${LLMEVAL_TASKS}"
echo "LLMEVAL_SEEDS=${LLMEVAL_SEEDS}"
echo "LLMEVAL_MAX_LENGTH=${LLMEVAL_MAX_LENGTH}"
echo "LLMEVAL_ROUTER_PREFIX_TOKENS=${LLMEVAL_ROUTER_PREFIX_TOKENS}"
echo "LLMEVAL_BATCH_SIZE=${LLMEVAL_BATCH_SIZE}"
echo "LLMEVAL_DTYPE=${LLMEVAL_DTYPE}"
echo "LLMEVAL_SKIP_COUNT=${LLMEVAL_SKIP_COUNT}"
echo "LLMEVAL_OUTPUT_ROOT=${LLMEVAL_OUTPUT_ROOT}"
echo "LLMEVAL_REPORT_MD=${LLMEVAL_REPORT_MD}"
echo "LLMEVAL_RUNNER_BACKEND=${LLMEVAL_RUNNER_BACKEND}"
echo "LLMEVAL_PREFETCH_LIMIT=${LLMEVAL_PREFETCH_LIMIT}"
echo "LLMEVAL_COMPENSATION_MODE=${LLMEVAL_COMPENSATION_MODE}"
echo "LLMEVAL_COMPENSATION_RANK=${LLMEVAL_COMPENSATION_RANK}"
echo "LLMEVAL_COMPENSATION_STATIC_GATE=${LLMEVAL_COMPENSATION_STATIC_GATE}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "Evaluator=lm-evaluation-harness simple_evaluate"
echo "This script consumes existing WikiText routers/artifacts only; it does not rerun WikiText PPL."

for seed in $LLMEVAL_SEEDS; do
  echo "=== Seed ${seed}: resolve existing WikiText artifacts ==="
  label_run_id="wikitext2_${model_tag}_${LLMEVAL_MASK_IMPL_TAG}_seq${LLMEVAL_MAX_LENGTH}_pref${LLMEVAL_ROUTER_PREFIX_TOKENS}_m${LLMEVAL_LABEL_SAMPLES}_seed${seed}_skip${skip_tag}"
  sanity_ckpt_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_sanity/${label_run_id}"
  related_ckpt_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_related/${label_run_id}"
  val_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_val_ckpt"
  val_metric_root="${REPO_DIR}/results/wikitext2_public_lm_sanity/val_ckpt_metrics"

  pudding_ckpt="${related_ckpt_root}/pudding_candidate_quality/candidate_router.pt"
  ig_artifact="${related_ckpt_root}/ig_prefix_k8.pt"
  layerwise_ckpt="${related_ckpt_root}/layerwise_hidden_bce/risk_router.pt"

  raw_best_json="${val_metric_root}/${label_run_id}_raw_valckpt/best_validation_checkpoint.json"
  opal_best_json="${val_metric_root}/${label_run_id}_valckpt/best_validation_checkpoint.json"
  raw_ckpt_dir="${val_root}/${label_run_id}_raw_valckpt/raw_embedding_bce/epoch_checkpoints"
  opal_ckpt_dir="${val_root}/${label_run_id}_valckpt/prefix_hk_raw_attn_bce/epoch_checkpoints"
  raw_epoch="$(best_epoch_from_artifacts "$raw_best_json" "${val_metric_root}/${label_run_id}_raw_valckpt" raw "$raw_ckpt_dir")"
  opal_epoch="$(best_epoch_from_artifacts "$opal_best_json" "${val_metric_root}/${label_run_id}_valckpt" opal "$opal_ckpt_dir")"
  raw_ckpt="${raw_ckpt_dir}/risk_router_epoch${raw_epoch}.pt"
  opal_ckpt="${opal_ckpt_dir}/risk_router_epoch${opal_epoch}.pt"

  require_file "$pudding_ckpt"
  require_file "$ig_artifact"
  require_file "$layerwise_ckpt"
  require_file "$raw_ckpt"
  require_file "$opal_ckpt"
  echo "Seed ${seed}: Raw best epoch ${raw_epoch}; OPAL best epoch ${opal_epoch}"

  eval_method "$seed" "full" "full" "Full"
  eval_method "$seed" "static_ends_heavy" "static" "Static ends_heavy" --static_strategy ends_heavy
  eval_method "$seed" "static_best_on_val" "static" "Static best-on-val" --static_strategy ends_heavy
  eval_method "$seed" "pudding" "candidate_router" "PuDDing-style" --candidate_router_ckpt "$pudding_ckpt"
  eval_method "$seed" "ig" "ig" "IG-style" --ig_artifact "$ig_artifact"
  eval_method "$seed" "layerwise" "router" "layerwise_hidden_router" --risk_router_ckpt "$layerwise_ckpt"
  eval_method "$seed" "raw_best_val" "router" "Raw-SetBCE best-on-val" --risk_router_ckpt "$raw_ckpt" --prefix_depth 0
  eval_method "$seed" "opal_best_val" "router" "OPAL-SetBCE best-on-val" --risk_router_ckpt "$opal_ckpt" --prefix_depth "$LLMEVAL_PREFIX_DEPTH"
done

if [ "$LLMEVAL_RUNNER_BACKEND" = "single_gpu_pool" ]; then
  run_single_gpu_pool
fi

echo "=== Summarize downstream results ==="
python3 ./eval_lm_eval_harness_opal.py summarize \
  --output_root "$LLMEVAL_OUTPUT_ROOT" \
  --output_md "$LLMEVAL_REPORT_MD" \
  --model "$LLMEVAL_MODEL_PATH" \
  --tasks "$LLMEVAL_TASKS" \
  --seeds "$(printf "%s" "$LLMEVAL_SEEDS" | tr ' ' ',')" \
  --max_length "$LLMEVAL_MAX_LENGTH" \
  --router_prefix_tokens "$LLMEVAL_ROUTER_PREFIX_TOKENS" \
  --skip_rate "$LLMEVAL_SKIP_RATE" \
  --skip_count "$LLMEVAL_SKIP_COUNT" \
  --protected_head "$LLMEVAL_PROTECTED_HEAD" \
  --protected_tail "$LLMEVAL_PROTECTED_TAIL" \
  --compensation_mode "$LLMEVAL_COMPENSATION_MODE" \
  --compensation_rank "$LLMEVAL_COMPENSATION_RANK" \
  --compensation_static_gate "$LLMEVAL_COMPENSATION_STATIC_GATE" \
  --date "$(date +%F)"

echo "=== Done: downstream lm-eval-style run ==="
echo "Report: ${LLMEVAL_REPORT_MD}"
