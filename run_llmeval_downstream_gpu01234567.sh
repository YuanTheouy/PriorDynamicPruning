#!/usr/bin/env bash
set -euo pipefail

# Unattended no-comp downstream evaluation for OPAL layer skipping via
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

best_epoch_from_json() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    raise SystemExit(f"Missing best checkpoint JSON: {path}")
payload = json.load(open(path))
epoch = payload.get("best_epoch")
if epoch is None:
    raise SystemExit(f"best_epoch missing from {path}")
print(f"{int(epoch):03d}")
PY
}

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "Missing required artifact: ${path}" >&2
    exit 2
  fi
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
    --prefix_depth "$LLMEVAL_PREFIX_DEPTH" \
    --seed "$seed" \
    --limit "$LLMEVAL_LIMIT" \
    --run_name "$run_name" \
    --output_dir "${LLMEVAL_OUTPUT_ROOT}/rows/${run_name}" \
    --output_json "$out" \
    "$@"
}

echo "=== OPAL downstream lm-eval-style no-comp run ==="
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
  raw_epoch="$(best_epoch_from_json "$raw_best_json")"
  opal_epoch="$(best_epoch_from_json "$opal_best_json")"
  raw_ckpt="${val_root}/${label_run_id}_raw_valckpt/raw_embedding_bce/epoch_checkpoints/risk_router_epoch${raw_epoch}.pt"
  opal_ckpt="${val_root}/${label_run_id}_valckpt/prefix_hk_raw_attn_bce/epoch_checkpoints/risk_router_epoch${opal_epoch}.pt"

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
  --date "$(date +%F)"

echo "=== Done: downstream no-comp run ==="
echo "Report: ${LLMEVAL_REPORT_MD}"
