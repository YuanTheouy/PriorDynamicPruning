#!/usr/bin/env bash
set -euo pipefail

# Overnight second-stage run:
#   1. WikiText-2 PPL with the same compensation module applied to every skip method.
#   2. lm-eval downstream with the same compensation module applied to every skip method.
#
# This script consumes existing WikiText routers/checkpoints only. It does not
# rebuild labels, retrain routers, or rerun no-comp WikiText PPL.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export COMP_BASE_PORT="${COMP_BASE_PORT:-59200}"
export NEXT_PORT="$COMP_BASE_PORT"

export PATH="/root/venvs/planrec/bin:${HOME}/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export COMPENSATION_MODE="${COMPENSATION_MODE:-ghost}"
export COMPENSATION_RANK="${COMPENSATION_RANK:-64}"
export COMPENSATION_STATIC_GATE="${COMPENSATION_STATIC_GATE:-1.0}"
export COMP_TAG="${COMP_TAG:-${COMPENSATION_MODE}_r${COMPENSATION_RANK}_g${COMPENSATION_STATIC_GATE}}"
export COMP_TAG="${COMP_TAG//./p}"
export COMP_FORCE_OUTPUT_ROOTS="${COMP_FORCE_OUTPUT_ROOTS:-1}"
export COMP_REQUIRE_FINAL_ROUTERS="${COMP_REQUIRE_FINAL_ROUTERS:-0}"

export WIKITEXT_MODEL_PATH="${WIKITEXT_MODEL_PATH:-/workspace/ckpts/Qwen2.5-1.5B}"
export WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
export WIKITEXT_SEEDS="${WIKITEXT_SEEDS:-42 13 3407}"
export WIKITEXT_LABEL_SAMPLES="${WIKITEXT_LABEL_SAMPLES:-2000}"
export WIKITEXT_EVAL_WINDOWS="${WIKITEXT_EVAL_WINDOWS:-512}"
export WIKITEXT_SEQ_LEN="${WIKITEXT_SEQ_LEN:-1024}"
export WIKITEXT_ROUTER_PREFIX_TOKENS="${WIKITEXT_ROUTER_PREFIX_TOKENS:-256}"
export WIKITEXT_SKIP_RATE="${WIKITEXT_SKIP_RATE:-0.25}"
export WIKITEXT_SKIP_COUNT="${WIKITEXT_SKIP_COUNT:-7}"
export WIKITEXT_PROTECTED_HEAD="${WIKITEXT_PROTECTED_HEAD:-4}"
export WIKITEXT_PROTECTED_TAIL="${WIKITEXT_PROTECTED_TAIL:-2}"
export WIKITEXT_PREFIX_DEPTH="${WIKITEXT_PREFIX_DEPTH:-4}"
export WIKITEXT_PRECISION="${WIKITEXT_PRECISION:-bf16}"
export WIKITEXT_EVAL_BATCH_SIZE="${WIKITEXT_EVAL_BATCH_SIZE:-1}"
export WIKITEXT_DATASET_CACHE_DIR="${WIKITEXT_DATASET_CACHE_DIR:-}"
export WIKITEXT_MASK_IMPL_TAG="${WIKITEXT_MASK_IMPL_TAG:-maskcfg}"
default_wikitext_comp_output_root="${REPO_DIR}/results/wikitext2_compensated_ppl/${COMP_TAG}"
default_wikitext_comp_report_md="${REPO_DIR}/docs/WIKITEXT2_COMPENSATED_PPL_RESULTS.md"
if [ "$COMP_FORCE_OUTPUT_ROOTS" = "1" ]; then
  export WIKITEXT_COMP_OUTPUT_ROOT="$default_wikitext_comp_output_root"
  export WIKITEXT_COMP_REPORT_MD="$default_wikitext_comp_report_md"
else
  export WIKITEXT_COMP_OUTPUT_ROOT="${WIKITEXT_COMP_OUTPUT_ROOT:-$default_wikitext_comp_output_root}"
  export WIKITEXT_COMP_REPORT_MD="${WIKITEXT_COMP_REPORT_MD:-$default_wikitext_comp_report_md}"
fi
export WIKITEXT_COMP_METRIC_ROOT="${WIKITEXT_COMP_OUTPUT_ROOT}/metrics"
export WIKITEXT_COMP_LOG_ROOT="${WIKITEXT_COMP_OUTPUT_ROOT}/logs"

export LLMEVAL_MODEL_PATH="${LLMEVAL_MODEL_PATH:-$WIKITEXT_MODEL_PATH}"
export LLMEVAL_TASKS="${LLMEVAL_TASKS:-piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge}"
export LLMEVAL_SEEDS="${LLMEVAL_SEEDS:-$WIKITEXT_SEEDS}"
export LLMEVAL_MAX_LENGTH="${LLMEVAL_MAX_LENGTH:-$WIKITEXT_SEQ_LEN}"
export LLMEVAL_ROUTER_PREFIX_TOKENS="${LLMEVAL_ROUTER_PREFIX_TOKENS:-$WIKITEXT_ROUTER_PREFIX_TOKENS}"
export LLMEVAL_LABEL_SAMPLES="${LLMEVAL_LABEL_SAMPLES:-$WIKITEXT_LABEL_SAMPLES}"
export LLMEVAL_SKIP_RATE="${LLMEVAL_SKIP_RATE:-$WIKITEXT_SKIP_RATE}"
export LLMEVAL_SKIP_COUNT="${LLMEVAL_SKIP_COUNT:-$WIKITEXT_SKIP_COUNT}"
export LLMEVAL_PROTECTED_HEAD="${LLMEVAL_PROTECTED_HEAD:-$WIKITEXT_PROTECTED_HEAD}"
export LLMEVAL_PROTECTED_TAIL="${LLMEVAL_PROTECTED_TAIL:-$WIKITEXT_PROTECTED_TAIL}"
export LLMEVAL_PREFIX_DEPTH="${LLMEVAL_PREFIX_DEPTH:-$WIKITEXT_PREFIX_DEPTH}"
export LLMEVAL_BATCH_SIZE="${LLMEVAL_BATCH_SIZE:-8}"
export LLMEVAL_DTYPE="${LLMEVAL_DTYPE:-bfloat16}"
export LLMEVAL_BASE_PORT="${LLMEVAL_BASE_PORT:-$((COMP_BASE_PORT + 500))}"
export LLMEVAL_RUNNER_BACKEND="${LLMEVAL_RUNNER_BACKEND:-single_gpu_pool}"
export LLMEVAL_PREFETCH_LIMIT="${LLMEVAL_PREFETCH_LIMIT:-1}"
export LLMEVAL_COMPENSATION_MODE="${LLMEVAL_COMPENSATION_MODE:-$COMPENSATION_MODE}"
export LLMEVAL_COMPENSATION_RANK="${LLMEVAL_COMPENSATION_RANK:-$COMPENSATION_RANK}"
export LLMEVAL_COMPENSATION_STATIC_GATE="${LLMEVAL_COMPENSATION_STATIC_GATE:-$COMPENSATION_STATIC_GATE}"
default_llmeval_output_root="${REPO_DIR}/results/llmeval_downstream_compensated/${COMP_TAG}"
default_llmeval_report_md="${REPO_DIR}/docs/LLMEVAL_COMPENSATED_DOWNSTREAM_RESULTS.md"
if [ "$COMP_FORCE_OUTPUT_ROOTS" = "1" ]; then
  export LLMEVAL_OUTPUT_ROOT="$default_llmeval_output_root"
  export LLMEVAL_REPORT_MD="$default_llmeval_report_md"
else
  export LLMEVAL_OUTPUT_ROOT="${LLMEVAL_OUTPUT_ROOT:-$default_llmeval_output_root}"
  export LLMEVAL_REPORT_MD="${LLMEVAL_REPORT_MD:-$default_llmeval_report_md}"
fi

cd "$REPO_DIR"

mkdir -p "$WIKITEXT_COMP_METRIC_ROOT" "$WIKITEXT_COMP_LOG_ROOT" "$(dirname "$WIKITEXT_COMP_REPORT_MD")" "$(dirname "$LLMEVAL_REPORT_MD")"

model_tag="$(basename "$WIKITEXT_MODEL_PATH" | tr ' ./:' '____')"
skip_tag="$(printf "%s" "$WIKITEXT_SKIP_RATE" | tr "." "p")"

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
    local port_log="${WIKITEXT_COMP_LOG_ROOT}/accelerate_${port}.log"
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
if not math.isfinite(float(payload.get("nll", "nan"))):
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

best_json, metric_dir, prefix, ckpt_dir = sys.argv[1:]

if os.path.exists(best_json) and os.path.getsize(best_json) > 0:
    payload = json.load(open(best_json))
    best = payload.get("best") or {}
    epoch = best.get("epoch")
    ckpt = best.get("checkpoint")
    if epoch is not None:
        epoch = int(epoch)
        candidate = ckpt or os.path.join(ckpt_dir, f"risk_router_epoch{epoch:03d}.pt")
        if os.path.exists(candidate):
            print(f"{epoch:03d}")
            raise SystemExit(0)

rows = []
for path in sorted(glob.glob(os.path.join(metric_dir, f"{prefix}_epoch*_validation.json"))):
    payload = json.load(open(path))
    name = os.path.basename(path)
    match = re.search(r"epoch(\d+)_validation", name)
    if not match:
        continue
    epoch = int(match.group(1))
    ckpt = os.path.join(ckpt_dir, f"risk_router_epoch{epoch:03d}.pt")
    if os.path.exists(ckpt):
        rows.append((float(payload["nll"]), epoch))
if not rows:
    raise SystemExit(f"Missing best checkpoint artifacts. Checked best_json={best_json}, metric_dir={metric_dir}, ckpt_dir={ckpt_dir}")
rows.sort(key=lambda item: (item[0], item[1]))
print(f"{rows[0][1]:03d}")
PY
}

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "Missing required artifact: ${path}" >&2
    exit 2
  fi
}

eval_ppl_method() {
  local seed="$1"
  local slug="$2"
  local method="$3"
  local label="$4"
  shift 4
  local seed_dir="${WIKITEXT_COMP_METRIC_ROOT}/seed${seed}"
  local out="${seed_dir}/${slug}.json"
  mkdir -p "$seed_dir"
  if json_usable "$out"; then
    echo "=== Reuse compensated PPL metric: ${out} ==="
    return 0
  fi
  local run_name="wikitext2_comp_${COMP_TAG}_${model_tag}_${WIKITEXT_MASK_IMPL_TAG}_seq${WIKITEXT_SEQ_LEN}_pref${WIKITEXT_ROUTER_PREFIX_TOKENS}_m${WIKITEXT_LABEL_SAMPLES}_seed${seed}_skip${skip_tag}_${slug}"
  run_accelerate ./eval_wikitext_opal_ppl.py eval \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split test \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
    --method "$method" \
    --method_label "$label" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --run_name "$run_name" \
    --batch_size "$WIKITEXT_EVAL_BATCH_SIZE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$seed" \
    --compensation_mode "$COMPENSATION_MODE" \
    --compensation_rank "$COMPENSATION_RANK" \
    --compensation_static_gate "$COMPENSATION_STATIC_GATE" \
    --output_json "$out" \
    "$@"
}

echo "=== Compensated PPL + downstream overnight run ==="
echo "REPO_DIR=${REPO_DIR}"
echo "COMPENSATION_MODE=${COMPENSATION_MODE}"
echo "COMPENSATION_RANK=${COMPENSATION_RANK}"
echo "COMPENSATION_STATIC_GATE=${COMPENSATION_STATIC_GATE}"
echo "WIKITEXT_MODEL_PATH=${WIKITEXT_MODEL_PATH}"
echo "WIKITEXT_SEEDS=${WIKITEXT_SEEDS}"
echo "WIKITEXT_COMP_OUTPUT_ROOT=${WIKITEXT_COMP_OUTPUT_ROOT}"
echo "WIKITEXT_COMP_REPORT_MD=${WIKITEXT_COMP_REPORT_MD}"
echo "LLMEVAL_OUTPUT_ROOT=${LLMEVAL_OUTPUT_ROOT}"
echo "LLMEVAL_REPORT_MD=${LLMEVAL_REPORT_MD}"
echo "COMP_FORCE_OUTPUT_ROOTS=${COMP_FORCE_OUTPUT_ROOTS}"
echo "COMP_REQUIRE_FINAL_ROUTERS=${COMP_REQUIRE_FINAL_ROUTERS}"

for seed in $WIKITEXT_SEEDS; do
  echo "=== WikiText compensated PPL seed ${seed}: resolve existing artifacts ==="
  label_run_id="wikitext2_${model_tag}_${WIKITEXT_MASK_IMPL_TAG}_seq${WIKITEXT_SEQ_LEN}_pref${WIKITEXT_ROUTER_PREFIX_TOKENS}_m${WIKITEXT_LABEL_SAMPLES}_seed${seed}_skip${skip_tag}"
  sanity_ckpt_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_sanity/${label_run_id}"
  related_ckpt_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_related/${label_run_id}"
  val_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_val_ckpt"
  val_metric_root="${REPO_DIR}/results/wikitext2_public_lm_sanity/val_ckpt_metrics"

  raw_final_ckpt="${sanity_ckpt_root}/raw_embedding_bce/risk_router.pt"
  opal_final_ckpt="${sanity_ckpt_root}/prefix_hk_raw_attn_bce/risk_router.pt"
  pudding_ckpt="${related_ckpt_root}/pudding_candidate_quality/candidate_router.pt"
  ig_artifact="${related_ckpt_root}/ig_prefix_k8.pt"
  layerwise_ckpt="${related_ckpt_root}/layerwise_hidden_bce/risk_router.pt"

  raw_best_json="${val_metric_root}/${label_run_id}_raw_valckpt/best_validation_checkpoint.json"
  opal_best_json="${val_metric_root}/${label_run_id}_valckpt/best_validation_checkpoint.json"
  raw_ckpt_dir="${val_root}/${label_run_id}_raw_valckpt/raw_embedding_bce/epoch_checkpoints"
  opal_ckpt_dir="${val_root}/${label_run_id}_valckpt/prefix_hk_raw_attn_bce/epoch_checkpoints"
  raw_epoch="$(best_epoch_from_artifacts "$raw_best_json" "${val_metric_root}/${label_run_id}_raw_valckpt" raw "$raw_ckpt_dir")"
  opal_epoch="$(best_epoch_from_artifacts "$opal_best_json" "${val_metric_root}/${label_run_id}_valckpt" opal "$opal_ckpt_dir")"
  raw_best_ckpt="${raw_ckpt_dir}/risk_router_epoch${raw_epoch}.pt"
  opal_best_ckpt="${opal_ckpt_dir}/risk_router_epoch${opal_epoch}.pt"

  require_file "$pudding_ckpt"
  require_file "$ig_artifact"
  require_file "$layerwise_ckpt"
  require_file "$raw_best_ckpt"
  require_file "$opal_best_ckpt"
  if [ "$COMP_REQUIRE_FINAL_ROUTERS" = "1" ]; then
    require_file "$raw_final_ckpt"
    require_file "$opal_final_ckpt"
  fi
  echo "Seed ${seed}: Raw best epoch ${raw_epoch}; OPAL best epoch ${opal_epoch}"

  eval_ppl_method "$seed" "full" "full" "Full"
  eval_ppl_method "$seed" "static_ends_heavy" "static" "Static ends_heavy" --static_strategy ends_heavy
  eval_ppl_method "$seed" "static_best_on_val" "static" "Static best-on-val" --static_strategy ends_heavy
  eval_ppl_method "$seed" "pudding" "candidate_router" "PuDDing-style" --candidate_router_ckpt "$pudding_ckpt"
  eval_ppl_method "$seed" "ig" "ig" "IG-style" --ig_artifact "$ig_artifact"
  eval_ppl_method "$seed" "layerwise" "router" "layerwise_hidden_router" --risk_router_ckpt "$layerwise_ckpt"
  if [ -s "$raw_final_ckpt" ]; then
    eval_ppl_method "$seed" "raw_final" "router" "Raw-SetBCE final epoch" --risk_router_ckpt "$raw_final_ckpt" --prefix_depth 0
  else
    echo "=== Skip optional Raw final PPL; missing ${raw_final_ckpt} ==="
  fi
  eval_ppl_method "$seed" "raw_best_val" "router" "Raw-SetBCE best-on-val" --risk_router_ckpt "$raw_best_ckpt" --prefix_depth 0
  if [ -s "$opal_final_ckpt" ]; then
    eval_ppl_method "$seed" "opal_final" "router" "OPAL-SetBCE final epoch" --risk_router_ckpt "$opal_final_ckpt" --prefix_depth "$WIKITEXT_PREFIX_DEPTH"
  else
    echo "=== Skip optional OPAL final PPL; missing ${opal_final_ckpt} ==="
  fi
  eval_ppl_method "$seed" "opal_best_val" "router" "OPAL-SetBCE best-on-val" --risk_router_ckpt "$opal_best_ckpt" --prefix_depth "$WIKITEXT_PREFIX_DEPTH"
done

echo "=== Summarize compensated WikiText PPL ==="
python3 ./summarize_wikitext2_compensated_ppl_results.py \
  --output_root "$WIKITEXT_COMP_OUTPUT_ROOT" \
  --output_md "$WIKITEXT_COMP_REPORT_MD" \
  --model "$WIKITEXT_MODEL_PATH" \
  --seeds "$(printf "%s" "$WIKITEXT_SEEDS" | tr ' ' ',')" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --label_samples "$WIKITEXT_LABEL_SAMPLES" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --skip_rate "$WIKITEXT_SKIP_RATE" \
  --skip_count "$WIKITEXT_SKIP_COUNT" \
  --protected_head "$WIKITEXT_PROTECTED_HEAD" \
  --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
  --compensation_mode "$COMPENSATION_MODE" \
  --compensation_rank "$COMPENSATION_RANK" \
  --compensation_static_gate "$COMPENSATION_STATIC_GATE" \
  --date "$(date +%F)"

echo "=== Run compensated downstream ==="
bash ./run_llmeval_downstream_gpu01234567.sh

echo "=== Done: compensated PPL + downstream overnight run ==="
echo "WikiText PPL report: ${WIKITEXT_COMP_REPORT_MD}"
echo "Downstream report: ${LLMEVAL_REPORT_MD}"
