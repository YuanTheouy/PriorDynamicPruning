#!/usr/bin/env bash
set -euo pipefail

# WikiText-2 fairness run for a shared learned low-rank compensation adapter.
# The adapter is a single seed-level repair plugin shared by all skip methods:
# h_out = h_in + B_l A_l RMSNorm(h_in). Base LLM stays frozen.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export WIKITEXT_BASE_PORT="${WIKITEXT_BASE_PORT:-59400}"
export NEXT_PORT="$WIKITEXT_BASE_PORT"

export PATH="/root/venvs/planrec/bin:${HOME}/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export WIKITEXT_MODEL_PATH="${WIKITEXT_MODEL_PATH:-/workspace/ckpts/Qwen2.5-1.5B}"
export WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
export WIKITEXT_SEED="${WIKITEXT_SEED:-42}"
export WIKITEXT_LABEL_SAMPLES="${WIKITEXT_LABEL_SAMPLES:-2000}"
export WIKITEXT_TRAIN_WINDOWS="${WIKITEXT_TRAIN_WINDOWS:-2000}"
export WIKITEXT_EVAL_WINDOWS="${WIKITEXT_EVAL_WINDOWS:-512}"
export WIKITEXT_SEQ_LEN="${WIKITEXT_SEQ_LEN:-1024}"
export WIKITEXT_ROUTER_PREFIX_TOKENS="${WIKITEXT_ROUTER_PREFIX_TOKENS:-256}"
export WIKITEXT_SKIP_RATE="${WIKITEXT_SKIP_RATE:-0.25}"
export WIKITEXT_SKIP_COUNT="${WIKITEXT_SKIP_COUNT:-7}"
export WIKITEXT_PROTECTED_HEAD="${WIKITEXT_PROTECTED_HEAD:-4}"
export WIKITEXT_PROTECTED_TAIL="${WIKITEXT_PROTECTED_TAIL:-2}"
export WIKITEXT_PREFIX_DEPTH="${WIKITEXT_PREFIX_DEPTH:-4}"
export WIKITEXT_LABEL_SEARCH="${WIKITEXT_LABEL_SEARCH:-greedy}"
export WIKITEXT_LABEL_BEAM_WIDTH="${WIKITEXT_LABEL_BEAM_WIDTH:-1}"
export WIKITEXT_PRECISION="${WIKITEXT_PRECISION:-bf16}"
export WIKITEXT_EVAL_BATCH_SIZE="${WIKITEXT_EVAL_BATCH_SIZE:-1}"
export WIKITEXT_COMP_TRAIN_BATCH_SIZE="${WIKITEXT_COMP_TRAIN_BATCH_SIZE:-1}"
export WIKITEXT_COMP_EPOCHS="${WIKITEXT_COMP_EPOCHS:-1}"
export WIKITEXT_COMP_LR="${WIKITEXT_COMP_LR:-1e-4}"
export WIKITEXT_COMP_RANK="${WIKITEXT_COMP_RANK:-16}"
export WIKITEXT_COMP_STATIC_GATE="${WIKITEXT_COMP_STATIC_GATE:-1.0}"
export WIKITEXT_COMP_TRAIN_GPU="${WIKITEXT_COMP_TRAIN_GPU:-0}"
export WIKITEXT_MASK_IMPL_TAG="${WIKITEXT_MASK_IMPL_TAG:-maskcfg}"
export WIKITEXT_DATASET_CACHE_DIR="${WIKITEXT_DATASET_CACHE_DIR:-}"

export SHARED_COMP_OUTPUT_ROOT="${SHARED_COMP_OUTPUT_ROOT:-${REPO_DIR}/results/wikitext2_shared_comp_fairness/rank${WIKITEXT_COMP_RANK}}"
export SHARED_COMP_REPORT_MD="${SHARED_COMP_REPORT_MD:-${REPO_DIR}/docs/SHARED_COMPENSATION_FAIRNESS_RESULTS.md}"
export SHARED_COMP_RELATED_MD="${SHARED_COMP_RELATED_MD:-${REPO_DIR}/docs/WIKITEXT2_PUBLIC_LM_RELATED_RESULTS.md}"

cd "$REPO_DIR"

mkdir -p "$SHARED_COMP_OUTPUT_ROOT" "$(dirname "$SHARED_COMP_REPORT_MD")"

model_tag="$(basename "$WIKITEXT_MODEL_PATH" | tr ' ./:' '____')"
skip_tag="$(printf "%s" "$WIKITEXT_SKIP_RATE" | tr "." "p")"
WIKITEXT_RESOLVED_SKIP_COUNT="${WIKITEXT_RESOLVED_SKIP_COUNT:-$(
python3 - "$WIKITEXT_MODEL_PATH" "$WIKITEXT_SKIP_RATE" "$WIKITEXT_SKIP_COUNT" <<'PY'
import sys

from transformers import AutoConfig

model_path, skip_rate_raw, skip_count_raw = sys.argv[1:]
skip_count = int(skip_count_raw)
if skip_count > 0:
    print(skip_count)
    raise SystemExit(0)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
num_layers = getattr(config, "num_hidden_layers", None)
if num_layers is None and getattr(config, "text_config", None) is not None:
    num_layers = getattr(config.text_config, "num_hidden_layers", None)
if num_layers is None:
    raise SystemExit(f"Could not resolve num_hidden_layers from {model_path}")
print(max(0, min(int(num_layers), int(round(int(num_layers) * float(skip_rate_raw))))))
PY
)}"
skip_budget_tag="_K${WIKITEXT_RESOLVED_SKIP_COUNT}"
label_search_run_tag=""
if [ "$WIKITEXT_LABEL_SEARCH" = "beam" ]; then
  label_search_run_tag="_beam${WIKITEXT_LABEL_BEAM_WIDTH}"
fi
default_label_run_id="wikitext2_${model_tag}_${WIKITEXT_MASK_IMPL_TAG}_seq${WIKITEXT_SEQ_LEN}_pref${WIKITEXT_ROUTER_PREFIX_TOKENS}_m${WIKITEXT_LABEL_SAMPLES}_seed${WIKITEXT_SEED}_skip${skip_tag}${skip_budget_tag}${label_search_run_tag}"
label_run_id="${WIKITEXT_RUN_ID:-${WIKITEXT_LABEL_RUN_ID:-$default_label_run_id}}"

metric_dir="${SHARED_COMP_OUTPUT_ROOT}/metrics/seed${WIKITEXT_SEED}"
adapter_dir="${SHARED_COMP_OUTPUT_ROOT}/adapter/seed${WIKITEXT_SEED}"
log_dir="${SHARED_COMP_OUTPUT_ROOT}/logs/seed${WIKITEXT_SEED}"
spec_json="${SHARED_COMP_OUTPUT_ROOT}/method_specs_seed${WIKITEXT_SEED}.json"
adapter_ckpt="${adapter_dir}/shared_lowrank_adapter.pt"
mkdir -p "$metric_dir" "$adapter_dir" "$log_dir"

sanity_ckpt_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_sanity/${label_run_id}"
related_ckpt_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_related/${label_run_id}"
val_root="${REPO_DIR}/policy_ckpts/wikitext2_public_lm_val_ckpt"
val_metric_root="${REPO_DIR}/results/wikitext2_public_lm_sanity/val_ckpt_metrics"

pudding_ckpt="${related_ckpt_root}/pudding_candidate_quality/candidate_router.pt"
ig_artifact="${WIKITEXT_IG_ARTIFACT:-${related_ckpt_root}/ig_prefix_k${WIKITEXT_IG_CLUSTERS:-8}.pt}"
layerwise_ckpt="${related_ckpt_root}/layerwise_hidden_bce/risk_router.pt"
raw_best_json="${val_metric_root}/${label_run_id}_raw_valckpt/best_validation_checkpoint.json"
opal_best_json="${val_metric_root}/${label_run_id}_valckpt/best_validation_checkpoint.json"
raw_ckpt_dir="${val_root}/${label_run_id}_raw_valckpt/raw_embedding_bce/epoch_checkpoints"
opal_ckpt_dir="${val_root}/${label_run_id}_valckpt/prefix_hk_raw_attn_bce/epoch_checkpoints"

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
    local port_log="${log_dir}/accelerate_${port}.log"
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
sys.exit(0 if math.isfinite(float(payload.get("nll", "nan"))) else 1)
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
    match = re.search(r"epoch(\d+)_validation", os.path.basename(path))
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

raw_epoch="$(best_epoch_from_artifacts "$raw_best_json" "${val_metric_root}/${label_run_id}_raw_valckpt" raw "$raw_ckpt_dir")"
opal_epoch="$(best_epoch_from_artifacts "$opal_best_json" "${val_metric_root}/${label_run_id}_valckpt" opal "$opal_ckpt_dir")"
raw_best_ckpt="${raw_ckpt_dir}/risk_router_epoch${raw_epoch}.pt"
opal_best_ckpt="${opal_ckpt_dir}/risk_router_epoch${opal_epoch}.pt"

require_file "$pudding_ckpt"
require_file "$ig_artifact"
require_file "$layerwise_ckpt"
require_file "$raw_best_ckpt"
require_file "$opal_best_ckpt"

python3 - "$spec_json" "$pudding_ckpt" "$ig_artifact" "$raw_best_ckpt" "$layerwise_ckpt" "$opal_best_ckpt" "$WIKITEXT_PREFIX_DEPTH" <<'PY'
import json
import sys

spec_json, pudding_ckpt, ig_artifact, raw_best_ckpt, layerwise_ckpt, opal_best_ckpt, prefix_depth = sys.argv[1:]
specs = [
    {
        "slug": "static_best_on_val",
        "method": "static",
        "method_label": "Static best-on-val",
        "static_strategy": "ends_heavy",
    },
    {
        "slug": "pudding",
        "method": "candidate_router",
        "method_label": "PuDDing-style",
        "candidate_router_ckpt": pudding_ckpt,
    },
    {
        "slug": "ig",
        "method": "ig",
        "method_label": "IG-style",
        "ig_artifact": ig_artifact,
    },
    {
        "slug": "raw_best_val",
        "method": "router",
        "method_label": "Raw-SetBCE best-on-val",
        "risk_router_ckpt": raw_best_ckpt,
        "prefix_depth": 0,
    },
    {
        "slug": "layerwise",
        "method": "router",
        "method_label": "layerwise_hidden_router",
        "risk_router_ckpt": layerwise_ckpt,
    },
    {
        "slug": "opal_best_val",
        "method": "router",
        "method_label": "OPAL-SetBCE best-on-val",
        "risk_router_ckpt": opal_best_ckpt,
        "prefix_depth": int(prefix_depth),
    },
]
open(spec_json, "w").write(json.dumps(specs, indent=2) + "\n")
print(f"Wrote method specs to {spec_json}")
PY

echo "=== WikiText-2 shared compensation fairness ==="
echo "label_run_id=${label_run_id}"
echo "raw_best_epoch=${raw_epoch}"
echo "opal_best_epoch=${opal_epoch}"
echo "adapter_ckpt=${adapter_ckpt}"
echo "output_root=${SHARED_COMP_OUTPUT_ROOT}"

eval_method() {
  local slug="$1"
  local suffix="$2"
  local method="$3"
  local label="$4"
  shift 4
  local out="${metric_dir}/${slug}_${suffix}.json"
  if json_usable "$out"; then
    echo "=== Reuse metric: ${out} ==="
    return 0
  fi
  local run_name="wikitext2_shared_comp_seed${WIKITEXT_SEED}_r${WIKITEXT_COMP_RANK}_${slug}_${suffix}"
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
    --seed "$WIKITEXT_SEED" \
    --output_json "$out" \
    "$@"
}

eval_method "full" "no_comp" "full" "Full"

if [ ! -s "$adapter_ckpt" ]; then
  echo "=== Train shared rank-${WIKITEXT_COMP_RANK} low-rank residual adapter on GPU ${WIKITEXT_COMP_TRAIN_GPU} ==="
  CUDA_VISIBLE_DEVICES="$WIKITEXT_COMP_TRAIN_GPU" \
  python3 ./eval_wikitext_opal_ppl.py train_shared_compensation_adapter \
    --teacher_model "$WIKITEXT_MODEL_PATH" \
    --split train \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --dataset_cache_dir "$WIKITEXT_DATASET_CACHE_DIR" \
    --seq_len "$WIKITEXT_SEQ_LEN" \
    --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
    --method_specs_json "$spec_json" \
    --train_windows "$WIKITEXT_TRAIN_WINDOWS" \
    --sample_strategy random \
    --sample_seed "$WIKITEXT_SEED" \
    --skip_rate "$WIKITEXT_SKIP_RATE" \
    --skip_count "$WIKITEXT_SKIP_COUNT" \
    --protected_head "$WIKITEXT_PROTECTED_HEAD" \
    --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
    --batch_size "$WIKITEXT_COMP_TRAIN_BATCH_SIZE" \
    --epochs "$WIKITEXT_COMP_EPOCHS" \
    --lr "$WIKITEXT_COMP_LR" \
    --compensation_rank "$WIKITEXT_COMP_RANK" \
    --compensation_static_gate "$WIKITEXT_COMP_STATIC_GATE" \
    --precision "$WIKITEXT_PRECISION" \
    --seed "$WIKITEXT_SEED" \
    --output_dir "$adapter_dir" 2>&1 | tee "${log_dir}/train_shared_adapter.log"
else
  echo "=== Reuse shared adapter: ${adapter_ckpt} ==="
fi

require_file "$adapter_ckpt"

eval_method "static_best_on_val" "no_comp" "static" "Static best-on-val" --static_strategy ends_heavy
eval_method "pudding" "no_comp" "candidate_router" "PuDDing-style" --candidate_router_ckpt "$pudding_ckpt"
eval_method "ig" "no_comp" "ig" "IG-style" --ig_artifact "$ig_artifact"
eval_method "raw_best_val" "no_comp" "router" "Raw-SetBCE best-on-val" --risk_router_ckpt "$raw_best_ckpt" --prefix_depth 0
eval_method "layerwise" "no_comp" "router" "layerwise_hidden_router" --risk_router_ckpt "$layerwise_ckpt"
eval_method "opal_best_val" "no_comp" "router" "OPAL-SetBCE best-on-val" --risk_router_ckpt "$opal_best_ckpt" --prefix_depth "$WIKITEXT_PREFIX_DEPTH"

comp_args=(
  --compensation_mode learned_lowrank
  --compensation_rank "$WIKITEXT_COMP_RANK"
  --compensation_static_gate "$WIKITEXT_COMP_STATIC_GATE"
  --compensation_adapter_ckpt "$adapter_ckpt"
)

eval_method "static_best_on_val" "same_comp" "static" "Static best-on-val +same-comp" --static_strategy ends_heavy "${comp_args[@]}"
eval_method "pudding" "same_comp" "candidate_router" "PuDDing-style +same-comp" --candidate_router_ckpt "$pudding_ckpt" "${comp_args[@]}"
eval_method "ig" "same_comp" "ig" "IG-style +same-comp" --ig_artifact "$ig_artifact" "${comp_args[@]}"
eval_method "raw_best_val" "same_comp" "router" "Raw-SetBCE best-on-val +same-comp" --risk_router_ckpt "$raw_best_ckpt" --prefix_depth 0 "${comp_args[@]}"
eval_method "layerwise" "same_comp" "router" "layerwise_hidden_router +same-comp" --risk_router_ckpt "$layerwise_ckpt" "${comp_args[@]}"
eval_method "opal_best_val" "same_comp" "router" "OPAL-SetBCE best-on-val +same-comp" --risk_router_ckpt "$opal_best_ckpt" --prefix_depth "$WIKITEXT_PREFIX_DEPTH" "${comp_args[@]}"

echo "=== Summarize shared compensation fairness ==="
python3 ./summarize_wikitext2_shared_comp_fairness.py \
  --output_root "$SHARED_COMP_OUTPUT_ROOT" \
  --output_md "$SHARED_COMP_REPORT_MD" \
  --related_md "$SHARED_COMP_RELATED_MD" \
  --seed "$WIKITEXT_SEED" \
  --model "$WIKITEXT_MODEL_PATH" \
  --seq_len "$WIKITEXT_SEQ_LEN" \
  --router_prefix_tokens "$WIKITEXT_ROUTER_PREFIX_TOKENS" \
  --train_windows "$WIKITEXT_TRAIN_WINDOWS" \
  --eval_windows "$WIKITEXT_EVAL_WINDOWS" \
  --skip_rate "$WIKITEXT_SKIP_RATE" \
  --skip_count "$WIKITEXT_SKIP_COUNT" \
  --protected_head "$WIKITEXT_PROTECTED_HEAD" \
  --protected_tail "$WIKITEXT_PROTECTED_TAIL" \
  --rank "$WIKITEXT_COMP_RANK" \
  --epochs "$WIKITEXT_COMP_EPOCHS" \
  --date "$(date +%F)"

echo "=== Done: WikiText-2 shared compensation fairness ==="
echo "Report: ${SHARED_COMP_REPORT_MD}"
echo "Related report updated: ${SHARED_COMP_RELATED_MD}"
