#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

WIKITEXT_DATASET_DISK_PATH="${WIKITEXT_DATASET_DISK_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
SPEED_ROOT="${WIKITEXT_SPEED_ROOT:-${REPO_DIR}/results/wikitext2_layer_skip_speed}"
REPORT_MD="${WIKITEXT_SPEED_REPORT_MD:-${REPO_DIR}/docs/WIKITEXT2_LAYER_SKIP_SPEED_RESULTS.md}"
NUM_WINDOWS="${WIKITEXT_SPEED_WINDOWS:-96}"
BATCH1_WINDOWS="${WIKITEXT_SPEED_BATCH1_WINDOWS:-32}"
BATCH_SIZE="${WIKITEXT_SPEED_BATCH_SIZE:-8}"
PRECISION="${WIKITEXT_PRECISION:-bf16}"

mkdir -p "$SPEED_ROOT"

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "Missing required speed-benchmark artifact: $path" >&2
    return 1
  fi
}

build_methods_json() {
  local output="$1"
  shift
  python3 - "$output" "$@" <<'PY'
import json
import sys

output = sys.argv[1]
items = []
for raw in sys.argv[2:]:
    slug, label, method, metric_json = raw.split("::", 3)
    items.append(
        {
            "slug": slug,
            "label": label,
            "method": method,
            "metric_json": metric_json,
        }
    )
with open(output, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2)
print(output)
PY
}

run_qwen3_seed42_speed() {
  local model="/workspace/Models/Qwen3-8B"
  local run="wikitext2_Qwen3-8B_maskcfg_auto_qwen3_k9_pref1024_seq1536_pref1024_m2000_seed42_skip0p25_K9"
  local metric_dir="results/wikitext2_public_lm_sanity/metrics/${run}"
  local related_dir="results/wikitext2_public_lm_sanity/related_metrics/${run}"
  local val_dir="results/wikitext2_public_lm_sanity/val_ckpt_metrics"
  local methods_json="${SPEED_ROOT}/qwen3_seed42_methods.json"
  local output_json="${SPEED_ROOT}/qwen3_seed42_speed.json"
  local output_md="${SPEED_ROOT}/qwen3_seed42_speed.md"

  require_file "${metric_dir}/full_test.json"
  require_file "${metric_dir}/static_best_on_val_c6_test.json"
  require_file "${val_dir}/${run}_raw_valckpt/raw_best_val_epoch003_test.json"
  require_file "${val_dir}/${run}_valckpt/opal_best_val_epoch001_test.json"
  require_file "${related_dir}/pudding_style_test.json"
  require_file "${related_dir}/ig_style_test.json"
  require_file "${related_dir}/layerwise_hidden_router_test.json"

  build_methods_json "$methods_json" \
    "full::Full::full::${metric_dir}/full_test.json" \
    "static_best::Static best-on-val::static::${metric_dir}/static_best_on_val_c6_test.json" \
    "pudding::PuDDing-style::candidate_router::${related_dir}/pudding_style_test.json" \
    "ig::IG-style::ig::${related_dir}/ig_style_test.json" \
    "layerwise::layerwise_hidden_router::router::${related_dir}/layerwise_hidden_router_test.json" \
    "raw_bce_best::Raw BCE best-on-val::router::${val_dir}/${run}_raw_valckpt/raw_best_val_epoch003_test.json" \
    "opal_bce_best::OPAL BCE best-on-val::router::${val_dir}/${run}_valckpt/opal_best_val_epoch001_test.json"

  python3 ./benchmark_wikitext_layer_skip_speed.py \
    --model "$model" \
    --model_label Qwen3-8B \
    --run_label "$run" \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --seq_len 1536 \
    --router_prefix_tokens 1024 \
    --num_windows "$NUM_WINDOWS" \
    --batch1_windows "$BATCH1_WINDOWS" \
    --batch_size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --skip_rate 0.25 \
    --skip_count 9 \
    --protected_head 4 \
    --protected_tail 2 \
    --seed 42 \
    --methods_json_file "$methods_json" \
    --output_json "$output_json" \
    --output_md "$output_md"
}

run_llama_seed42_speed() {
  local model="/workspace/Models/Llama-3.1-8B-Instruct"
  local run="wikitext2_Llama3_1-8B-Instruct_maskcfg_llama31_k8_pref1024_seq1536_pref1024_m1572_seed42_skip0p25_K8"
  local metric_dir="results/wikitext2_public_lm_sanity/metrics/${run}"
  local related_dir="results/wikitext2_public_lm_sanity/related_metrics/${run}"
  local val_dir="results/wikitext2_public_lm_sanity/val_ckpt_metrics"
  local methods_json="${SPEED_ROOT}/llama31_seed42_methods.json"
  local output_json="${SPEED_ROOT}/llama31_seed42_speed.json"
  local output_md="${SPEED_ROOT}/llama31_seed42_speed.md"

  require_file "${metric_dir}/full_test.json"
  require_file "${metric_dir}/static_best_on_val_c6_test.json"
  require_file "${val_dir}/${run}_raw_valckpt/raw_best_val_epoch006_test.json"
  require_file "${val_dir}/${run}_valckpt/opal_best_val_epoch017_test.json"
  require_file "${related_dir}/pudding_style_test.json"
  require_file "${related_dir}/ig_style_test.json"
  require_file "${related_dir}/layerwise_hidden_router_test.json"

  build_methods_json "$methods_json" \
    "full::Full::full::${metric_dir}/full_test.json" \
    "static_best::Static best-on-val::static::${metric_dir}/static_best_on_val_c6_test.json" \
    "pudding::PuDDing-style::candidate_router::${related_dir}/pudding_style_test.json" \
    "ig::IG-style::ig::${related_dir}/ig_style_test.json" \
    "layerwise::layerwise_hidden_router::router::${related_dir}/layerwise_hidden_router_test.json" \
    "raw_bce_best::Raw BCE best-on-val::router::${val_dir}/${run}_raw_valckpt/raw_best_val_epoch006_test.json" \
    "opal_bce_best::OPAL BCE best-on-val::router::${val_dir}/${run}_valckpt/opal_best_val_epoch017_test.json"

  if [ -s "${val_dir}/${run}_opal_exactk_valckpt/opal_best_val_epoch010_test.json" ]; then
    python3 - "$methods_json" "${val_dir}/${run}_opal_exactk_valckpt/opal_best_val_epoch010_test.json" <<'PY'
import json
import sys
path, metric = sys.argv[1], sys.argv[2]
rows = json.load(open(path))
rows.append(
    {
        "slug": "opal_exactk_best",
        "label": "OPAL Exact-K best-on-val",
        "method": "router",
        "metric_json": metric,
    }
)
json.dump(rows, open(path, "w"), indent=2)
PY
  fi

  python3 ./benchmark_wikitext_layer_skip_speed.py \
    --model "$model" \
    --model_label Llama3.1-8B-Instruct \
    --run_label "$run" \
    --dataset_disk_path "$WIKITEXT_DATASET_DISK_PATH" \
    --seq_len 1536 \
    --router_prefix_tokens 1024 \
    --num_windows "$NUM_WINDOWS" \
    --batch1_windows "$BATCH1_WINDOWS" \
    --batch_size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --skip_rate 0.25 \
    --skip_count 8 \
    --protected_head 4 \
    --protected_tail 2 \
    --seed 42 \
    --methods_json_file "$methods_json" \
    --output_json "$output_json" \
    --output_md "$output_md"
}

echo "=== WikiText-2 layer-skip speed benchmark ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_WINDOWS=${NUM_WINDOWS}"
echo "BATCH1_WINDOWS=${BATCH1_WINDOWS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "REPORT_MD=${REPORT_MD}"

run_qwen3_seed42_speed
run_llama_seed42_speed

python3 ./summarize_wikitext_layer_skip_speed_results.py \
  --input_json "${SPEED_ROOT}/qwen3_seed42_speed.json" \
  --input_json "${SPEED_ROOT}/llama31_seed42_speed.json" \
  --output_md "$REPORT_MD"

echo "=== Done WikiText-2 speed benchmark ==="
echo "Report: ${REPORT_MD}"
