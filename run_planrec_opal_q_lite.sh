#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${REPO_ROOT}/transformers/src:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}
CATEGORY=${CATEGORY:-Office_Products}
OUTPUT_DIR=${OUTPUT_DIR:-./results/planrec_experiments}
BATCH_SIZE=${BATCH_SIZE:-1}
TOP_K_ITEMS=${TOP_K_ITEMS:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_GPUS=${NUM_GPUS:-1}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16}
WARMUP_BATCHES=${WARMUP_BATCHES:-0}
TIMED_BATCHES=${TIMED_BATCHES:-0}
MAX_BATCHES=${MAX_BATCHES:-2}
PREFIX_DEPTH=${PREFIX_DEPTH:-4}
SKIP_RATE=${SKIP_RATE:-0.25}
OPAL_STAGE=${OPAL_STAGE:-1}
POLICY_CKPT=${POLICY_CKPT:-}
POLICY_CKPT_DIR=${POLICY_CKPT_DIR:-./policy_ckpts/opal_prefix${PREFIX_DEPTH}/${CATEGORY}}
ORACLE_CACHE=${ORACLE_CACHE:-}
ALLOW_FALLBACK_ROUTER=${ALLOW_FALLBACK_ROUTER:-0}
ALLOW_LEGACY_POLICY_CKPT=${ALLOW_LEGACY_POLICY_CKPT:-0}
MAX_CONSECUTIVE_SKIPS=${MAX_CONSECUTIVE_SKIPS:-0}
NUM_STAGES=${NUM_STAGES:-1}
MIN_KEEP_PER_STAGE=${MIN_KEEP_PER_STAGE:-0}

test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing MODEL_PATH: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$test_file" || ! -f "$info_file" ]]; then
  echo "Missing test/info files for CATEGORY=$CATEGORY"
  exit 1
fi

if [[ -z "$POLICY_CKPT" && -d "$POLICY_CKPT_DIR" ]]; then
  POLICY_CKPT=$(ls -v "${POLICY_CKPT_DIR}"/policy_epoch_*.pt 2>/dev/null | tail -1)
fi

router_flags=()
if [[ -n "$POLICY_CKPT" ]]; then
  router_flags+=(--policy_ckpt "$POLICY_CKPT")
elif [[ "$ALLOW_FALLBACK_ROUTER" == "1" ]]; then
  router_flags+=(--allow_fallback_router)
else
  echo "POLICY_CKPT is required. Set ALLOW_FALLBACK_ROUTER=1 only for smoke/debug."
  exit 1
fi
if [[ "$ALLOW_LEGACY_POLICY_CKPT" == "1" ]]; then
  router_flags+=(--allow_legacy_policy_ckpt)
fi

oracle_flags=()
if [[ -n "$ORACLE_CACHE" ]]; then
  oracle_flags+=(--oracle_cache "$ORACLE_CACHE")
fi

accelerate launch --num_processes "$NUM_GPUS" ./eval_planrec_opal.py \
  --method opal_q \
  --opal_stage "$OPAL_STAGE" \
  --teacher_model "$MODEL_PATH" \
  "${router_flags[@]}" \
  "${oracle_flags[@]}" \
  --test_file "$test_file" \
  --info_file "$info_file" \
  --category "$CATEGORY" \
  --batch_size "$BATCH_SIZE" \
  --top_k_items "$TOP_K_ITEMS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --prefix_depth "$PREFIX_DEPTH" \
  --skip_rate "$SKIP_RATE" \
  --max_consecutive_skips "$MAX_CONSECUTIVE_SKIPS" \
  --num_stages "$NUM_STAGES" \
  --min_keep_per_stage "$MIN_KEEP_PER_STAGE" \
  --compensation none \
  --seed "$SEED" \
  --warmup_batches "$WARMUP_BATCHES" \
  --timed_batches "$TIMED_BATCHES" \
  --max_batches "$MAX_BATCHES" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "${CATEGORY}_opal_q_s${OPAL_STAGE}_prefix${PREFIX_DEPTH}_skip${SKIP_RATE}_bs${BATCH_SIZE}_seed${SEED}"

