#!/usr/bin/env bash
set -euo pipefail

# Required closest-baseline completion run.
# This does not introduce new OPAL methods. It fills PuDDing-style and
# IG-style rows under objective-aligned Delta_NLL and KL settings.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/planrec_experiments}"

cd "$REPO_DIR"
unset SUBMISSION_RUN_GROUP

run_submission_step() {
  local name="$1"
  shift
  echo
  echo "=== Required related-baseline step: ${name} ==="
  env "$@" bash ./run_submission_convergence_experiments_gpu01234567.sh
}

summarize_group() {
  local group="$1"
  echo
  echo "=== Summarize ${group} ==="
  python3 summarize_planrec_results.py \
    --output_dir "$OUTPUT_DIR" \
    --table_name "summary_${group}.csv" \
    --run_name_contains "$group" \
    --exclude_debug_sanity
  cat "$OUTPUT_DIR/tables/quality_retention.md"
}

echo "=== Required PuDDing/IG baseline completion ==="
echo "REPO_DIR=${REPO_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# Delta_NLL-aligned rows for the submission-convergence setting.
run_submission_step "Delta_NLL Office_Products 25%" \
  SUBMISSION_TASKS=office25 \
  SUBMISSION_SEEDS="42 13 3407" \
  RISK_OBJECTIVE=Delta_NLL \
  SUBMISSION_RUN_PUDDING=1 \
  SUBMISSION_RUN_IG=1 \
  SUBMISSION_RUN_LAYERWISE=0

run_submission_step "Delta_NLL Industrial_and_Scientific 25%" \
  SUBMISSION_TASKS=industrial25 \
  SUBMISSION_SEEDS="42 13 3407" \
  RISK_OBJECTIVE=Delta_NLL \
  SECOND_DATASET_RELATED=all \
  SUBMISSION_RUN_PUDDING=1 \
  SUBMISSION_RUN_IG=1 \
  SUBMISSION_RUN_LAYERWISE=0

run_submission_step "Delta_NLL Office_Products 35.7%" \
  SUBMISSION_TASKS=office36 \
  SUBMISSION_SEEDS="42 13 3407" \
  RISK_OBJECTIVE=Delta_NLL \
  SUBMISSION_OFFICE36_RELATED=all \
  SUBMISSION_RUN_PUDDING=1 \
  SUBMISSION_RUN_IG=1 \
  SUBMISSION_RUN_LAYERWISE=0

# KL-aligned rows for KL-primary ablations/tables.
run_submission_step "KL Office_Products 25%" \
  SUBMISSION_TASKS=office25 \
  SUBMISSION_SEEDS="42 13 3407" \
  RISK_OBJECTIVE=KL \
  SUBMISSION_RUN_PUDDING=1 \
  SUBMISSION_RUN_IG=1 \
  SUBMISSION_RUN_LAYERWISE=0

summarize_group "submission_office25_delta_m2000"
summarize_group "submission_industrial25_delta_m2000"
summarize_group "submission_office36_delta_m2000"
summarize_group "submission_office25_kl_m2000"

echo
echo "=== Done required PuDDing/IG baseline completion ==="
