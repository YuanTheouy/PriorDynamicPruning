#!/usr/bin/env bash
set -euo pipefail

# Submission-convergence runner.  No new methods are introduced here: this only
# orchestrates existing OPAL and related-baseline evaluators under one fair
# same-budget setting.

export REPO_DIR="${REPO_DIR:-/workspace/PriorDynamicPruning}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${SUBMISSION_BASE_PORT:-53000}"
export NEXT_PORT="$BASE_PORT"

export PATH="/root/venvs/planrec/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/transformers/src:${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/planrec_experiments}"
export RISK_DIR="${RISK_DIR:-${REPO_DIR}/results/opal_risk_labels}"
export CKPT_ROOT="${CKPT_ROOT:-${REPO_DIR}/policy_ckpts/submission_convergence}"
export RELATED_DIR="${RELATED_DIR:-${REPO_DIR}/policy_ckpts/submission_related_baselines}"
export LEGACY_OPAL_CKPT_ROOT="${LEGACY_OPAL_CKPT_ROOT:-${REPO_DIR}/policy_ckpts/final_opal_attn}"

export PRECISION="${PRECISION:-bf16}"
export PREFIX_DEPTH="${PREFIX_DEPTH:-4}"
export TOP_K_ITEMS="${TOP_K_ITEMS:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export LABEL_MAX_SAMPLES="${LABEL_MAX_SAMPLES:-2000}"
export LABEL_BATCH_SIZE="${LABEL_BATCH_SIZE:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export LAYERWISE_TRAIN_BATCH_SIZE="${LAYERWISE_TRAIN_BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EPOCHS="${EPOCHS:-20}"
export LAYERWISE_EPOCHS="${LAYERWISE_EPOCHS:-5}"
export LR="${LR:-1e-4}"
export RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0.1}"
export SKIP_SET_LOSS_WEIGHT="${SKIP_SET_LOSS_WEIGHT:-0.1}"
export HUBER_BETA="${HUBER_BETA:-1.0}"
export ROUTER_DIM="${ROUTER_DIM:-256}"
export ROUTER_HEADS="${ROUTER_HEADS:-4}"
export RECENT_TOKENS="${RECENT_TOKENS:-32}"
export RECENT_DECAY="${RECENT_DECAY:-0.85}"
export WARMUP_BATCHES="${WARMUP_BATCHES:-0}"
export TIMED_BATCHES="${TIMED_BATCHES:-0}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
export RISK_OBJECTIVE="${RISK_OBJECTIVE:-Delta_NLL}"
export CANDIDATE_COUNT="${CANDIDATE_COUNT:-16}"
export BEST_STATIC_CANDIDATE_COUNT="${BEST_STATIC_CANDIDATE_COUNT:-32}"
export BEST_STATIC_VAL_MAX_SAMPLES="${BEST_STATIC_VAL_MAX_SAMPLES:-0}"
export IG_CLUSTERS="${IG_CLUSTERS:-8}"
export INPUT_GUIDED_NUM_BINS="${INPUT_GUIDED_NUM_BINS:-16}"

export SUBMISSION_TASKS="${SUBMISSION_TASKS:-office25}"
export SUBMISSION_SEEDS="${SUBMISSION_SEEDS:-42}"
export SUBMISSION_RUN_LAYERWISE="${SUBMISSION_RUN_LAYERWISE:-1}"
export SUBMISSION_RUN_PUDDING="${SUBMISSION_RUN_PUDDING:-1}"
export SUBMISSION_RUN_IG="${SUBMISSION_RUN_IG:-1}"
export SUBMISSION_RUN_RANDOM_DYNAMIC="${SUBMISSION_RUN_RANDOM_DYNAMIC:-1}"
export SUBMISSION_USE_LEGACY_OFFICE25="${SUBMISSION_USE_LEGACY_OFFICE25:-1}"
export SUBMISSION_OFFICE36_RELATED="${SUBMISSION_OFFICE36_RELATED:-ig}"

cd "$REPO_DIR"
mkdir -p "$RISK_DIR" "$CKPT_ROOT" "$RELATED_DIR" "$OUTPUT_DIR"

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
    port_log="$(mktemp "/tmp/submission_accelerate_${port}_XXXX.log")"
    set +e
    accelerate launch \
      --num_processes "$NUM_GPUS" \
      --num_machines 1 \
      --main_process_port "$port" \
      --mixed_precision "$PRECISION" \
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
    rm -f "$port_log"
    return "$status"
  done
}

line_count_at_least() {
  local path="$1"
  local min_rows="$2"
  test -s "$path" && [ "$(wc -l < "$path")" -ge "$min_rows" ]
}

final_json_exists() {
  local run_name="$1"
  test -s "${OUTPUT_DIR}/raw_json/${run_name}.json"
}

skip_tag() {
  printf "%s" "$1" | tr "." "p"
}

configure_dataset() {
  local dataset="$1"
  case "$dataset" in
    Office_Products)
      export MODEL_PATH="${OFFICE_MODEL_PATH:-/workspace/ckpts/MiniOneRec/Office_ckpt}"
      export CATEGORY="Office_Products"
      export TRAIN_FILE="${REPO_DIR}/data/Amazon/train/Office_Products_5_2016-10-2018-11.csv"
      export VALID_FILE="${REPO_DIR}/data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv"
      export TEST_FILE="${REPO_DIR}/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv"
      export INFO_FILE="${REPO_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"
      ;;
    Industrial_and_Scientific)
      export MODEL_PATH="${INDUSTRIAL_MODEL_PATH:-/workspace/ckpts/MiniOneRec/Industrial_ckpt}"
      export CATEGORY="Industrial_and_Scientific"
      export TRAIN_FILE="${REPO_DIR}/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
      export VALID_FILE="${REPO_DIR}/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
      export TEST_FILE="${REPO_DIR}/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
      export INFO_FILE="${REPO_DIR}/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

configure_task() {
  local task="$1"
  case "$task" in
    office25)
      configure_dataset Office_Products
      export SKIP_RATE="0.25"
      export TOP_K_LAYERS="21"
      export RUN_GROUP="${SUBMISSION_RUN_GROUP:-submission_office25_delta_m${LABEL_MAX_SAMPLES}}"
      export TASK_PROFILE="office_main"
      ;;
    industrial25)
      configure_dataset Industrial_and_Scientific
      export SKIP_RATE="0.25"
      export TOP_K_LAYERS="21"
      export RUN_GROUP="${SUBMISSION_RUN_GROUP:-submission_industrial25_delta_m${LABEL_MAX_SAMPLES}}"
      export TASK_PROFILE="second_dataset"
      ;;
    office36)
      configure_dataset Office_Products
      export SKIP_RATE="${OFFICE36_SKIP_RATE:-0.357}"
      export TOP_K_LAYERS="${OFFICE36_TOP_K_LAYERS:-18}"
      export RUN_GROUP="${SUBMISSION_RUN_GROUP:-submission_office36_delta_m${LABEL_MAX_SAMPLES}}"
      export TASK_PROFILE="second_skip_rate"
      ;;
    *)
      echo "Unknown SUBMISSION_TASKS entry: ${task}" >&2
      exit 1
      ;;
  esac
  export SKIP_TAG
  SKIP_TAG="$(skip_tag "$SKIP_RATE")"
  export SUMMARY_TABLE="summary_${RUN_GROUP}.csv"
}

risk_label_file_for_seed() {
  local seed="$1"
  printf "%s/%s_risk_%s_random_m%s_seed%s.jsonl" "$RISK_DIR" "$CATEGORY" "$RISK_OBJECTIVE" "$LABEL_MAX_SAMPLES" "$seed"
}

build_labels_if_needed() {
  local seed="$1"
  local label_file="$2"
  if line_count_at_least "$label_file" "$LABEL_MAX_SAMPLES"; then
    echo "=== Reuse one-layer ${RISK_OBJECTIVE} risk labels: ${label_file} ==="
    return
  fi
  run_accelerate ./build_layer_risk_labels.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --prefix_depth "$PREFIX_DEPTH" \
    --max_samples "$LABEL_MAX_SAMPLES" \
    --sample_strategy random \
    --sample_seed "$seed" \
    --objective "$RISK_OBJECTIVE" \
    --output "$label_file" \
    --batch_size "$LABEL_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$seed"
}

legacy_office25_ckpt() {
  local seed="$1"
  local variant="$2"
  if [ "$SUBMISSION_USE_LEGACY_OFFICE25" = "1" ] \
    && [ "$CATEGORY" = "Office_Products" ] \
    && [ "$SKIP_RATE" = "0.25" ] \
    && [ "$TOP_K_LAYERS" = "21" ] \
    && [ "$RISK_OBJECTIVE" = "Delta_NLL" ] \
    && [ "$LABEL_MAX_SAMPLES" = "2000" ]; then
    printf "%s/final_opal_attn_delta_m2000_seed%s/%s/risk_router.pt" "$LEGACY_OPAL_CKPT_ROOT" "$seed" "$variant"
  fi
}

risk_ckpt_for_variant() {
  local seed="$1"
  local variant="$2"
  local local_ckpt="${CKPT_ROOT}/${RUN_GROUP}_seed${seed}/${variant}/risk_router.pt"
  local legacy_ckpt
  legacy_ckpt="$(legacy_office25_ckpt "$seed" "$variant" || true)"
  if [ -n "$legacy_ckpt" ] && [ -s "$legacy_ckpt" ]; then
    printf "%s" "$legacy_ckpt"
  else
    printf "%s" "$local_ckpt"
  fi
}

train_risk_router_if_needed() {
  local seed="$1"
  local label_file="$2"
  local variant="$3"
  local router_input="$4"
  local risk_pooling="$5"
  local ckpt
  ckpt="$(risk_ckpt_for_variant "$seed" "$variant")"
  if [ -s "$ckpt" ]; then
    echo "=== Reuse risk-router checkpoint for ${variant}: ${ckpt} ==="
    return
  fi

  local out_dir="${CKPT_ROOT}/${RUN_GROUP}_seed${seed}/${variant}"
  local args=(
    ./train_opal_risk_router.py
    --teacher_model "$MODEL_PATH"
    --train_file "$TRAIN_FILE"
    --info_file "$INFO_FILE"
    --category "$CATEGORY"
    --risk_label_file "$label_file"
    --router_input "$router_input"
    --prefix_depth "$PREFIX_DEPTH"
    --batch_size "$TRAIN_BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT"
    --skip_set_loss_weight "$SKIP_SET_LOSS_WEIGHT"
    --skip_rate "$SKIP_RATE"
    --top_k_layers "$TOP_K_LAYERS"
    --huber_beta "$HUBER_BETA"
    --router_dim "$ROUTER_DIM"
    --router_heads "$ROUTER_HEADS"
    --recent_tokens "$RECENT_TOKENS"
    --recent_decay "$RECENT_DECAY"
    --output_dir "$out_dir"
    --precision "$PRECISION"
    --loss_log_interval 50
    --seed "$seed"
  )
  if [ "$risk_pooling" != "none" ]; then
    args+=(--risk_pooling "$risk_pooling")
  fi
  echo "=== Train frozen-family risk router ${variant} seed=${seed} ==="
  run_accelerate "${args[@]}"
}

eval_method_if_needed() {
  local run_name="$1"
  shift
  if final_json_exists "$run_name"; then
    echo "=== Reuse eval JSON: ${run_name} ==="
    return
  fi
  run_accelerate ./eval_planrec_opal.py \
    "$@" \
    --teacher_model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --top_k_items "$TOP_K_ITEMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --precision "$PRECISION" \
    --prefix_depth "$PREFIX_DEPTH" \
    --skip_rate "$SKIP_RATE" \
    --seed "$SEED" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --max_batches "$EVAL_MAX_BATCHES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$run_name"
}

build_best_static_library_if_needed() {
  local seed="$1"
  local label_file="$2"
  export BEST_STATIC_LABEL_FILE="${RISK_DIR}/${CATEGORY}_best_static_val_${RISK_OBJECTIVE}_C${BEST_STATIC_CANDIDATE_COUNT}_k${TOP_K_LAYERS}_skip${SKIP_TAG}_seed${seed}.jsonl"
  export BEST_STATIC_LIBRARY="${RISK_DIR}/${CATEGORY}_best_static_val_${RISK_OBJECTIVE}_C${BEST_STATIC_CANDIDATE_COUNT}_k${TOP_K_LAYERS}_skip${SKIP_TAG}_seed${seed}.json"

  if ! line_count_at_least "$BEST_STATIC_LABEL_FILE" 1; then
    run_accelerate ./build_candidate_mask_labels.py \
      --teacher_model "$MODEL_PATH" \
      --calibration_file "$VALID_FILE" \
      --info_file "$INFO_FILE" \
      --category "$CATEGORY" \
      --risk_label_file "$label_file" \
      --candidate_count "$BEST_STATIC_CANDIDATE_COUNT" \
      --skip_rate "$SKIP_RATE" \
      --top_k_layers "$TOP_K_LAYERS" \
      --max_samples "$BEST_STATIC_VAL_MAX_SAMPLES" \
      --sample_strategy first \
      --objective "$RISK_OBJECTIVE" \
      --output "$BEST_STATIC_LABEL_FILE" \
      --batch_size "$LABEL_BATCH_SIZE" \
      --precision "$PRECISION" \
      --seed "$seed"
  else
    echo "=== Reuse best-static validation labels: ${BEST_STATIC_LABEL_FILE} ==="
  fi

  if [ -s "$BEST_STATIC_LIBRARY" ]; then
    echo "=== Reuse best-static mask library: ${BEST_STATIC_LIBRARY} ==="
  else
    python3 ./select_best_static_mask_from_candidate_labels.py \
      --candidate_label_file "$BEST_STATIC_LABEL_FILE" \
      --objective "$RISK_OBJECTIVE" \
      --output "$BEST_STATIC_LIBRARY"
  fi
  export BEST_STATIC_MASK_ID
  BEST_STATIC_MASK_ID="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_mask_id"])' "$BEST_STATIC_LIBRARY")"
}

build_candidate_labels_if_needed() {
  local seed="$1"
  local label_file="$2"
  export CANDIDATE_LABEL_FILE="${RISK_DIR}/${CATEGORY}_candidate_${RISK_OBJECTIVE}_C${CANDIDATE_COUNT}_m${LABEL_MAX_SAMPLES}_k${TOP_K_LAYERS}_skip${SKIP_TAG}_seed${seed}.jsonl"
  if line_count_at_least "$CANDIDATE_LABEL_FILE" "$LABEL_MAX_SAMPLES"; then
    echo "=== Reuse PuDDing candidate labels: ${CANDIDATE_LABEL_FILE} ==="
    return
  fi
  run_accelerate ./build_candidate_mask_labels.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$label_file" \
    --use_risk_label_sample_ids \
    --candidate_count "$CANDIDATE_COUNT" \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --objective "$RISK_OBJECTIVE" \
    --output "$CANDIDATE_LABEL_FILE" \
    --batch_size "$LABEL_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$seed"
}

train_pudding_if_needed() {
  local seed="$1"
  export PUDDING_DIR="${RELATED_DIR}/${RUN_GROUP}_seed${seed}/pudding_prompt_candidate_C${CANDIDATE_COUNT}_k${TOP_K_LAYERS}"
  if [ -s "${PUDDING_DIR}/candidate_router.pt" ]; then
    echo "=== Reuse PuDDing-style checkpoint: ${PUDDING_DIR}/candidate_router.pt ==="
    return
  fi
  run_accelerate ./train_prompt_candidate_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --candidate_label_file "$CANDIDATE_LABEL_FILE" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --huber_beta "$HUBER_BETA" \
    --output_dir "$PUDDING_DIR" \
    --precision "$PRECISION" \
    --seed "$seed"
}

build_ig_if_needed() {
  local seed="$1"
  local k="$2"
  export IG_ARTIFACT="${RELATED_DIR}/${RUN_GROUP}_seed${seed}/ig_cluster/ig_cluster_K${k}_k${TOP_K_LAYERS}.pt"
  if [ -s "$IG_ARTIFACT" ]; then
    echo "=== Reuse IG-style cluster artifact: ${IG_ARTIFACT} ==="
    return
  fi
  run_accelerate ./build_ig_cluster_artifact.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$LABEL_FILE" \
    --num_clusters "$k" \
    --normalize_embeddings \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --output "$IG_ARTIFACT" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --precision "$PRECISION" \
    --seed "$seed"
}

train_layerwise_if_needed() {
  local seed="$1"
  local label_file="$2"
  export LAYERWISE_DIR="${RELATED_DIR}/${RUN_GROUP}_seed${seed}/layerwise_hidden_router_k${TOP_K_LAYERS}"
  if [ -s "${LAYERWISE_DIR}/layerwise_hidden_router.pt" ]; then
    echo "=== Reuse layerwise hidden router: ${LAYERWISE_DIR}/layerwise_hidden_router.pt ==="
    return
  fi
  run_accelerate ./train_layerwise_hidden_router.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --info_file "$INFO_FILE" \
    --category "$CATEGORY" \
    --risk_label_file "$label_file" \
    --risk_pooling mean \
    --batch_size "$LAYERWISE_TRAIN_BATCH_SIZE" \
    --epochs "$LAYERWISE_EPOCHS" \
    --lr "$LR" \
    --ranking_loss_weight "$RANKING_LOSS_WEIGHT" \
    --skip_set_loss_weight "$SKIP_SET_LOSS_WEIGHT" \
    --skip_rate "$SKIP_RATE" \
    --top_k_layers "$TOP_K_LAYERS" \
    --huber_beta "$HUBER_BETA" \
    --output_dir "$LAYERWISE_DIR" \
    --precision "$PRECISION" \
    --seed "$seed"
}

run_base_methods() {
  local seed="$1"
  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_full" \
    --method full

  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_static_uniform_k${TOP_K_LAYERS}" \
    --method static \
    --static_strategy uniform

  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_static_ends_heavy_k${TOP_K_LAYERS}" \
    --method static \
    --static_strategy ends_heavy

  build_best_static_library_if_needed "$seed" "$LABEL_FILE"
  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_static_best_on_val_k${TOP_K_LAYERS}" \
    --method static \
    --mask_library "$BEST_STATIC_LIBRARY" \
    --mask_id "$BEST_STATIC_MASK_ID"
}

run_dynamic_baselines() {
  local seed="$1"
  local raw_ckpt prefix_last_ckpt attn_ckpt
  raw_ckpt="$(risk_ckpt_for_variant "$seed" raw_input_risk)"
  prefix_last_ckpt="$(risk_ckpt_for_variant "$seed" prefix_hk_last)"
  attn_ckpt="$(risk_ckpt_for_variant "$seed" prefix_hk_raw_attn)"

  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_raw_input_risk_skip${SKIP_RATE}" \
    --method raw_input_risk \
    --risk_router_ckpt "$raw_ckpt"

  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_opal_prefixlast_skip${SKIP_RATE}" \
    --method opal_risk \
    --risk_router_ckpt "$prefix_last_ckpt"

  eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_opal_attn_skip${SKIP_RATE}" \
    --method opal_risk \
    --risk_router_ckpt "$attn_ckpt"

  if [ "$SUBMISSION_RUN_RANDOM_DYNAMIC" = "1" ]; then
    eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_random_dynamic_hash_skip${SKIP_RATE}" \
      --method input_guided \
      --input_guided_selector hash \
      --input_guided_feature_set hash \
      --input_guided_num_bins "$INPUT_GUIDED_NUM_BINS"
  fi
}

run_related_methods() {
  local seed="$1"
  local related_mode="${2:-all}"

  if [ "$SUBMISSION_RUN_PUDDING" = "1" ] && { [ "$related_mode" = "all" ] || [ "$related_mode" = "pudding" ]; }; then
    build_candidate_labels_if_needed "$seed" "$LABEL_FILE"
    train_pudding_if_needed "$seed"
    eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_pudding_prompt_candidate_C${CANDIDATE_COUNT}" \
      --method pudding_prompt_candidate \
      --candidate_router_ckpt "${PUDDING_DIR}/candidate_router.pt"
  fi

  if [ "$SUBMISSION_RUN_IG" = "1" ] && { [ "$related_mode" = "all" ] || [ "$related_mode" = "ig" ]; }; then
    for K in $IG_CLUSTERS; do
      build_ig_if_needed "$seed" "$K"
      eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_ig_cluster_K${K}" \
        --method ig_cluster_mask \
        --ig_cluster_artifact "$IG_ARTIFACT"
    done
  fi

  if [ "$SUBMISSION_RUN_LAYERWISE" = "1" ] && { [ "$related_mode" = "all" ] || [ "$related_mode" = "layerwise" ]; }; then
    train_layerwise_if_needed "$seed" "$LABEL_FILE"
    eval_method_if_needed "${CATEGORY}_${RUN_GROUP}_seed${seed}_layerwise_hidden_router" \
      --method layerwise_hidden_router \
      --layerwise_router_ckpt "${LAYERWISE_DIR}/layerwise_hidden_router.pt"
  fi
}

run_seed_for_task() {
  local seed="$1"
  export SEED="$seed"
  export LABEL_FILE
  LABEL_FILE="$(risk_label_file_for_seed "$seed")"

  build_labels_if_needed "$seed" "$LABEL_FILE"
  train_risk_router_if_needed "$seed" "$LABEL_FILE" raw_input_risk raw_embedding none
  train_risk_router_if_needed "$seed" "$LABEL_FILE" prefix_hk_last prefix_hk last
  train_risk_router_if_needed "$seed" "$LABEL_FILE" prefix_hk_raw_attn prefix_hk_raw_attn none

  run_base_methods "$seed"
  run_dynamic_baselines "$seed"

  case "$TASK_PROFILE" in
    office_main)
      run_related_methods "$seed" all
      ;;
    second_dataset)
      run_related_methods "$seed" "${SECOND_DATASET_RELATED:-ig}"
      ;;
    second_skip_rate)
      run_related_methods "$seed" "$SUBMISSION_OFFICE36_RELATED"
      ;;
  esac
}

for TASK in $SUBMISSION_TASKS; do
  configure_task "$TASK"
  echo "=== Submission convergence task=${TASK} ==="
  echo "CATEGORY=${CATEGORY}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "SKIP_RATE=${SKIP_RATE}"
  echo "TOP_K_LAYERS=${TOP_K_LAYERS}"
  echo "RUN_GROUP=${RUN_GROUP}"
  echo "SEEDS=${SUBMISSION_SEEDS}"

  for SEED in $SUBMISSION_SEEDS; do
    run_seed_for_task "$SEED"
  done

  python3 summarize_planrec_results.py \
    --output_dir "$OUTPUT_DIR" \
    --table_name "$SUMMARY_TABLE" \
    --run_name_contains "$RUN_GROUP" \
    --exclude_debug_sanity

  cat "$OUTPUT_DIR/tables/quality_retention.md"
  echo "=== Done submission convergence task=${TASK} ==="
done
