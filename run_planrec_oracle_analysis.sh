#!/bin/bash
set -euo pipefail

CATEGORY=${CATEGORY:-Office_Products}
OUTPUT_DIR=${OUTPUT_DIR:-./results/planrec_experiments}
TOP_K_LAYERS=${TOP_K_LAYERS:-21}
SEED=${SEED:-42}
PREDICTION_METHOD=${PREDICTION_METHOD:-dynamic_template}
INPUT_GUIDED_SELECTOR=${INPUT_GUIDED_SELECTOR:-length_hash}
INPUT_GUIDED_FEATURE_SET=${INPUT_GUIDED_FEATURE_SET:-length_hash}
ORACLE_JSON=${ORACLE_JSON:-${OUTPUT_DIR}/raw_json/${CATEGORY}_oracle_k${TOP_K_LAYERS}_seed${SEED}.json}
PREDICTION_JSON=${PREDICTION_JSON:-}

test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)

if [[ -z "$PREDICTION_JSON" ]]; then
  case "$PREDICTION_METHOD" in
    dynamic_template)
      PREDICTION_JSON=${OUTPUT_DIR}/raw_json/${CATEGORY}_dynamic_template_k${TOP_K_LAYERS}_seed${SEED}.json
      ;;
    input_guided)
      PREDICTION_JSON=${OUTPUT_DIR}/raw_json/${CATEGORY}_input_guided_${INPUT_GUIDED_SELECTOR}_${INPUT_GUIDED_FEATURE_SET}_k${TOP_K_LAYERS}_seed${SEED}.json
      ;;
    layerwise_router)
      PREDICTION_JSON=${OUTPUT_DIR}/raw_json/${CATEGORY}_layerwise_router_k${TOP_K_LAYERS}_seed${SEED}.json
      ;;
    *)
      echo "Unknown PREDICTION_METHOD=$PREDICTION_METHOD; set PREDICTION_JSON explicitly"
      exit 1
      ;;
  esac
fi

if [[ ! -f "$ORACLE_JSON" ]]; then
  echo "Missing ORACLE_JSON: $ORACLE_JSON"
  exit 1
fi
if [[ ! -f "$PREDICTION_JSON" ]]; then
  echo "Missing PREDICTION_JSON: $PREDICTION_JSON"
  exit 1
fi

name="${CATEGORY}_${PREDICTION_METHOD}_oracle_analysis_k${TOP_K_LAYERS}_seed${SEED}"
if [[ "$PREDICTION_METHOD" == "input_guided" ]]; then
  name="${CATEGORY}_${PREDICTION_METHOD}_${INPUT_GUIDED_SELECTOR}_${INPUT_GUIDED_FEATURE_SET}_oracle_analysis_k${TOP_K_LAYERS}_seed${SEED}"
fi
python3 ./analyze_planrec_oracle.py \
  --oracle_json "$ORACLE_JSON" \
  --prediction_json "$PREDICTION_JSON" \
  --test_file "$test_file" \
  --output_dir "$OUTPUT_DIR" \
  --name "$name"
