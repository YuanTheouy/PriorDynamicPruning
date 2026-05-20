#!/bin/bash
set -euo pipefail

export OUTPUT_DIR=${OUTPUT_DIR:-./results/planrec_experiments}
export WARMUP_BATCHES=${WARMUP_BATCHES:-2}
export TIMED_BATCHES=${TIMED_BATCHES:-20}
export MAX_BATCHES=${MAX_BATCHES:-0}

bash ./run_planrec_full_teacher.sh
bash ./run_planrec_static_templates.sh
bash ./run_planrec_dynamic_planner.sh
bash ./run_planrec_input_guided_selector.sh
bash ./run_planrec_layerwise_router.sh
python3 ./summarize_planrec_results.py --output_dir "$OUTPUT_DIR"
