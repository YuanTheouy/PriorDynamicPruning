#!/bin/bash

# ================= Configuration =================
MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"
BATCH_SIZE=8 
TOP_K_LAYERS=12
TOP_K_ITEMS=50

# Output Base Dir
BASE_OUTPUT_DIR="./results/${CATEGORY}/static_baselines"
mkdir -p "$BASE_OUTPUT_DIR"

# Check dependencies
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Error: Teacher model path does not exist: $MODEL_PATH"
    exit 1
fi

test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -f "$test_file" ]] || [[ ! -f "$info_file" ]]; then
    echo "❌ Error: Data files not found"
    exit 1
fi

echo "Test File: $test_file"
echo "Info File: $info_file"
echo "Top-K Layers: $TOP_K_LAYERS"
echo "--------------------------------------------------------"

# Function to run a single baseline strategy
run_baseline() {
    STRATEGY=$1
    echo ">>> Running Baseline Strategy: $STRATEGY"
    
    OUTPUT_FILE="${BASE_OUTPUT_DIR}/${STRATEGY}_result.json"
    
    accelerate launch --num_processes 8 ./eval_static_baseline.py \
        --teacher_model "$MODEL_PATH" \
        --test_file "$test_file" \
        --info_file "$info_file" \
        --category "$CATEGORY" \
        --batch_size $BATCH_SIZE \
        --output_file "$OUTPUT_FILE" \
        --top_k_layers $TOP_K_LAYERS \
        --top_k_items $TOP_K_ITEMS \
        --strategy "$STRATEGY"
        
    # Aggregate and Compute Metrics
    python3 -c "
import json
import glob
import os
import math
import pandas as pd

output_file = '${OUTPUT_FILE}'
rank_files = glob.glob(output_file.replace('.json', '_rank*.json'))
all_data = []
for f in rank_files:
    try:
        with open(f, 'r') as fd:
            all_data.extend(json.load(fd))
        os.remove(f)
    except: pass

with open(output_file, 'w') as f:
    json.dump(all_data, f, indent=2)

# Metrics
test_df = pd.read_csv('${test_file}')
if 'dedup' in test_df.columns: test_df = test_df[test_df['dedup'] == 0]
if 'item_sid' in test_df.columns:
    ground_truths = test_df['item_sid'].astype(str).tolist()
else:
    ground_truths = test_df.iloc[:, -1].astype(str).tolist()

if len(all_data) > len(ground_truths): all_data = all_data[:len(ground_truths)]
elif len(all_data) < len(ground_truths): ground_truths = ground_truths[:len(all_data)]

valid_topk = [1, 3, 5, 10, 20, 50]
ALLNDCG = [0.0] * len(valid_topk)
ALLHR = [0.0] * len(valid_topk)

for index, item_data in enumerate(all_data):
    target_item = ground_truths[index].strip(' \n\"')
    sample_preds = item_data.get('sample_predictions', [])
    minID = 1000000
    for i, pred_token_str in enumerate(sample_preds):
        if pred_token_str == target_item:
            minID = i
            break
    for i, topk in enumerate(valid_topk):
        if minID < topk:
            ALLNDCG[i] += (1 / math.log(minID + 2))
            ALLHR[i] += 1

num = len(all_data)
ndcg = [val / num / (1.0 / math.log(2)) for val in ALLNDCG]
hr = [val / num for val in ALLHR]

print(f'\n[${STRATEGY}] NDCG: {[round(x, 4) for x in ndcg]}')
print(f'[${STRATEGY}] HR:   {[round(x, 4) for x in hr]}')
"
}

# Run all requested baselines
run_baseline "uniform"       # 均匀保留
run_baseline "first_k"       # 保前层
run_baseline "last_k"        # 保后层
run_baseline "ends_heavy"    # 保两头（前+后）
run_baseline "middle_heavy"  # 保中间

echo "=== All Baselines Completed! ==="
