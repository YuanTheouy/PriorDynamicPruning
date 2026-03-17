#!/bin/bash

# ================= Configuration =================

# 🔴 Option 1: Office Products
MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"

# Policy Hyperparameters
BATCH_SIZE=8 
TOP_K_LAYERS=12
TOP_K_ITEMS=50

# Output
OUTPUT_FILE="./results/${CATEGORY}/policy_joint_result.json"

# Check if user passed --unfrozen argument
UNFROZEN_MODE=false
for arg in "$@"
do
    if [ "$arg" == "--unfrozen" ]; then
        UNFROZEN_MODE=true
    fi
done

# =================================================

echo "=== Starting Joint Policy Evaluation ==="
echo "Teacher Model: $MODEL_PATH"
echo "Category: $CATEGORY"
echo "Top-K Layers: $TOP_K_LAYERS"
echo "Output File: $OUTPUT_FILE"

# Check if teacher model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Error: Teacher model path does not exist: $MODEL_PATH"
    exit 1
fi

# Automatically find test data and info file
test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -f "$test_file" ]]; then
    echo "❌ Error: Test file not found for category $CATEGORY"
    exit 1
fi

if [[ ! -f "$info_file" ]]; then
    echo "❌ Error: Info file not found for category $CATEGORY"
    exit 1
fi

# =========================================================================
# Checkpoint Detection Logic
# =========================================================================

# 1. Look for Policy Checkpoint
if [ "$UNFROZEN_MODE" = true ]; then
    POLICY_CKPT_DIR="./policy_ckpts/unfrozon/${CATEGORY}"
    echo "🔍 Mode: Unfrozen (Looking in $POLICY_CKPT_DIR)"
else
    POLICY_CKPT_DIR="./policy_ckpts/${CATEGORY}"
    echo "🔍 Mode: Standard (Looking in $POLICY_CKPT_DIR)"
fi

if [[ ! -d "$POLICY_CKPT_DIR" ]]; then
    echo "❌ Error: Policy checkpoint directory not found: $POLICY_CKPT_DIR"
    exit 1
fi

latest_policy_ckpt=$(ls -v ${POLICY_CKPT_DIR}/policy_epoch_*.pt 2>/dev/null | tail -1)

if [[ -z "$latest_policy_ckpt" ]]; then
    echo "❌ Error: No policy checkpoints found in $POLICY_CKPT_DIR"
    exit 1
fi

# 2. Look for Student Checkpoint
# Priority: Fine-tuned student in Policy dir > Pre-trained student in Student dir
latest_student_ckpt=$(ls -v ${POLICY_CKPT_DIR}/finetuned_student_epoch_*.pt 2>/dev/null | tail -1)

if [[ -n "$latest_student_ckpt" ]]; then
    echo "✅ Found Fine-tuned Student: $latest_student_ckpt"
else
    # Fallback to pre-trained student
    STUDENT_CKPT_DIR="./student_ckpts/${CATEGORY}"
    latest_student_ckpt=$(ls -v ${STUDENT_CKPT_DIR}/student_epoch_*.pt 2>/dev/null | tail -1)
    
    if [[ -z "$latest_student_ckpt" ]]; then
        echo "❌ Error: No student checkpoints found in $STUDENT_CKPT_DIR"
        exit 1
    fi
    echo "⚠️  Using Pre-trained Student (No fine-tuning detected): $latest_student_ckpt"
fi

echo "Test File: $test_file"
echo "Info File: $info_file"
echo "Student Checkpoint: $latest_student_ckpt"
echo "Policy Checkpoint: $latest_policy_ckpt"
echo "--------------------------------------------------------"

# Run the evaluation script using accelerate for multi-GPU
# We use 8 processes by default
NUM_GPUS=8

accelerate launch --num_processes $NUM_GPUS ./eval_policy_joint.py \
    --teacher_model "$MODEL_PATH" \
    --student_ckpt "$latest_student_ckpt" \
    --policy_ckpt "$latest_policy_ckpt" \
    --test_file "$test_file" \
    --info_file "$info_file" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --output_file "$OUTPUT_FILE" \
    --top_k_layers $TOP_K_LAYERS \
    --top_k_items $TOP_K_ITEMS

# Wait for all processes to finish (accelerate launch usually waits)
echo "=== Aggregating Results & Calculating Metrics... ==="

# Python script to merge rank files and calculate NDCG/HR
python3 -c "
import json
import glob
import os
import math
import pandas as pd

output_file = '${OUTPUT_FILE}'
rank_files = glob.glob(output_file.replace('.json', '_rank*.json'))
print(f'Found {len(rank_files)} rank files to merge.')

all_data = []
# Sort files to ensure deterministic order if possible, though 'sample_predictions' order matters vs ground truth
rank_files = sorted(rank_files)

for f in rank_files:
    try:
        with open(f, 'r') as fd:
            data = json.load(fd)
            all_data.extend(data)
    except Exception as e:
        print(f'Error reading {f}: {e}')

# Remove rank files
for f in rank_files:
    os.remove(f)

# Save merged file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(all_data, f, indent=2)

print(f'Successfully merged {len(all_data)} predictions into {output_file}')

# --- Calculate Metrics (NDCG/HR) ---
test_file = '${test_file}'
test_df = pd.read_csv(test_file)

# Handle dedup if present (align with eval_student.py)
if 'dedup' in test_df.columns:
    test_df = test_df[test_df['dedup'] == 0]

# Extract Ground Truths
if 'item_sid' in test_df.columns:
    ground_truths = test_df['item_sid'].astype(str).tolist()
else:
    ground_truths = test_df.iloc[:, -1].astype(str).tolist()

# Ensure length alignment
if len(all_data) > len(ground_truths):
    print(f'Warning: Predictions ({len(all_data)}) > Ground Truths ({len(ground_truths)}). Truncating.')
    all_data = all_data[:len(ground_truths)]
elif len(all_data) < len(ground_truths):
    print(f'Warning: Predictions ({len(all_data)}) < Ground Truths ({len(ground_truths)}). Using available subset.')
    ground_truths = ground_truths[:len(all_data)]

valid_topk = [1, 3, 5, 10, 20, 50]
ALLNDCG = [0.0] * len(valid_topk)
ALLHR = [0.0] * len(valid_topk)

for index, item_data in enumerate(all_data):
    target_item = ground_truths[index].strip(' \n\"')
    sample_preds = item_data.get('sample_predictions', [])
    
    # Find rank of target_item in predictions
    minID = 1000000
    for i, pred_token_str in enumerate(sample_preds):
        if pred_token_str == target_item:
            minID = i
            break
            
    for i, topk in enumerate(valid_topk):
        if minID < topk:
            ALLNDCG[i] += (1 / math.log(minID + 2))
            ALLHR[i] += 1

num_samples = len(all_data)
ndcg_res = [val / num_samples / (1.0 / math.log(2)) for val in ALLNDCG]
hr_res = [val / num_samples for val in ALLHR]

print(f'\n=== Final Metrics (Pruned Top-{${TOP_K_LAYERS}} Layers) ===')
print(f'Evaluated on {num_samples} samples.')
print(f'TopK: {valid_topk}')
print(f'Full-Sequence NDCG: {[round(x, 4) for x in ndcg_res]}')
print(f'Full-Sequence HR:   {[round(x, 4) for x in hr_res]}')

# --- Analyze Mask Usage ---
if len(all_data) > 0:
    num_layers = len(all_data[0]['layer_mask'])
    avg_layers = 0
    layer_counts = [0] * num_layers
    
    for p in all_data:
        mask = p['layer_mask']
        current_sum = sum(mask)
        avg_layers += current_sum
        for i, val in enumerate(mask):
            layer_counts[i] += val
            
    avg_layers /= len(all_data)
    print(f'\n=== Layer Usage ===')
    print(f'Average Layers Kept: {avg_layers:.2f} / {num_layers}')
    print(f'Layer Usage Distribution: {layer_counts}')
"

echo "=== Evaluation Completed! Results saved to $OUTPUT_FILE ==="
