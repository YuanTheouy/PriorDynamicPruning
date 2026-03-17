#!/bin/bash

# ================= Configuration =================

# 🔴 Option 1: Office Products
TEACHER_MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"

# 🔴 Option 2: Industrial and Scientific (Uncomment to use)
# TEACHER_MODEL_PATH="/workspace/ckpts/MiniOneRec/Industrial_ckpt"
# CATEGORY="Industrial_and_Scientific"

# Which student checkpoint to evaluate
EPOCH=20
STUDENT_CKPT="./student_ckpts/${CATEGORY}/student_epoch_${EPOCH}.pt"

# GPU Settings
# Number of GPUs for parallel evaluation
NUM_GPUS=8

# Evaluation Hyperparameters
BATCH_SIZE=32
TOP_K=50

# =================================================

CUDA_LIST=$(seq -s, 0 $(($NUM_GPUS - 1)))
CUDA_SPACE_LIST=$(seq -s " " 0 $(($NUM_GPUS - 1)))

echo "=== Starting Student Model Evaluation ==="
echo "Teacher Model (for Tokenizer): $TEACHER_MODEL_PATH"
echo "Student Checkpoint: $STUDENT_CKPT"
echo "Category: $CATEGORY"
echo "Using $NUM_GPUS GPUs"

# Check if teacher model exists
if [[ ! -d "$TEACHER_MODEL_PATH" ]]; then
    echo "❌ Error: Teacher model path does not exist: $TEACHER_MODEL_PATH"
    exit 1
fi

# Check if student checkpoint exists
if [[ ! -f "$STUDENT_CKPT" ]]; then
    echo "❌ Error: Student checkpoint does not exist: $STUDENT_CKPT"
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

echo "Test File: $test_file"
echo "Info File: $info_file"
echo "--------------------------------------------------------"

exp_name_clean=$(basename "$TEACHER_MODEL_PATH")
temp_dir="./temp_student_eval/${CATEGORY}-${exp_name_clean}_epoch${EPOCH}"
mkdir -p "$temp_dir"

# 1. Split Data
echo ">>> [1/4] Splitting test data for $NUM_GPUS GPUs..."
python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$CUDA_LIST"

# 2. Parallel Evaluation
echo ">>> [2/4] Running Evaluation..."
for i in $CUDA_SPACE_LIST
do
    echo "    Starting GPU $i ..."
    CUDA_VISIBLE_DEVICES=$i python -u ./eval_student.py \
        --teacher_model "$TEACHER_MODEL_PATH" \
        --student_ckpt "$STUDENT_CKPT" \
        --test_file "$temp_dir/${i}.csv" \
        --info_file "$info_file" \
        --category "$CATEGORY" \
        --batch_size $BATCH_SIZE \
        --top_k $TOP_K \
        --output_file "$temp_dir/${i}.json" &
done

# Wait for all background processes
wait
echo ">>> All GPU inference completed"

# 3. Merge Results
echo ">>> [3/4] Merging results..."
output_dir="./results/student_eval"
mkdir -p "$output_dir"
final_output="$output_dir/student_result_${CATEGORY}_epoch${EPOCH}.json"

# We skip merge.py because it doesn't handle the dict format of eval_student.py output.
# The aggregation is done in the inline python script below.

# 4. Aggregate Metrics
echo ">>> [4/4] Aggregating final metrics..."
python -c "
import json
import math
import glob

import pandas as pd
test_df = pd.read_csv('$test_file')
if 'dedup' in test_df.columns:
    test_df = test_df[test_df['dedup'] == 0]

# Check column names
if 'item_sid' in test_df.columns:
    ground_truths = test_df['item_sid'].tolist()
elif 'output' in test_df.columns:
    ground_truths = test_df['output'].tolist()
else:
    # Fallback: try to find a column that looks like SID
    # Assuming the last column is target if named 'target' or similar, 
    # but based on provided CSV snippet, 'item_sid' seems correct.
    ground_truths = test_df.iloc[:, -1].tolist()

files = glob.glob('$temp_dir/*.json')
# We need to sort files to ensure order matches ground_truths if order is preserved
# But parallel execution might scramble file order vs split order.
# The safest way is to read the 'merge.py' output which is sorted and merged correctly!
# Or, simply read the predictions in the order of file indices: 0.json, 1.json, ... 7.json

all_preds = []
# Sort files by numeric index (0.json, 1.json, ...)
files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        all_preds.extend(data.get('sample_predictions', []))



valid_topk = [1, 3, 5, 10, 20, 50]
ALLNDCG = [0.0] * len(valid_topk)
ALLHR = [0.0] * len(valid_topk)

for index, sample_preds in enumerate(all_preds):
    target_item = ground_truths[index].strip(' \n\"')
    
    minID = 1000000
    for i, pred_token_str in enumerate(sample_preds):
        if pred_token_str == target_item:
            minID = i
            break
            
    for i, topk in enumerate(valid_topk):
        if minID < topk:
            ALLNDCG[i] += (1 / math.log(minID + 2))
            ALLHR[i] += 1
            
num_samples = len(all_preds)
print(f'Evaluated on {num_samples} samples.')
print(f'TopK: {valid_topk}')

ndcg_res = [val / num_samples / (1.0 / math.log(2)) for val in ALLNDCG]
hr_res = [val / num_samples for val in ALLHR]

print(f'Full-Sequence NDCG: {[round(x, 4) for x in ndcg_res]}')
print(f'Full-Sequence HR:   {[round(x, 4) for x in hr_res]}')

# Save final aggregated results
with open('$final_output', 'w') as f:
    json.dump({
        'metrics': {
            'topk': valid_topk,
            'full_sequence_ndcg': ndcg_res,
            'full_sequence_hr': hr_res
        },
        'sample_predictions': all_preds[:10]
    }, f, indent=4)
"

echo "=== Evaluation Completed! ==="
