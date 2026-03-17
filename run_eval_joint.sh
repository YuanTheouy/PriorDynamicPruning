#!/bin/bash

# ================= Configuration =================

# 🔴 Option 1: Office Products
TEACHER_MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"

# 🔴 Option 2: Industrial and Scientific (Uncomment to use)
# TEACHER_MODEL_PATH="/workspace/ckpts/MiniOneRec/Industrial_ckpt"
# CATEGORY="Industrial_and_Scientific"

# Which student checkpoint to evaluate
EPOCH=3
STUDENT_CKPT="./student_ckpts/${CATEGORY}/student_epoch_${EPOCH}.pt"

# GPU Settings
# Number of GPUs for parallel evaluation
NUM_GPUS=8

# Evaluation Hyperparameters
BATCH_SIZE=8
TOP_K_STUDENT=50
NUM_BEAMS=50
MAX_NEW_TOKENS=256

# =================================================

CUDA_LIST=$(seq -s, 0 $(($NUM_GPUS - 1)))
CUDA_SPACE_LIST=$(seq -s " " 0 $(($NUM_GPUS - 1)))

echo "=== Starting Joint Inference (Student -> Teacher) ==="
echo "Teacher Model: $TEACHER_MODEL_PATH"
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

exp_name_clean=$(basename "$TEACHER_MODEL_PATH")
temp_dir="./temp_joint/${CATEGORY}-${exp_name_clean}_epoch${EPOCH}"
mkdir -p "$temp_dir"

# 1. Split Data
echo ">>> [1/4] Splitting test data for $NUM_GPUS GPUs..."
python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$CUDA_LIST"

# 2. Parallel Evaluation
echo ">>> [2/4] Running Joint Inference..."
for i in $CUDA_SPACE_LIST
do
    echo "    Starting GPU $i ..."
    CUDA_VISIBLE_DEVICES=$i python -u ./eval_joint.py \
        --teacher_model "$TEACHER_MODEL_PATH" \
        --student_ckpt "$STUDENT_CKPT" \
        --test_file "$temp_dir/${i}.csv" \
        --info_file "$info_file" \
        --category "$CATEGORY" \
        --batch_size $BATCH_SIZE \
        --top_k_student $TOP_K_STUDENT \
        --num_beams $NUM_BEAMS \
        --max_new_tokens $MAX_NEW_TOKENS \
        --output_file "$temp_dir/${i}.json" &
done

# Wait for all background processes
wait
echo ">>> All GPU inference completed"

# 3. Merge Results
echo ">>> [3/4] Merging results..."
output_dir="./results/joint_eval/${exp_name_clean}"
mkdir -p "$output_dir"
final_output="$output_dir/joint_result_${CATEGORY}_epoch${EPOCH}.json"

actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')

python ./merge.py \
    --input_path "$temp_dir" \
    --output_path "$final_output" \
    --cuda_list "$actual_cuda_list"

# 4. Calculate Metrics
echo ">>> [4/4] Calculating final metrics (NDCG, HR)..."
python ./calc.py \
    --path "$final_output" \
    --item_path "$info_file"

echo "=== Joint Evaluation Completed! ==="
