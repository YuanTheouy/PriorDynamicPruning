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

# =================================================

echo "=== Starting Joint Policy Evaluation ==="
echo "Teacher Model: $MODEL_PATH"
echo "Category: $CATEGORY"
echo "Top-K Layers: $TOP_K_LAYERS"

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

# Automatically find the latest student checkpoint
STUDENT_CKPT_DIR="./student_ckpts/${CATEGORY}"
latest_student_ckpt=$(ls -v ${STUDENT_CKPT_DIR}/student_epoch_*.pt 2>/dev/null | tail -1)

if [[ -z "$latest_student_ckpt" ]]; then
    echo "❌ Error: No student checkpoints found in $STUDENT_CKPT_DIR"
    exit 1
fi

# Automatically find the latest policy checkpoint
POLICY_CKPT_DIR="./policy_ckpts/${CATEGORY}"
latest_policy_ckpt=$(ls -v ${POLICY_CKPT_DIR}/policy_epoch_*.pt 2>/dev/null | tail -1)

if [[ -z "$latest_policy_ckpt" ]]; then
    echo "❌ Error: No policy checkpoints found in $POLICY_CKPT_DIR"
    exit 1
fi

echo "Test File: $test_file"
echo "Info File: $info_file"
echo "Student Checkpoint: $latest_student_ckpt"
echo "Policy Checkpoint: $latest_policy_ckpt"
echo "Output File: $OUTPUT_FILE"
echo "--------------------------------------------------------"

# Run the evaluation script
python ./eval_policy_joint.py \
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

echo "=== Evaluation Completed! Results saved to $OUTPUT_FILE ==="
