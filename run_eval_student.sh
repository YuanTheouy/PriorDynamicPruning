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

# GPU to use for evaluation (single GPU is usually enough for inference)
GPU_ID=0

# Evaluation Hyperparameters
BATCH_SIZE=32
TOP_K=50
OUTPUT_FILE="./results/student_${CATEGORY}_epoch${EPOCH}_eval.json"

# =================================================

echo "=== Starting Student Model Evaluation ==="
echo "Teacher Model (for Tokenizer): $TEACHER_MODEL_PATH"
echo "Student Checkpoint: $STUDENT_CKPT"
echo "Category: $CATEGORY"
echo "Using GPU: $GPU_ID"

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
echo "Output File: $OUTPUT_FILE"
echo "--------------------------------------------------------"

# Run the evaluation script
CUDA_VISIBLE_DEVICES=$GPU_ID python -u ./eval_student.py \
    --teacher_model "$TEACHER_MODEL_PATH" \
    --student_ckpt "$STUDENT_CKPT" \
    --test_file "$test_file" \
    --info_file "$info_file" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --top_k $TOP_K \
    --output_file "$OUTPUT_FILE"

echo "=== Evaluation Completed! ==="
