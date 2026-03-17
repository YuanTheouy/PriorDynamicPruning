#!/bin/bash

# ================= Configuration =================

# 🔴 Option 1: Office Products
MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"

# 🔴 Option 2: Industrial and Scientific (Uncomment to use)
# MODEL_PATH="/workspace/ckpts/MiniOneRec/Industrial_ckpt"
# CATEGORY="Industrial_and_Scientific"

# Number of GPUs to use
NUM_GPUS=8

# Policy Training Hyperparameters
BATCH_SIZE=8 # This will be per-GPU batch size
EPOCHS=3
LEARNING_RATE=1e-4
TEMPERATURE=1.0
TOP_K_LAYERS=12

# Output Directory
OUTPUT_DIR="./policy_ckpts/${CATEGORY}"

# =================================================

echo "=== Starting Policy Network (Router) Distillation ==="
echo "Teacher Model: $MODEL_PATH"
echo "Category: $CATEGORY"
echo "Using $NUM_GPUS GPUs"

# Check if teacher model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Error: Teacher model path does not exist: $MODEL_PATH"
    exit 1
fi

# Automatically find training data and info file based on the run_test.sh logic
train_file=$(ls ./data/Amazon/train/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -f "$train_file" ]]; then
    echo "❌ Error: Training file not found for category $CATEGORY"
    exit 1
fi

if [[ ! -f "$info_file" ]]; then
    echo "❌ Error: Info file not found for category $CATEGORY"
    exit 1
fi

# Automatically find the latest student checkpoint
# We look for the student_ckpts directory for the current category
STUDENT_CKPT_DIR="./student_ckpts/${CATEGORY}"
if [[ ! -d "$STUDENT_CKPT_DIR" ]]; then
    echo "❌ Error: Student checkpoint directory not found: $STUDENT_CKPT_DIR"
    echo "Please run run_distill.sh first to train the student model."
    exit 1
fi

# Get the latest checkpoint (e.g. student_epoch_20.pt)
# Sort by modification time or name? Usually name implies epoch.
# Let's try to find the one with the highest epoch number.
latest_student_ckpt=$(ls -v ${STUDENT_CKPT_DIR}/student_epoch_*.pt 2>/dev/null | tail -1)

if [[ -z "$latest_student_ckpt" ]]; then
    echo "❌ Error: No student checkpoints found in $STUDENT_CKPT_DIR"
    exit 1
fi

echo "Training File: $train_file"
echo "Info File: $info_file"
echo "Student Checkpoint: $latest_student_ckpt"
echo "Output Directory: $OUTPUT_DIR"
echo "Target Layers: $TOP_K_LAYERS"
echo "--------------------------------------------------------"

# Run the policy distillation script using accelerate for multi-GPU
accelerate launch --num_processes $NUM_GPUS ./train_policy_distill.py \
    --teacher_model "$MODEL_PATH" \
    --student_ckpt "$latest_student_ckpt" \
    --train_file "$train_file" \
    --info_file "$info_file" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --top_k_layers $TOP_K_LAYERS \
    --output_dir "$OUTPUT_DIR"

echo "=== Policy Distillation Completed! Checkpoints saved to $OUTPUT_DIR ==="
