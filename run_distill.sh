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

# Training Hyperparameters
BATCH_SIZE=8 # This will be per-GPU batch size
EPOCHS=20
LEARNING_RATE=5e-5
TEMPERATURE=1.0

# Output Directory
OUTPUT_DIR="./student_ckpts/${CATEGORY}"

# =================================================

echo "=== Starting Teacher-Student Single Layer Distillation ==="
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

echo "Training File: $train_file"
echo "Info File: $info_file"
echo "Output Directory: $OUTPUT_DIR"
echo "--------------------------------------------------------"

# Run the distillation script using accelerate for multi-GPU
accelerate launch --num_processes $NUM_GPUS ./train_student_distill.py \
    --teacher_model "$MODEL_PATH" \
    --train_file "$train_file" \
    --info_file "$info_file" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --output_dir "$OUTPUT_DIR"

echo "=== Distillation Completed! Checkpoints saved to $OUTPUT_DIR ==="
