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
TOP_K_LAYERS=21
ROUTER_INPUT_SOURCE=${ROUTER_INPUT_SOURCE:-teacher_prefix}
PREFIX_DEPTH=${PREFIX_DEPTH:-4}
TRAIN_STUDENT=${TRAIN_STUDENT:-0}
ORACLE_CACHE=${ORACLE_CACHE:-}
MASK_DISTILL_WEIGHT=${MASK_DISTILL_WEIGHT:-0.0}
RISK_RANKING_WEIGHT=${RISK_RANKING_WEIGHT:-0.0}
KL_DISTILL_WEIGHT=${KL_DISTILL_WEIGHT:-1.0}

# Output Directory
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    if [[ "$ROUTER_INPUT_SOURCE" == "teacher_prefix" ]]; then
        OUTPUT_DIR="./policy_ckpts/opal_prefix${PREFIX_DEPTH}/${CATEGORY}"
    else
        OUTPUT_DIR="./policy_ckpts/student_router/${CATEGORY}"
    fi
fi

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

student_flags=()
latest_student_ckpt=""
if [[ "$ROUTER_INPUT_SOURCE" == "student" ]]; then
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
    student_flags+=(--student_ckpt "$latest_student_ckpt")
    if [[ "$TRAIN_STUDENT" == "1" ]]; then
        student_flags+=(--train_student)
    fi
fi

echo "Training File: $train_file"
echo "Info File: $info_file"
echo "Router Input Source: $ROUTER_INPUT_SOURCE"
echo "Prefix Depth: $PREFIX_DEPTH"
if [[ -n "$latest_student_ckpt" ]]; then
    echo "Student Checkpoint: $latest_student_ckpt"
fi
echo "Output Directory: $OUTPUT_DIR"
echo "Target Layers: $TOP_K_LAYERS"
echo "Train Student: $TRAIN_STUDENT"
echo "Oracle Cache: ${ORACLE_CACHE:-none}"
echo "--------------------------------------------------------"

oracle_flags=()
if [[ -n "$ORACLE_CACHE" ]]; then
    oracle_flags+=(--oracle_cache "$ORACLE_CACHE")
    oracle_flags+=(--mask_distill_weight "$MASK_DISTILL_WEIGHT")
    oracle_flags+=(--risk_ranking_weight "$RISK_RANKING_WEIGHT")
    oracle_flags+=(--kl_distill_weight "$KL_DISTILL_WEIGHT")
fi

# Run the policy distillation script using accelerate for multi-GPU
accelerate launch --num_processes $NUM_GPUS ./train_policy_distill.py \
    --teacher_model "$MODEL_PATH" \
    "${student_flags[@]}" \
    --train_file "$train_file" \
    --info_file "$info_file" \
    --category "$CATEGORY" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --top_k_layers $TOP_K_LAYERS \
    --router_input_source "$ROUTER_INPUT_SOURCE" \
    --prefix_depth "$PREFIX_DEPTH" \
    "${oracle_flags[@]}" \
    --output_dir "$OUTPUT_DIR" \

echo "=== Policy Distillation Completed! Checkpoints saved to $OUTPUT_DIR ==="
