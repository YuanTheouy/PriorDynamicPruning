
#!/bin/bash

# ================= 配置区域 =================
MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"
NUM_GPUS=8
# 剪枝配置
TOP_K_LAYERS=28
STRATEGY="uniform"
# ===========================================

CUDA_LIST=$(seq -s, 0 $(($NUM_GPUS - 1)))
CUDA_SPACE_LIST=$(seq -s " " 0 $(($NUM_GPUS - 1)))

echo "=== 开始 [剪枝] 评估流程 ==="
echo "模型路径: $MODEL_PATH"
echo "测试类别: $CATEGORY"
echo "剪枝策略: $STRATEGY, 保留层数: $TOP_K_LAYERS"

test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

exp_name_clean=$(basename "$MODEL_PATH")
temp_dir="./temp/${CATEGORY}-${exp_name_clean}_pruned_${STRATEGY}_${TOP_K_LAYERS}"
mkdir -p "$temp_dir"

echo ">>> [1/4] 正在切分测试数据..."
python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$CUDA_LIST"

echo ">>> [2/4] 正在运行推理评估..."
for i in $CUDA_SPACE_LIST
do
    echo "    启动 GPU $i ..."
    # [CRITICAL] We run evaluate_pruned.py instead of evaluate.py
    CUDA_VISIBLE_DEVICES=$i python -u ./evaluate_pruned.py \
        --base_model "$MODEL_PATH" \
        --info_file "$info_file" \
        --category ${CATEGORY} \
        --test_data_path "$temp_dir/${i}.csv" \
        --result_json_data "$temp_dir/${i}.json" \
        --batch_size 8 \
        --num_beams 50 \
        --max_new_tokens 256 \
        --top_k_layers $TOP_K_LAYERS \
        --strategy $STRATEGY &
done

wait
echo ">>> 所有 GPU 推理完成"

echo ">>> [3/4] 正在合并结果..."
output_dir="./results/${exp_name_clean}_pruned"
mkdir -p "$output_dir"

actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')

python ./merge.py \
    --input_path "$temp_dir" \
    --output_path "$output_dir/final_result_${CATEGORY}_${STRATEGY}_${TOP_K_LAYERS}.json" \
    --cuda_list "$actual_cuda_list"

echo ">>> [4/4] 正在计算最终指标 (NDCG, HR)..."
python ./calc.py \
    --path "$output_dir/final_result_${CATEGORY}_${STRATEGY}_${TOP_K_LAYERS}.json" \
    --item_path "$info_file"

echo "=== 评估完成！ ==="
