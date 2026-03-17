#!/bin/bash

# ================= 配置区域 =================

# 🔴 选项 1: 如果你想测 Office Products (对应 ckpts/MiniOneRec/Office_ckpt)
MODEL_PATH="/workspace/ckpts/MiniOneRec/Office_ckpt"
CATEGORY="Office_Products"

# 🔴 选项 2: 如果你想测 Industrial (对应 ckpts/MiniOneRec/Industrial_ckpt)
# (想测哪个就把上面的注释掉，把下面的取消注释)
# MODEL_PATH="/workspace/ckpts/MiniOneRec/Industrial_ckpt"
# CATEGORY="Industrial_and_Scientific"

# 🔴 GPU 设置
# 设置你要使用的 GPU 数量 (例如 1, 2, 4, 8)
NUM_GPUS=8
# ===========================================

# 自动生成 GPU 列表字符串
CUDA_LIST=$(seq -s, 0 $(($NUM_GPUS - 1)))
CUDA_SPACE_LIST=$(seq -s " " 0 $(($NUM_GPUS - 1)))

echo "=== 开始评估流程 ==="
echo "模型路径: $MODEL_PATH"
echo "使用 GPU: 0 到 $(($NUM_GPUS - 1)) (共 $NUM_GPUS 个)"
echo "测试类别: $CATEGORY"

# 检查模型路径是否存在
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo "请检查路径是否正确，或是否需要修改脚本中的 MODEL_PATH 变量"
    exit 1
fi

# 自动查找数据文件
test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -f "$test_file" ]]; then
    echo "❌ 错误: 未找到测试文件 $test_file"
    exit 1
fi

if [[ ! -f "$info_file" ]]; then
    echo "❌ 错误: 未找到 Info 文件 $info_file"
    exit 1
fi

# 创建临时目录
exp_name_clean=$(basename "$MODEL_PATH")
temp_dir="./temp/${CATEGORY}-${exp_name_clean}"
mkdir -p "$temp_dir"

# 1. 切分数据 (Split Data)
echo ">>> [1/4] 正在切分测试数据..."
python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$CUDA_LIST"

# 2. 并行评估 (Parallel Evaluation)
echo ">>> [2/4] 正在运行推理评估..."
for i in $CUDA_SPACE_LIST
do
    echo "    启动 GPU $i ..."
    CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
        --base_model "$MODEL_PATH" \
        --info_file "$info_file" \
        --category ${CATEGORY} \
        --test_data_path "$temp_dir/${i}.csv" \
        --result_json_data "$temp_dir/${i}.json" \
        --batch_size 8 \
        --num_beams 50 \
        --max_new_tokens 256 &
done

# 等待所有后台任务完成
wait
echo ">>> 所有 GPU 推理完成"

# 3. 合并结果 (Merge Results)
echo ">>> [3/4] 正在合并结果..."
output_dir="./results/${exp_name_clean}"
mkdir -p "$output_dir"

# 获取实际生成的 json 文件列表对应的 GPU id
actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')

python ./merge.py \
    --input_path "$temp_dir" \
    --output_path "$output_dir/final_result_${CATEGORY}.json" \
    --cuda_list "$actual_cuda_list"

# 4. 计算指标 (Calculate Metrics)
echo ">>> [4/4] 正在计算最终指标 (NDCG, HR)..."
python ./calc.py \
    --path "$output_dir/final_result_${CATEGORY}.json" \
    --item_path "$info_file"

echo "=== 评估完成！结果已保存至 $output_dir ==="