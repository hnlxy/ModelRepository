#!/bin/bash

# 评估测试执行脚本
# 此脚本不可修改

echo "Starting evaluation..."

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/algorithm"

# 运行测试脚本
python algorithm/test_classification.py \
    --log_dir output/log/classification \
    --model pointnet2_cls_ssg \
    --batch_size 24 \
    --num_point 1024 \
    --use_normals\
    --use_cpu

echo "Evaluation completed!"
echo "Results saved to output/log/classification/"
