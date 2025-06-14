#!/bin/bash

# 推理预测执行脚本
# 用于对input文件夹中的点云数据进行分类预测

echo "Starting inference..."

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/algorithm"

# 运行推理脚本
python algorithm/inference.py \
    --model pointnet2_cls_ssg \
    --batch_size 24 \
    --num_point 1024 \
    --log_dir output/log/classification \
    --input_dir input \
    --output_dir output/results \
    --use_normals

echo "Inference completed!"
echo "Results saved to output/results/"
