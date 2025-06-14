#!/bin/bash

# 训练执行脚本
# 此脚本不可修改

echo "Starting training..."

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/algorithm"

# 运行训练脚本
python algorithm/train_classification.py \
    --model pointnet2_cls_ssg \
    --batch_size 24 \
    --epoch 1 \
    --learning_rate 0.001 \
    --optimizer Adam \
    --log_dir output/log/classification \
    --decay_rate 1e-4 \
    --use_normals \
    --use_cpu
    
echo "Training completed!"
