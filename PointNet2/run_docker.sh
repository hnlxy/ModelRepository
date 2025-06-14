#!/bin/bash

# Docker容器运行脚本
# 用于启动训练/测试容器

# 从配置文件读取镜像名称
IMAGE_NAME=$(grep "dockerImage:" config/config_train.yml | awk '{print $2}')

if [ -z "$IMAGE_NAME" ]; then
    echo "错误：无法从config/config_train.yml读取镜像名称"
    exit 1
fi

echo "使用镜像: $IMAGE_NAME"

# 检查镜像是否存在
if ! docker images | grep -q "$(echo "$IMAGE_NAME" | cut -d':' -f1)"; then
    echo "警告：镜像 $IMAGE_NAME 不存在，请先运行 ./build_docker.sh 构建镜像"
    exit 1
fi

# 检查GPU是否可用
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_ARGS="--gpus all"
    echo "检测到GPU，启用GPU支持"
else
    echo "未检测到GPU或GPU不可用，使用CPU模式"
fi

# 运行容器
echo "启动Docker容器..."
docker run -it \
    --platform linux/amd64 \
    $GPU_ARGS \
    -v "$(pwd)":/workspace \
    -v "$(pwd)/dataset":/workspace/dataset \
    -v "$(pwd)/input":/workspace/input \
    -v "$(pwd)/output":/workspace/output \
    --name pointnet2_container \
    --rm \
    "$IMAGE_NAME"

echo "容器已退出"
