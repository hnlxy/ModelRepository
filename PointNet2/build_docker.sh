#!/bin/bash

# Docker镜像构建脚本
# 根据config_train.yml配置构建Docker镜像

echo "===== Docker镜像构建开始 ====="

# 从config_train.yml读取镜像名称
IMAGE_NAME=$(grep "dockerImage:" config/config_train.yml | awk '{print $2}')

if [ -z "$IMAGE_NAME" ]; then
    echo "错误：无法从config/config_train.yml读取镜像名称"
    exit 1
fi

echo "镜像名称: $IMAGE_NAME"

# 构建Docker镜像
echo "开始构建Docker镜像..."
docker build -t $IMAGE_NAME -f algorithm/docker/Dockerfile .

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "===== Docker镜像构建成功 ====="
    echo "镜像名称: $IMAGE_NAME"
    echo ""
    echo "验证镜像是否存在:"
    docker images | grep $(echo $IMAGE_NAME | cut -d':' -f1)
else
    echo "===== Docker镜像构建失败 ====="
    exit 1
fi

echo ""
echo "构建完成！可以使用以下命令运行容器:"
echo "docker run -it --gpus all -v \$(pwd):/workspace $IMAGE_NAME"