#!/bin/bash

# 配置文件验证脚本
# 检查所有配置文件是否符合要求

echo "===== 配置文件验证 ====="

# 检查必要的配置文件是否存在
echo "1. 检查配置文件存在性..."
if [ ! -f "config/config_train.yml" ]; then
    echo "❌ config/config_train.yml 不存在"
    exit 1
fi

if [ ! -f "config/config_test.yml" ]; then
    echo "❌ config/config_test.yml 不存在"
    exit 1
fi

echo "✅ 配置文件存在性检查通过"

# 检查Docker镜像名称是否统一
echo ""
echo "2. 检查Docker镜像名称统一性..."
TRAIN_IMAGE=$(grep "dockerImage:" config/config_train.yml | awk '{print $2}')
TEST_IMAGE=$(grep "dockerImage:" config/config_test.yml | awk '{print $2}')

echo "训练配置镜像: $TRAIN_IMAGE"
echo "测试配置镜像: $TEST_IMAGE"

if [ "$TRAIN_IMAGE" != "$TEST_IMAGE" ]; then
    echo "❌ 镜像名称不统一"
    exit 1
fi

echo "✅ Docker镜像名称统一性检查通过"

# 检查必须保留的参数
echo ""
echo "3. 检查必须保留参数..."

# 训练配置必须参数
REQUIRED_TRAIN_PARAMS=("DATASET_ROOT_PATH" "TASK_ROOT_PATH" "TRAIN_FILE_PATH" "VAL_FILE_PATH" "LOG_PATH" "CHECKPOINT_PATH")
for param in "${REQUIRED_TRAIN_PARAMS[@]}"; do
    if ! grep -q "$param:" config/config_train.yml; then
        echo "❌ config_train.yml 缺少必须参数: $param"
        exit 1
    fi
done

# 测试配置必须参数
REQUIRED_TEST_PARAMS=("SHOW_PATH" "TEST_FILE_PATH" "RESULT_PATH" "MODEL_PATH")
for param in "${REQUIRED_TEST_PARAMS[@]}"; do
    if ! grep -q "$param:" config/config_test.yml; then
        echo "❌ config_test.yml 缺少必须参数: $param"
        exit 1
    fi
done

echo "✅ 必须保留参数检查通过"

# 检查Dockerfile是否存在
echo ""
echo "4. 检查Docker相关文件..."
if [ ! -f "algorithm/docker/Dockerfile" ]; then
    echo "❌ algorithm/docker/Dockerfile 不存在"
    exit 1
fi

if [ ! -f "build_docker.sh" ]; then
    echo "❌ build_docker.sh 不存在"
    exit 1
fi

echo "✅ Docker相关文件检查通过"

# 显示配置摘要
echo ""
echo "5. 配置摘要:"
echo "镜像名称: $TRAIN_IMAGE"
echo "训练脚本: $(grep "scriptPath:" config/config_train.yml | awk '{print $2}')"
echo "测试脚本: $(grep "scriptPath:" config/config_test.yml | awk '{print $2}')"

echo ""
echo "===== 所有验证通过 ✅ ====="
echo "可以运行 ./build_docker.sh 开始构建镜像"
