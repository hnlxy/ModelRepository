# Docker PointNet2 配置使用说明

## 📁 配置文件说明

### 1. 训练配置 (`config/config_train.yml`)
- **Docker镜像**: `harbor.yzai/cgf-ml-algorithm/ht/ultralytics:v1_flask`
- **必须保留参数**:
  - `DATASET_ROOT_PATH`: 数据集根路径
  - `TASK_ROOT_PATH`: 任务根路径
  - `TRAIN_FILE_PATH`: 训练文件路径
  - `VAL_FILE_PATH`: 验证文件路径
  - `LOG_PATH`: 日志路径
  - `CHECKPOINT_PATH`: 模型保存路径

### 2. 测试配置 (`config/config_test.yml`)
- **Docker镜像**: `harbor.yzai/cgf-ml-algorithm/ht/ultralytics:v1_flask`
- **必须保留参数**:
  - `SHOW_PATH`: 可视化结果路径
  - `TEST_FILE_PATH`: 测试文件路径
  - `RESULT_PATH`: 结果输出路径
  - `MODEL_PATH`: 模型路径

## 🐳 Docker使用流程

### 步骤1: 验证配置
```bash
./validate_config.sh
```

### 步骤2: 构建Docker镜像
```bash
./build_docker.sh
```

### 步骤3: 验证镜像构建
```bash
docker images | grep harbor.yzai/cgf-ml-algorithm/ht/ultralytics
```

### 步骤4: 运行容器
```bash
./run_docker.sh
```

## 🔧 配置工具

### 查看配置摘要
```bash
# 查看训练配置
python config/config_loader.py --config config/config_train.yml --show_summary

# 查看测试配置
python config/config_loader.py --config config/config_test.yml --show_summary
```

## 📋 目录结构要求

确保以下目录存在：
- `dataset/data/` - 数据集存放目录
- `input/` - 推理输入目录
- `output/log/classification/` - 训练日志目录
- `output/models/` - 模型保存目录
- `output/results/` - 结果输出目录
- `output/visualization/` - 可视化结果目录

## ⚠️ 注意事项

1. **镜像名称统一**: 所有配置文件中的`dockerImage`必须保持一致
2. **必须保留参数**: 标记为【必须保留】的参数不可删除
3. **路径映射**: 容器内路径统一使用`/workspace`前缀
4. **GPU支持**: 构建的镜像支持CUDA，需要`--gpus all`参数

## 🚀 快速开始

```bash
# 1. 验证配置
./validate_config.sh

# 2. 构建镜像
./build_docker.sh

# 3. 运行训练（在容器内）
./train.sh

# 4. 运行测试（在容器内）
./test.sh

# 5. 运行推理（在容器内）
./inference.sh
```
