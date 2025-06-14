# PointNet2 项目目录结构说明

本项目已按照标准的机器学习项目结构重新组织，便于训练、测试和部署。

## 目录结构

```
docker-PointNet2/
├── algorithm/           # 算法文件和镜像文件
│   ├── models/         # 模型定义文件
│   ├── data_utils/     # 数据处理工具
│   ├── visualizer/     # 可视化工具
│   ├── provider.py     # 数据增强提供器
│   ├── train_classification.py  # 训练脚本
│   └── test_classification.py   # 测试脚本
├── config/             # 参数配置（用于超参数配置）
│   ├── config.py       # 基本配置
│   └── hyperparameters.py  # 超参数配置
├── dataset/            # 数据集位置（用于训练）
│   └── data/
│       ├── ModelNet40/
│       └── modelnet40_normal_resampled/
├── input/              # 输入数据（用于推理预测）
│   ├── README.md
│   └── batch_inference/
├── output/             # 输出文件（模型权重文件，推理结果、评估结果）
│   ├── log/            # 训练日志
│   ├── models/         # 保存的模型权重
│   ├── results/        # 推理结果
│   └── evaluation/     # 评估结果
├── train.sh           # 训练执行脚本（不可修改）
├── test.sh            # 评估测试执行脚本（不可修改）
├── README.md          # 项目说明文档
└── LICENSE            # 许可证文件
```

## 使用方法

### 1. 训练模型
```bash
# 执行训练脚本
./train.sh
```

### 2. 测试评估
```bash
# 执行测试脚本
./test.sh
```

### 3. 推理预测
将要预测的点云数据放入 `input/` 文件夹，然后运行相应的推理脚本。

## 配置说明

- `config/config.py`: 包含基本的模型和数据路径配置
- `config/hyperparameters.py`: 包含训练超参数的详细配置

## 输出说明

- `output/log/`: 存放训练和测试的日志文件
- `output/models/`: 存放训练好的模型权重文件
- `output/results/`: 存放推理预测的结果
- `output/evaluation/`: 存放模型评估的结果

## 注意事项

1. `train.sh` 和 `test.sh` 脚本不可修改
2. 训练前请确保数据集已正确放置在 `dataset/data/` 目录下
3. 可通过修改 `config/` 目录下的配置文件来调整模型参数
