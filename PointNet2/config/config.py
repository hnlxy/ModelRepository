# PointNet2 分类模型配置文件

# 训练参数
BATCH_SIZE = 24
EPOCHS = 200
LEARNING_RATE = 0.001
DECAY_RATE = 1e-4
OPTIMIZER = "Adam"

# 模型参数
MODEL_NAME = "pointnet2_cls_ssg"
NUM_POINT = 1024
NUM_CLASSES = 40
NORMAL = True

# 数据集路径
DATASET_PATH = "dataset/data"
MODELNET40_PATH = "dataset/data/modelnet40_normal_resampled"

# 输出路径
LOG_DIR = "output/log/classification"
MODEL_SAVE_PATH = "output/models"

# 输入数据路径（用于推理）
INPUT_PATH = "input"

# GPU设置
USE_GPU = True
GPU_ID = 0
