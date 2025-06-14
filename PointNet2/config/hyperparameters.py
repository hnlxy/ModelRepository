# 超参数配置文件
# 用于调整模型训练的各种超参数

hyperparameters = {
    # 学习率相关
    "learning_rate": 0.001,
    "lr_scheduler": "StepLR",
    "lr_decay_step": 20,
    "lr_decay_rate": 0.7,
    
    # 优化器相关
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "momentum": 0.9,
    
    # 训练相关
    "batch_size": 24,
    "epochs": 200,
    "early_stopping_patience": 50,
    
    # 数据增强
    "use_random_rotation": True,
    "use_random_jitter": True,
    "use_random_scale": True,
    
    # 模型结构
    "dropout_rate": 0.5,
    "use_normals": True,
    "num_points": 1024,
    
    # 验证相关
    "validation_split": 0.1,
    "save_best_model": True
}
