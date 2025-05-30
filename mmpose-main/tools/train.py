# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmpose.utils import register_all_modules  # 注册 MMPose 模块

# 设置当前工作目录为 MMPose 主目录（可选）
os.chdir('C:/models/mmpose-main')

def main():
    # 注册所有 MMPose 模块
    register_all_modules()

    # 手动设置参数
    config_path = r"C:\models\mmpose-main\configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    work_dir = './work_dirs/hrnet_coco'
    resume = False  # 可以是 True 或 'auto'
    use_amp = True
    auto_scale_lr = False
    show = False
    show_dir = None
    no_validate = False

    # 加载配置文件
    cfg = Config.fromfile(config_path)

    # 覆盖参数
    cfg.work_dir = work_dir
    if resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif isinstance(resume, str):
        cfg.resume = True
        cfg.load_from = resume
    else:
        cfg.resume = False
        cfg.load_from = None

    if use_amp:
        from mmengine.optim import AmpOptimWrapper
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    if show or show_dir is not None:
        assert 'visualization' in cfg.default_hooks, \
            'You must define visualization hook in default_hooks'
        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = show
        cfg.default_hooks.visualization.out_dir = show_dir
        cfg.default_hooks.visualization.interval = 1
        cfg.default_hooks.visualization.wait_time = 1

    # 日志与可视化设置（TensorBoard）
    cfg.default_hooks.logger = dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True
    )

    cfg.visualizer = dict(
        type='PoseLocalVisualizer',
        vis_backends=[
            dict(type='TensorboardVisBackend', save_dir=osp.join(cfg.work_dir, 'tensorboard'))
        ],
        name='visualizer'
    )

    cfg.log_processor = dict(
        type='LogProcessor',
        window_size=50,
        by_epoch=True
    )

    # 补充 data_preprocessor（兼容旧版本）
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor', cfg.get('preprocess_cfg', {}))

    # 构建 Runner 并训练
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
