from mmengine.config import Config
from mmengine.runner import Runner

def main():
    config_path = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    cfg = Config.fromfile(config_path)

    # 工作目录
    cfg.work_dir = './work_dirs/hrnet_coco_custom'

    # 启用 AMP（可选）
    from mmengine.optim import AmpOptimWrapper
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # ✅ 添加 TensorBoard 日志钩子
    cfg.default_hooks.logger = dict(
        type='LoggerHook',
        interval=10,  # 每多少个 iteration 记录一次
        log_metric_by_epoch=True
    )

    # ✅ 添加 TensorBoard 可视化支持
    cfg.visualizer = dict(
        type='TensorboardVisBackend',
        save_dir=cfg.work_dir,
        name='tensorboard_vis'
    )
    cfg.log_processor = dict(
        type='LogProcessor',
        window_size=50,
        by_epoch=True
    )

    # ✅ 启用默认训练 loss/AP 可视化
    cfg.default_hooks.visualization = dict(
        type='PoseVisualizationHook',
        enable=False  # 训练时不需要图像弹窗
    )

    # 启动训练
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
