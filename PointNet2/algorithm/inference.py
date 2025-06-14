#!/usr/bin/env python3
"""
推理预测脚本示例
用于对 input 文件夹中的点云数据进行分类预测
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# 添加项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

from data_utils.ModelNetDataLoader import pc_normalize
import importlib

def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--model', type=str, default='pointnet2_cls_ssg', help='model name')
    parser.add_argument('--batch_size', type=int, default=24, help='batch Size during training')
    parser.add_argument('--num_point', type=int, default=1024, help='point Number')
    parser.add_argument('--log_dir', type=str, default='../output/log/classification', help='log path')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate classification scores with voting')
    parser.add_argument('--input_dir', type=str, default='../input', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='../output/results', help='output results directory')
    return parser.parse_args()

def load_point_cloud(file_path):
    """加载点云数据"""
    if file_path.endswith('.txt'):
        # 尝试不同的分隔符
        try:
            points = np.loadtxt(file_path, delimiter=',')
        except:
            try:
                points = np.loadtxt(file_path)
            except:
                raise ValueError(f"Cannot load point cloud from {file_path}")
    elif file_path.endswith('.npy'):
        points = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # 只取前3列作为坐标，如果有法向量则取前6列
    if points.shape[1] >= 6:
        points = points[:, :6]
    else:
        points = points[:, :3]
    
    return points

def preprocess_point_cloud(points, num_point=1024, use_normals=False):
    """预处理点云数据"""
    # 如果点数超过num_point，随机采样
    if points.shape[0] >= num_point:
        choice = np.random.choice(points.shape[0], num_point, replace=False)
    else:
        # 如果点数不足，随机重复采样
        choice = np.random.choice(points.shape[0], num_point, replace=True)
    
    points = points[choice, :]
    points[:, :3] = pc_normalize(points[:, :3])
    
    # 如果模型需要法向量但输入数据没有，则添加零向量
    if use_normals and points.shape[1] == 3:
        normals = np.zeros((points.shape[0], 3))
        points = np.concatenate([points, normals], axis=1)
    elif not use_normals and points.shape[1] > 3:
        # 如果模型不需要法向量但输入有，则只取坐标
        points = points[:, :3]
    
    return points

def predict_single_file(model, classifier, file_path, args):
    """对单个文件进行预测"""
    # 加载并预处理点云
    points = load_point_cloud(file_path)
    points = preprocess_point_cloud(points, args.num_point, args.use_normals)
    
    # 转换为tensor
    points = torch.from_numpy(points).float()
    points = points.unsqueeze(0)  # 添加batch维度: (1, num_points, 3或6)
    points = points.transpose(2, 1)  # 转置到 (1, 3或6, num_points)
    
    if torch.cuda.is_available():
        points = points.cuda()
    
    # 预测
    with torch.no_grad():
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
    
    return pred_choice.cpu().numpy()[0]

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = importlib.import_module(args.model)
    classifier = model.get_model(40, normal_channel=args.use_normals)
    
    if torch.cuda.is_available():
        classifier = classifier.cuda()
    
    # 加载训练好的权重
    checkpoint_path = os.path.join(args.log_dir, 'checkpoints', 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {checkpoint_path}')
    else:
        print(f'Warning: Model checkpoint not found at {checkpoint_path}')
        return
    
    classifier.eval()
    
    # 加载类别名称
    class_names_path = os.path.join(ROOT_DIR, 'dataset/data/modelnet40_normal_resampled/modelnet40_shape_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 遍历输入文件夹中的所有文件
    input_dir = Path(args.input_dir)
    results = []
    
    for file_path in input_dir.glob('**/*.txt'):
        try:
            pred_class = predict_single_file(model, classifier, str(file_path), args)
            class_name = class_names[pred_class]
            result = {
                'file': str(file_path.relative_to(input_dir)),
                'predicted_class': pred_class,
                'class_name': class_name
            }
            results.append(result)
            print(f'{file_path.name}: {class_name} (class {pred_class})')
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'prediction_results.txt')
    with open(output_file, 'w') as f:
        f.write('File\tPredicted_Class\tClass_Name\n')
        for result in results:
            f.write(f'{result["file"]}\t{result["predicted_class"]}\t{result["class_name"]}\n')
    
    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main()
