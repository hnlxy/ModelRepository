#!/usr/bin/env python3
"""
配置文件读取工具
用于从YAML配置文件中读取参数并应用到训练/测试脚本
"""

import yaml
import argparse
import os
import sys

def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"错误：无法读取配置文件 {config_path}: {e}")
        return None

def merge_args_with_config(args, config):
    """将配置文件参数与命令行参数合并"""
    # 将配置文件中的参数设置为默认值
    for key, value in config.items():
        if hasattr(args, key.lower()):
            # 如果命令行没有指定该参数，使用配置文件的值
            if getattr(args, key.lower()) is None:
                setattr(args, key.lower(), value)
    return args

def print_config_summary(config):
    """打印配置摘要"""
    print("=" * 50)
    print("配置文件参数摘要:")
    print("=" * 50)
    
    # 分类显示参数
    path_params = []
    training_params = []
    model_params = []
    other_params = []
    
    for key, value in config.items():
        if 'PATH' in key or 'path' in key.lower():
            path_params.append((key, value))
        elif key in ['BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE', 'OPTIMIZER', 'DECAY_RATE']:
            training_params.append((key, value))
        elif key in ['MODEL_NAME', 'NUM_POINT', 'NUM_CLASSES', 'USE_NORMALS']:
            model_params.append((key, value))
        else:
            other_params.append((key, value))
    
    if path_params:
        print("\n路径参数（必须保留）:")
        for key, value in path_params:
            print(f"  {key}: {value}")
    
    if training_params:
        print("\n训练参数:")
        for key, value in training_params:
            print(f"  {key}: {value}")
    
    if model_params:
        print("\n模型参数:")
        for key, value in model_params:
            print(f"  {key}: {value}")
    
    if other_params:
        print("\n其他参数:")
        for key, value in other_params:
            print(f"  {key}: {value}")
    
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='配置文件读取工具')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--show_summary', action='store_true', help='显示配置摘要')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    if config is None:
        sys.exit(1)
    
    if args.show_summary:
        print_config_summary(config)
