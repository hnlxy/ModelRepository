"""
Author: Benny
Date: Nov 2019
"""
# 修复 numpy 兼容性问题
import os
import sys

# 设置环境变量避免numpy兼容性问题
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
# 处理 numpy._core 模块问题
try:
    import numpy._core as np_core
except ImportError:
    try:
        import numpy.core as np_core
        sys.modules['numpy._core'] = np_core
    except:
        pass

# 修复 pickle 加载问题
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    # parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--model', type=str, default='pointnet2_cls_ssg', help='Model name')
    
    parser.add_argument('--log_dir', type=str, default="pointnet2_cls_ssg", help='Experiment root')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1, args=None):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        if args.use_cpu:
            vote_pool = torch.zeros(target.size()[0], num_class)
        else:
            vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = os.path.join(ROOT_DIR, 'dataset/data/modelnet40_normal_resampled/')

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    # 处理 numpy 兼容性问题和 CPU 模式
    checkpoint_path = str(experiment_dir) + '/checkpoints/best_model.pth'
    
    # 检查模型文件是否存在
    if not os.path.exists(checkpoint_path):
        log_string(f'Model checkpoint not found at: {checkpoint_path}')
        log_string('Please train the model first using train.sh')
        return
    
    log_string(f'Loading model from: {checkpoint_path}')
    
    try:
        # 使用安全的加载方式，避免numpy兼容性问题
        if args.use_cpu:
            # 在CPU模式下使用特殊的加载方式
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
        log_string('Model checkpoint loaded successfully')
        
    except Exception as e:
        log_string(f'Error loading checkpoint: {e}')
        log_string('Trying alternative loading method...')
        
        try:
            # 备用加载方法
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        except Exception as e2:
            log_string(f'Alternative loading also failed: {e2}')
            log_string('The model checkpoint may be corrupted or incompatible.')
            return
    
    try:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        log_string(f'Error loading model state dict: {e}')
        # 尝试直接加载模型参数
        try:
            classifier.load_state_dict(checkpoint)
        except Exception as e2:
            log_string(f'Error loading model directly: {e2}')
            raise e2

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class, args=args)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
