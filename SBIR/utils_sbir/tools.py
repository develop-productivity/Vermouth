
import torch
import math
from scipy.spatial.distance import cdist
import clip
import torch.nn.functional as F
import numpy as np

# train and test config file
# =======================================
# TODO:
# =======================================


sketchy_split1_settings = {
    'num_classes':100,
    'root_dir':'/data/sydong/datasets/SBIR/Sketchy/',
    'zero_version':'zeroshot1'
}

sketchy_split2_settings = {
    'num_classes':104,
    'root_dir':'/data/sydong/datasets/SBIR/Sketchy/',
    'zero_version':'zeroshot2'
}

tuberlin_setting = {
    'num_classes':220,
    'root_dir':'/data/sydong/datasets/SBIR/TUBerlin/',
    'zero_version':'zeroshot'
}

quickdraw_setting = {
    'num_classes':80,
    'root_dir':'/data/sydong/datasets/SBIR/QuickDraw/',
    'zero_version':'zeroshot'
}

def get_args(args):
    if args.dataset == 'sketchy_split1':
        setting = sketchy_split1_settings
    elif args.dataset == 'sketchy_split2':
        setting = sketchy_split2_settings
    elif args.dataset == 'tuberlin':
        setting = tuberlin_setting
    elif args.dataset == 'quickdraw':
        setting = quickdraw_setting
    args.root_dir = setting['root_dir']
    args.num_classes = setting['num_classes']
    args.zero_version = setting['zero_version']
    return args

def random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = True

def load_clip_to_cpu(args):
    if args.arch_teacher == 'RN50':
        arch = 'RN50'
        args.clip_feature = 1024
    elif args.arch_teacher == 'RN101':
        arch = 'RN101'
    elif args.arch_teacher == 'RN50x4':
        arch = 'RN50x4'
        args.clip_feature = 640
    elif args.arch_teacher == 'ViT-B16':
        arch = 'ViT-B/16'
        args.clip_feature = 512
    elif args.arch_teacher == 'ViT-L14':
        arch = 'ViT-L/14'
        args.clip_feature = 768
    elif args.arch_teacher == 'ViT-L14_336px':
        arch = 'ViT-L/14@336px'
        args.clip_feature = 768
    clip_model, process= clip.load(arch, download_root='/data/sydong/SBIR/SAKE/SAKE_2/model/clip_model', device='cpu')  # clip 作为teacher网络
    return clip_model, process


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
    # epoch_curr = min(epoch, 20)
    # lr = args.lr * math.pow(0.001, float(epoch_curr)/ 20 )

    # no ratio no warm up
    # lr = np.array(args.lr) * math.pow(0.001, float(epoch) / args.epochs)  # 指数学习率下降
    # print('epoch: {}, lr: {}'.format(epoch, lr))
    # for i, param_group in enumerate(optimizer.param_groups):
    #         param_group['lr'] = lr[i]

    # warm up no ratio
    if epoch in [0, 1, 2]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    else:
        lr = np.array(args.lr) * math.pow(0.001, float(epoch) / args.epochs)  # 指数学习率下降
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[i]

    # # ratio no warm up
    # lr = np.array(args.lr) * math.pow(0.001, float(epoch) / args.epochs)  # 指数学习率下降
    # print('epoch: {}, lr: {}'.format(epoch, lr))
    # for i, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = lr[i]

    # warm up and ratio
    # if epoch in [0, 1, 2]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-5
    # else:
    #     lr = np.array(args.lr) * math.pow(0.001, float(epoch) / args.epochs)  # 指数学习率下降
    #     print('epoch: {}, lr: {}'.format(epoch, lr))
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         param_group['lr'] = lr[i]
