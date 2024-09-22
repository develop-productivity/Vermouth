import argparse
import numpy as np
import torch
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import random

from datasets_few_shot import build_dataset
from datasets_few_shot.utils import build_data_loader
from data.datasets import get_prompts
from cls_model.model import DiffusionClassifier
from cls_model.model_pre import DiffusionClassifierPre

import sys
sys.path.append('/data/sydong/diffusion/diffusion_features')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# TODO:
# 1. 上采样
# 2. k-means
# 3. 输出mask

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=interpolation),
        # transforms.RandomCrop((256, 512)),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        transforms.Normalize(mean=(0.5), std=(0.5))  # align with diffusion pre-train
    ])
    return transform

def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)

def save_image(data):
    """
    data (numpy.Array):
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(data)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname="sofa_feature.jpg")

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    else:
        return False

def save_checkpoint(state_dict, file_name):
    torch.save(state_dict, file_name+'.pth')


def load_checkpoint(model, pth_path):
    checkpoint = torch.load(pth_path)
    model.classifier.load_state_dict(checkpoint['classifier'])
    if 'expert' in checkpoint and checkpoint['expert'] != {}:
        model.expert.load_state_dict(checkpoint['expert'])
    print(f"Loaded checkpoint from {pth_path}")

def main():
    parser = argparse.ArgumentParser()

    # run args
    parser.add_argument('--version', type=str, default='1-5', help='Stable Diffusion model version')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['oxford_pets', 'flowers', 'mnist', 'cifar10', 'food101', 'caltech101', 'imagenet','dtd', 'ucf101', 
                                 'stanford_cars', 'aircraft', 'eurosat', 'sun397'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    parser.add_argument('--img_size', type=int, default=256, choices=(256, 512, 224, 192))
    parser.add_argument('--max_attn_size', type=int, default=16, choices=(32, 64, 16))
    parser.add_argument('--num_epoches', type=int, default=100, help='Name of train epoches')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--cuda', type=int, default=3, help='cuda devices')
    parser.add_argument('--dtype', type=str, default='float32', choices=('float16', 'float32'),
                        help='Model data type to use')

    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--model_path', type=str, default='../stable-diffusion-v1-5', help='Number of workers to split the dataset across')
    parser.add_argument('--time_step', type=int, default=[200], nargs='+', help='which time steps to use')
    parser.add_argument('--place_in_unet', type=str, default=['up', 'mid'], nargs='+', help='which time steps to use')
    parser.add_argument('--inv_method', type=str, default='ddpm_schedule', choices=['ddim_inv', 'ddpm_schedule'], help='classifier')
    parser.add_argument('--upsample_mode', type=str, default='bilinear', help='Timesteps to compute features')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature-coefficient')
    parser.add_argument('--second_last_layer', type=str2bool, default='yes', help='')

    parser.add_argument('--classifier', type=str, default='fuse_expert', choices=['linear', 'adapter', 'fuse', 'no_fuse', 'fuse_expert'], help='classifier')
    parser.add_argument('--fuse_arch', type=str, default='tiny', choices=['pico', 'tiny', 'small'], help='classifier')
    parser.add_argument('--expert', type=str, default='resnet', choices=['resnet', 'no_expert'], help='')
    parser.add_argument('--frozen_expert', type=str2bool, default='yes', help='')
    parser.add_argument('--frozen_backbobe', type=str2bool, default='yes', help='')
    parser.add_argument('--fix_noise', type=str2bool, default='yes', help='')
    parser.add_argument('--cross_attn', type=str2bool, default='no', help='')
    parser.add_argument('--attn_selector', type=str, default='up_cross', help='Timesteps to compute features')
    parser.add_argument('--checkpoint_path', type=str, default='experiments/Classification/diffusion1-5/frozen/wo_attn/time_steps/fuse_expert/wo-finetune_imagenet_lr-1e-05_batch-64_16shot_diffusion_time-steps-999up-mid_ddpm_schedule_tiny_clip-text-proj_frozen_epoch-10.pth', help='Timesteps to compute features')



    parser.add_argument('--model_type', type=str, default='diffusion', choices=['diffusion','clip', 'mae', 'dino', 'swin', 'swinv2_192', 'swinv2_256'], help='model type')
    parser.add_argument('--log_dir', type=str, default='experiments/Classification')
    parser.add_argument('--commit', type=str, default='diffusion1-5', choices=['diffusion1-5','backbone'])
    parser.add_argument('--content', type=str, default='main', help='content of the experiment')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    # setup_seed(args.seed)
    interpolation = INTERPOLATIONS['bicubic']
    device = f"cuda:{args.cuda}"
    args.device = device

    generator = torch.Generator(device=device).manual_seed(8888)
    transform = get_transform(interpolation, args.img_size)

    
    prompts = get_prompts(args.dataset)
    dataset = build_dataset(args.dataset, 16, transform)
    if args.dataset == 'imagenet':
        train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    else:
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=transform, is_train=True, shuffle=True)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=args.batch_size, is_train=False, tfm=transform, shuffle=False)

    file_name = f"wo-finetune_{args.dataset}_lr-{args.lr}_batch-{args.batch_size}_16shot_{args.model_type}_time-steps-"
    file_name += "-".join([str(i) for i in args.time_step])
    file_name += "-".join([str(i) for i in args.place_in_unet])
    # file_name += f'_{args.inv_method}_{args.fuse_arch}'
    file_name += f'_{args.inv_method}_{args.fuse_arch}_clip-text-proj'
    # if args.l1_loss_weight != 0.0:
    #     file_name += f'_l1-{args.l1_loss_weight}-beta-0.02'
    # if args.use_bn:
    #     file_name += '_BN'
    if args.frozen_expert and args.expert == 'resnet':
        file_name += '_frozen'
    if not args.fix_noise:
        file_name += '_wo_fix-noise'
        generator = None
    model = DiffusionClassifierPre(torch.LongTensor(args.time_step), generator=generator, args=args, num_classes=dataset._num_classes).to(device)
    if args.checkpoint_path:
        load_checkpoint(model, args.checkpoint_path)
    model.registe_text_features(prompts)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    scaler = GradScaler()
    start = time.time()
    if args.frozen_backbobe:
        log_dir = os.path.join(args.log_dir, args.commit, 'frozen', args.content, args.classifier)
    else:
        log_dir = os.path.join(args.log_dir, args.commit, 'un_frozen', args.content, args.classifier)
    os.makedirs(log_dir, exist_ok=True)
    acc1_test = test(model, test_loader, prompts, args)
    print("Test Acc@1: {:.4f}".format(acc1_test))
    with open(os.path.join(log_dir, file_name + '.txt'), 'a') as f:
                f.writelines('Test Acc@1: {} \n'.format(acc1_test))
    # for epoch in range(args.num_epoches):
    #     # train(model, train_loader, prompts, optimizer, epoch, scaler, args)
    #     if epoch in [9, 19, 99]:
    #         acc1_test = test(model, test_loader, prompts, args)
    #         print("Test Acc@1: {:.4f}".format(acc1_test))
    #         with open(os.path.join(log_dir, file_name + '.txt'), 'a') as f:
    #             f.writelines('Epoch: {} Test Acc@1: {} \n'.format(epoch, acc1_test))

    #         save_checkpoint({'classifier':model.classifier.state_dict(),
    #                         'expert':model.expert.state_dict() if args.expert == 'resnet' else {},
    #                          }, os.path.join(log_dir, file_name + '_epoch-{}'.format(epoch+1)))
    # zero_shot(model, test_loader, prompts)
    end = time.time()
    print("time = {}".format(end - start))

def train(model, train_loader, prompts, optimizer, epoch, scaler, args):
    pbar = tqdm(train_loader, ncols=100)
    for inputs, labels in pbar:
            optimizer.zero_grad()        
            # with autocast:
            prompt=[prompts[idx] for idx in labels]

            # random prompt select
            length = len(prompt[0])
            rand_idx = np.random.randint(length, size=len(prompt))   
            prompt = [p[rand_idx[i]] for i, p in enumerate(prompt)]
            # prompt = [""] * len(inputs)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            with autocast():
                acc, loss = model(inputs, labels=labels, prompt=prompt)
            # pbar.set_postfix({"Epoch": epoch, "Acc@1": acc[0], "Loss": loss.item()})
            pbar.set_description("Epoch: {} Acc@1: {:.2f} Loss | ce: {:.4f}| total: {:.4f}".format(epoch, acc.item(), loss['ce'].item(), loss['total'].item()))
            # Scales loss. 通过最小化向下溢出的值，可以改善网络的收敛效果
            scaler.scale(loss['total']).backward()

            # # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1)
            # torch.clamp(model.expert_weight.data, 0, 1)
            
            # torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1)
            # for param in model.expert_weight.parameters():
            #     param.data.clamp_(0, 1)
            
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)
            scaler.update()

            # see what's wrong with gradient
            # msg_line = ['\n=============================================== \n']
            # sum_grad_norm = 0
            # idx=1
            # for name, param in model.feature_extractor.named_parameters():
            #     if 'norm' in name and 'unet' in name:
            #         if param.grad is not None:
            #             sum_grad_norm += torch.sum(param.grad**2)
            #             idx += 1
            # sum_grad_norm /= idx
            # msg_line.append(f'unet norm, grad_norm = {sum_grad_norm:.6f} \n')

            # for idx, param in enumerate(model.classifier.parameters()):
            #     sum_grad_norm += torch.sum(param.grad**2)
            # sum_grad_norm /=(idx + 1)
            # msg_line.append(f'namodel.classifier, grad_norm = {sum_grad_norm:.6f} \n')
            # msg_line.append('=============================================== \n')
            # for text_line in msg_line:
            #     print(text_line)
            # sum_grad_norm = 0
            # for idx, param in enumerate(model.classifier.parameters()):
            #     sum_grad_norm += torch.sum(param.grad**2)
            # sum_grad_norm /=(idx + 1)
            # msg_line.append(f'namodel.classifier, grad_norm = {sum_grad_norm:.6f} \n')
            # # sum_grad_norm = 0
            # # for idx, param in enumerate(model.attn_pool.parameters()):
            # #     sum_grad_norm += torch.sum(param.grad**2)
            # # sum_grad_norm /=(idx + 1)
            # # msg_line.append(f'namodel.attn_pool, grad_norm = {sum_grad_norm:.6f} \n')
            # msg_line.append('=============================================== \n')
            # for text_line in msg_line:
            #     print(text_line)

def test(model, test_loader, prompts, args):
    with torch.no_grad():
        model.eval()
        acc1_sum = 0
        pbar = tqdm(test_loader, ncols=100)
        for i, (inputs, labels) in enumerate(pbar):
            prompt=[prompts[idx] for idx in labels]
            # length = len(prompt[0])
            # rand_idx = np.random.randint(length, size=len(prompt))   
            # prompt = [p[rand_idx[i]] for i, p in enumerate(prompt)]
            # prompt=[p[0] for p in prompt]
            prompt = ["a photo of"] * len(inputs)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            with autocast():
                acc, loss = model(inputs, labels=labels, prompt=prompt)

            # _, _, acc = model(inputs, labels=labels, prompt=prompt)
            acc1_sum += acc.item()
        acc1_epoch = acc1_sum / (i+1)
    return acc1_epoch

if __name__ == '__main__':
    main()
    
