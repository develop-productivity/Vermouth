import os
import math
import numpy as np
from sbir_model import DiffusionSBIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import datetime
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import wandb

from src.model.captioner import Captioner
import utils_sbir as utils_sbir

os.environ["WANDB_MODE"] = "offline"
"""
Diffusion feature extractor for SBIR.
baseline is SAKE but no teacher distiallion
"""

def compute_logits(visaual_features, text_features, norm=True, comman_modality=True):
    if norm is True:
        visaual_features =  F.normalize(visaual_features, dim=-1)
        # visaual_features =  F.normalize(text_features, dim=-1)
    # text_features:(n_class * 2, 1024)
    if not comman_modality:
        num_sample_per_modality = visaual_features.shape[0] // 2
        logits_sk = torch.mm(visaual_features[:num_sample_per_modality], text_features[0].T)
        logits_im = torch.mm(visaual_features[num_sample_per_modality:], text_features[1].T)
        logits =  torch.cat((logits_sk, logits_im), dim=0)
    else:
        logits = torch.mm(visaual_features, text_features.T)
    return logits

def get_prompt(class_name):
    template = 'a photo of {}'
    prompt = [template.format(category) for category in class_name]
    return prompt

def random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = True

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def main():
    from SBIR.args_sbir import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # run args
    args = utils_sbir.tools.get_args(args)
    random_seed(args.seed)
    args.file_name = f'align_bs_{args.batch_size}_lr_{args.lr}_' + \
                     f'temp-{args.temperature}_img-{args.img_size}_{args.model_type}_max-attn-{args.max_attn_size}_time_step-' 
    args.file_name += "-".join([str(i) for i in args.time_step])
    args.file_name += "-".join([str(i) for i in args.place_in_unet])
    args.file_name += f'_{args.inv_method}_GN-{args.fuse_arch}_clip-proj'
    if args.frozen_expert:
        args.file_name += '_frozen'

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # write the configution to logs file
    # load data
    # immean = [0.485, 0.456, 0.406] # RGB channel mean for imagenet
    # imstd = [0.229, 0.224, 0.225]
    # immean_sk = [0.48145466, 0.4578275, 0.40821073]  # align with clip
    # imstd_sk = [0.26862954, 0.26130258, 0.27577711]
    immean = [0.5, 0.5, 0.5]  # align with diffusion
    imstd = [0.5, 0.5, 0.5]   # align with diffusion
    transformations = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize([args.img_size, args.img_size]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(immean, imstd)])
    sketch_train_loader, photo_train_loader, sk_test_loader, im_test_loader, train_class_name, test_class_name = \
                    utils_sbir.dataset.load_dataset(args, transformations)
    
    # diffusion SBIR
    generator = torch.Generator(device=device).manual_seed(8888)
    if not args.fix_noise:
        args.file_name += '_wo_fix-generator'
        generator = None

    args.file_name += 'BLIP_train_test'
    model = DiffusionSBIR(torch.LongTensor(args.time_step), generator=generator, args=args).to(device)
    train_prompts = get_prompt(train_class_name)
    test_prompts = get_prompt(test_class_name)
    model.register_text_features(train_prompts)
    captioner = Captioner('src/config/caption_coco.yaml').to(device)
    for name, params in captioner.named_parameters():
        params.requires_grad = False

    # backbone SBIR
    # train_prompts = get_prompt(train_class_name)
    # test_prompts = get_prompt(test_class_name)
    # model = BackboneSBIR(args=args, num_classes=len(train_class_name)).to(device)
    # model.register_text_features(train_prompts)
    # train_prompts = None
    # test_prompts = None
    print(str(datetime.datetime.now()) + ' all model inited.')

    # for diffusion based model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler_epoch = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: math.pow(0.001, float(x) / args.num_epoches) )
    if args.frozen_backbobe:
        args.save_dir = os.path.join(args.log_dir, 'frozen', args.dataset, args.content, args.commit, args.file_name)
    else:
        
        args.save_dir = os.path.join(args.log_dir, 'un_frozen', args.dataset, args.content, args.commit, args.file_name)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    wandb_config = vars(args)
    # log this config
    # with open(os.path.join(args.save_dir, 'logs.txt'), 'a') as f:
    #     for key,value in wandb_config.items():
    #         f.writelines('{}:{}\n'.format(key, value))
        
    wandb.init(project='ZeroShot-SBIR', dir=args.save_dir, config=wandb_config, name=args.file_name)
    scaler = GradScaler()
    for epoch in range(1):
        train(sketch_train_loader, photo_train_loader, model, captioner, optimizer, epoch, scaler, args)
        lr_scheduler_epoch.step()
        if epoch in [0, 4]:
            with torch.no_grad():
                map_valid = utils_sbir.evaluate.test_map(im_test_loader, sk_test_loader, epoch, model, captioner, test_prompts, args)
            with open(os.path.join(args.save_dir, 'logs.txt'), 'a') as f:
                f.writelines('Epoch: {} Test mAP@all: {} \n'.format(epoch, map_valid))
            wandb.log({'map_valid_each_epoch': map_valid})
        # save_checkpoint({'fuse':model.fuse.state_dict(),
        #                  'attn_pool':model.attn_pool.state_dict()
        #                   'expert':model.expert.mappers.state_dict(),}, 
        #                  os.path.join(args.save_dir, args.file_name + '_epoch.pth'.format(epoch+1)))
    wandb.finish()
    
def train(train_loader, train_loader_ext, model, captioner, optimizer, epoch, scaler, args):
    # top5 = utils_sbir.evaluate.AverageMeter()
    # top1 = utils_sbir.evaluate.AverageMeter()
    pbar = tqdm(zip(train_loader, train_loader_ext), ncols=120, total= min(len(train_loader), len(train_loader_ext)))
    for ((input, target), (input_ext, target_ext)) in pbar:
        input_all = torch.cat([input, input_ext],dim=0)  # concat [sketch, photo] 然后输入student 网络
        input_all =input_all.to(args.device)
        target_all = torch.cat([target, target_ext], dim=0)
        target_all = target_all.to(args.device)
        prompt=captioner(input_all)

        optimizer.zero_grad()
        with autocast():
            logits, loss = model(input_all, prompt, target_all)
        # top5.update(acc5[0], input_all.size(0))
        # top1.update(acc1[0], input_all.size(0))
        scaler.scale(loss['total']).backward()
        # scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        acc1, acc5 = utils_sbir.evaluate.accuracy(logits, target_all, topk=(1, 5))
        pbar.set_description("Epoch:{} Acc:{:.2f}-{:.2f} Loss | ce {:.4f} | total {:.4f}".format(epoch, acc1.item(), acc5.item(), loss['ce'].item(),  loss['total'].item()))
        wandb.log({'loss': loss['total'].item(), 'top1': acc1.item(), 'top5': acc5.item()})

        # # print the grad norm
        # msg_line = ['\n=============================================== \n']
        # sum_grad_norm = 0
        # idx=0
        # for param in model.expert.mappers.parameters():
        #     if param.requires_grad:
        #         sum_grad_norm += torch.sum(param.grad**2)
        #         idx += 1
        # sum_grad_norm /= idx
        # msg_line.append(f'namodel, grad_norm = {sum_grad_norm:.6f} \n')
        # msg_line.append('=============================================== \n')
        # for text_line in msg_line:
        #     print(text_line)


if __name__ == "__main__":
    main()