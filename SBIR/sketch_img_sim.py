import os
import math
import numpy as np
from sbir_model import DiffusionSBIR, BackboneSBIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import datetime
import pickle
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torchvision.transforms as transforms
from src.model.feature_extractors import LDMFeatureExtractor
from src.model.other_extractor import BaseExtractor
from src.model.captioner import Captioner
from sbir_model import DiffusionSBIR

import utils_sbir as utils_sbir

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


class SBIRFeature(nn.Module):
    def __init__(self, args):
        super(SBIRFeature, self).__init__()
        self.args = args
        self.backbone = LDMFeatureExtractor(args, use_checkpoint=False)
        # self.backbone = BaseExtractor(args=args, use_checkpoint=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, prompt, time_steps, generator=None):
        # x = self.backbone(x, prompt, time_steps, generator=generator)
        # x = self.avg_pool(x[0])
        # x = torch.flatten(x, 1)
        x = self.backbone(x, prompt)[0]
        return x 

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.fuse.load_state_dict(checkpoint['fuse'])
    model.attn_pool.load_state_dict(checkpoint['attn_pool'])
    return model


def get_features(model, data_loader, captioner, time_steps, generator=None):
    features_all = []
    targets_all = []
    for input, target in data_loader:
        input = input.to('cuda')
        target = target.to('cuda')
        # prompt = [prompts[idx] for idx in target]
        prompt = captioner(input)
        with torch.no_grad():
            # features = model(input, prompt, torch.LongTensor([time_steps]), generator=generator)
            features = model.extract_features(input, prompt)
        # 测试时的特征要规范化
        features = F.normalize(features, p=2, dim=1)
        features = features.cpu().detach().numpy()
        features_all.append(features.reshape(input.size()[0],-1))
        targets_all.append(target.cpu().detach().numpy())
    features_all = np.concatenate(features_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    print('features_all.shape', features_all.shape)
    return features_all, targets_all

def main():
    from SBIR.args_sbir import get_parser
    parser = get_parser()
    args = parser.parse_args()
    torch.cuda.set_device(2)
    # run args
    args = utils_sbir.tools.get_args(args)
    random_seed(args.seed)
    # args.file_name = f'align_bs_{args.batch_size}_lr_{args.lr}_' + \
    #                  f'temp-{args.temperature}_img-{args.img_size}_{args.model_type}_max-attn-{args.max_attn_size}_time_step-' 
    # args.file_name += "-".join([str(i) for i in args.time_step])
    # args.file_name += "-".join([str(i) for i in args.place_in_unet])
    # args.file_name += f'_{args.inv_method}_GN-{args.fuse_arch}_clip-proj'
    # if args.frozen_expert:
    #     args.file_name += '_frozen'

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # write the configution to logs file
    # load data
    # immean = [0.485, 0.456, 0.406] # RGB channel mean for imagenet
    # imstd = [0.229, 0.224, 0.225]
    # immean_sk = [0.48145466, 0.4578275, 0.40821073]  # align with clip
    # imstd_sk = [0.26862954, 0.26130258, 0.27577711]
    immean = [0.5, 0.5, 0.5]  # align with diffusion
    imstd = [0.5, 0.5, 0.5]   # align with diffusion
    # args.img_size=224
    transformations = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize([args.img_size, args.img_size]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(immean, imstd)])
    sketch_train_loader, photo_train_loader, sk_test_loader, im_test_loader, train_class_name, test_class_name = \
                    utils_sbir.dataset.load_dataset(args, transformations)
    test_prompts = get_prompt(test_class_name)
    # load model
    time_steps_list = [500, 200, 50]
    generator = torch.Generator(device=device).manual_seed(8888)

    # model = SBIRFeature(args).to(device)
    captioner = Captioner('src/config/caption_coco.yaml').to(device)
    print(str(datetime.datetime.now()) + ' all model inited.')
    for time_step in time_steps_list:
        model = DiffusionSBIR(torch.LongTensor([time_step]), generator, args).to(device)
        load_checkpoint(model, f'experiments/SBIR/sd1-5/frozen/sketchy_split2/main/BLIP_train_test/fuse/align_bs_32_lr_0.0001_temp-0.2_img-256_diffusion_max-attn-16_time_step-{time_step}mid-up_ddim_inv_GN-tiny_clip-proj_frozenBLIP_train_test/model.pth')
        file_name = f'SD_time_steps{time_step}_test_features.pkl'
        features_gallery, gt_labels_gallery = get_features(model, im_test_loader, captioner, time_step, generator=generator)
        features_query, gt_labels_query = get_features(model, sk_test_loader, captioner, time_step, generator=generator)
        scores = - cdist(features_query, features_gallery, metric='euclidean')  # similarity
        with open(os.path.join('experiments/SBIR/features/BLIP/trained', file_name),'wb') as fh:
            pickle.dump([features_gallery, features_query, gt_labels_gallery, gt_labels_query, scores],fh) 

    # time_step = 10
    # args.model_type='mae'
    # model = SBIRFeature(args).to(device)
    # captioner = Captioner('src/config/caption_coco.yaml').to(device)
    # file_name = f'mae_test_features.pkl'
    # # file_name = f'SD_time_steps{}_features.pkl'
    # features_gallery, gt_labels_gallery = get_features(model, im_test_loader, captioner, time_step, generator=generator)
    # features_query, gt_labels_query = get_features(model, sk_test_loader, captioner, time_step, generator=generator)
    # scores = - cdist(features_query, features_gallery, metric='euclidean')  # similarity
    # with open(os.path.join('experiments/SBIR/features', file_name),'wb') as fh:
    #     pickle.dump([features_gallery, features_query, gt_labels_gallery, gt_labels_query, scores],fh) 
    



        
    


if __name__ == "__main__":
    main()