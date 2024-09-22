import logging
import abc
import math
import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from inspect import isfunction
import torch.nn as nn
import clip
import yaml
from PIL import Image
import sys
# import ruamel_yaml as yaml
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode, resize

sys.path.append('/data/sydong/diffusion/diffusion_features')
from third_party.BLIP.models.blip import blip_decoder


class Captioner(nn.Module):
    def __init__(self, cfg_file) -> None:
        super().__init__()
        config = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
        self.model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'], med_config=config['med_config'])
        self.config = config
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def forward(self, inputs):
        # reshape the inputs to 224*224
        # inputs = resize(inputs, (224, 224), interpolation=InterpolationMode.BICUBIC)
        inputs = inputs * 0.5 + 0.5
        inputs = (inputs - self.mean.to(inputs.device)) / self.std.to(inputs.device)
        caption = self.model.generate(inputs, sample=False, num_beams=self.config['num_beams'], max_length=10, 
                                  min_length=self.config['min_length'])
        return caption
        

if __name__ == "__main__":
    # captioner = Captioner('src/config/caption_coco.yaml')
    captioner = Captioner('src/config/caption_coco.yaml')
    # img = Image.open('/data/sydong/diffusion/diffusion_features/datasets/simple/data/ade.jpg')
    # img = Image.open('/data/sydong/diffusion/diffusion_features/datasets/simple/data/five_women_and_umbrellas.jpg')
    # img = Image.open('/data/sydong/diffusion/diffusion_features/datasets/simple/data/ade.jpg')
    img = Image.open('datasets/simple/data/dog.jpg')
    # img = Image.open('/data/dataset_4_all_users/imagenet/train/n01534433/n01534433_47.JPEG')
    # img = Image.open('/data/sydong/datasets/food-101/images/chocolate_cake/27058.jpg')
    # img = Image.open('/data/sydong/datasets/food-101/images/hamburger/68979.jpg')
    # img = Image.open('/data/sydong/datasets/oxford_pets/images/havanese_5.jpg')
    # img = Image.open('/data/sydong/datasets/sun397/SUN397/c/classroom/sun_abbafjowbzcvbucc.jpg')
    # img = Image.open('/data/sydong/datasets/SBIR/Sketchy/256x256/sketch/tx_000000000000_ready/cat/n02121620_1039-2.png')
    # img = Image.open('/data/sydong/datasets/SBIR/Sketchy/256x256/sketch/tx_000000000000_ready/scissors/n03044934_7203-1.png')
    # img = Image.open('/data/sydong/datasets/ADEChallengeData2016/images/validation/ADE_val_00000035.jpg')
    # img = Image.open('/data/sydong/datasets/ADEChallengeData2016/images/validation/ADE_val_00000077.jpg')
    # img = Image.open('/data/sydong/datasets/coco_stuff10k/images/test2014/COCO_train2014_000000005755.jpg')
    
    # img_list = ['/data/sydong/datasets/oxford_pets/images/miniature_pinscher_196.jpg', '/data/sydong/datasets/oxford_pets/images/newfoundland_8.jpg', ''/data/sydong/datasets/oxford_pets/images/keeshond_196.jpg'', '/data/sydong/datasets/oxford_pets/images/english_cocker_spaniel_7.jpg']

    # img_list = ['/data/sydong/diffusion/diffusion_features/datasets/generate/flowers/globe-flower/globe_flowers.jpg', '/data/sydong/diffusion/diffusion_features/datasets/generate/flowers/buttercup/A_picture_of_a_buttercup_flowers._4.png', '/data/sydong/diffusion/diffusion_features/datasets/generate/flowers/blackberry_lily/blackberry-lily.jpg', '/data/sydong/datasets/flowers-102/jpg/image_00335.jpg']

    img_list = ['/data/sydong/datasets/flowers-102/jpg/image_06700.jpg']

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
            transforms.Resize((256, 256),interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            normalize,
            ])
    for img_name in img_list:
        img = Image.open(img_name)
        imgs = transform_test(img).unsqueeze(0)
        caption = captioner(imgs)
        print(caption)
    # imgs = transform_test(img).unsqueeze(0)
    # # imgs = torch.rand(2, 3, 256, 256)
    # caption = captioner(imgs)
    # print(caption)