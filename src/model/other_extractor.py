import torch
import torchvision
from torch import nn, einsum, Tensor
import numpy as np
from einops import rearrange, repeat
import os

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import argparse

# import clip

import sys
sys.path.append('/data/sydong/diffusion/diffusion_features')
from third_party.clip import clip
from third_party.mae.models_vit import vit_large_patch16 as mae_vit_large_patch16
from third_party.dino.vision_transformer import vit_base as dino_vitbase16
from third_party.Swin.config import get_config
from third_party.Swin.models import build_model
from third_party.dinov2.models.vision_transformer import vit_large as dinov2_vitl14
# from third_party.beitv3.modeling_utiles import  BEiT3Wrapper, _get_base_config, _get_large_config
from third_party.convxt_v2.convnextv2 import convnextv2_large

CHECKPOINT_ROOT_DIR = '/data/sydong/backbones/checkpoint'

# TODO: load pretrained weight
class CusttomConvNextV2(nn.Module):
    """ConvNextV2:
    :param args: args
    :param kwargs: kwargs
    """
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.model = convnextv2_large(drop_path_rate=0.,
            head_init_scale=0.001,)
        ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'convnextv2_large_22k_224_ema_finetune.pt')
        state_dict = torch.load(ckpt_path, map_location='cpu')  # check for right or not
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.hidden_dim = 1536
    
    def forward(self, x, prompt, batch_img_metas=None):
        return self.model.forward_features(x)

# class CustomBEITv3(BEiT3Wrapper):
#     """BEITv3:"""
#     def __init__(self, args, **kwargs) -> None:
#         cfg = _get_large_config()
#         super(CustomBEITv3, self).__init__(args=cfg)
#         self.model_type = args.model_type
#         self.hidden_dim = 1024
#         self.fc_norm = nn.LayerNorm(self.hidden_dim)
#         ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'beit3_large_patch16_224_in1k.pth')
#         state_dict = torch.load(ckpt_path, map_location='cpu')  # check for right or not
        

#     def forward(self, x, prompt, batch_img_metas=None):
#         x = self.beit3(textual_tokens=None, visual_tokens=x)["encoder_out"]
#         t = x[:, 1:, :]
#         return self.fc_norm(t.mean(1))  # cls tokens or global pooling

class CustomDINOv2(nn.Module):
    """DINOV2:"""
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.model = dinov2_vitl14(patch_size=14, img_size=518, init_values=1.0)
        ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'dinov2_vitl14_pretrain.pth')
        state_dict = torch.load(ckpt_path, map_location='cpu')  # check for right or not
        self.model.load_state_dict(state_dict, strict=False) 
        self.hidden_dim = 1024
    
    def forward(self, x, prompt, batch_img_metas=None):
        ret = self.model.forward_features(x)
        return ret['x_norm_clstoken'] # cls tokens or global pooling

class CustomDINO(nn.Module):
    """
    DINO:
    :param args: args
    :param kwargs: kwargs
    return:
        out_list: list of features
    """
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.model = dino_vitbase16()
        ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'dino_vitbase16_pretrain.pth')
        state_dict = torch.load(ckpt_path, map_location='cpu')  # check for right or not
        self.model.load_state_dict(state_dict, strict=False) 
        self.hidden_dim = 768
        self.dtype = torch.float32
    
    def forward(self, x, prompt, batch_img_metas=None):
        return self.model.forward(x)

class CustomSwin(nn.Module):
    """
    DINO:
    :param args: args
    :param kwargs: kwargs
    return:
        out_list: list of features
    """
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        # TODO: add swin transformer config
        if args.model_type == 'swin':
            args.cfg = 'third_party/Swin/configs/swin/swin_large_patch4_window7_224_22k.yaml'
            ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'swin_large_patch4_window7_224_22k.pth')
        elif args.model_type == 'swinv2_256':
            # pre-train in IN-21K and finetune in IN-1K
            args.cfg = 'third_party/Swin/configs/swinv2/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.yaml'
            ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth')
        elif args.model_type == 'swinv2_192':
            # IN-21K model
            args.cfg = 'third_party/Swin/configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml'
            ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'swinv2_large_patch4_window12_192_22k.pth')
        else:
            raise NotImplementedError
        config = get_config(args)
        self.model = build_model(config)
        del self.model.head
        state_dict = torch.load(ckpt_path)['model']
        # state_dict.pop('head.weight')
        self.model.load_state_dict(state_dict, strict=False) # check for right or not
        self.hidden_dim = 1536

        self.dtype = torch.float32
        self.model = self.model

    def forward(self, x, prompt, batch_img_metas=None):
        latents = self.model.forward_features(x)  # [N, 1536]
        return latents


class CustomMAE(nn.Module):
    """
    DINO:
    :param args: args
    :param kwargs: kwargs
    return:
        out_list: list of features
    """
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.model = mae_vit_large_patch16(global_pool=True)
        # del self.model.decoder_blocks, self.model.decoder_norm, self.model.decoder_pred, self.model.decoder_pos_embed
        if args.model_type == 'mae_fine_tune':
            ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'mae_finetuned_vit_large.pth')
            state_dict = torch.load(ckpt_path, map_location='cpu')['model']
        else:
            ckpt_path = os.path.join(CHECKPOINT_ROOT_DIR, 'mae_pretrain_vit_large.pth')
            state_dict = torch.load(ckpt_path, map_location='cpu')['model']  # check for right or not
        self.model.load_state_dict(state_dict, strict=False)
        self.hidden_dim = 1024

        self.model = self.model
        self.dtype = torch.float32

    
    def forward(self, x, prompt, batch_img_metas=None):
        """
        return:
            x, mask, ids_restore 
        """
        latents = self.model.forward_features(x)
        return latents  # cls tokens or global pooling

class CustomCLIP(nn.Module):
    """
    DINO:
    :param args: args
    :param kwargs: kwargs
    return:
        out_list: list of features
    """
    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.model, _ = clip.load('ViT-L/14', download_root=CHECKPOINT_ROOT_DIR, device='cpu')
        # state_dict = torch.load('checkpoint/clip_vit_large.pth', map_location='cpu')  # check for right or not
        self.model = self.model.visual
        self.hidden_dim = 768

        self.model = self.model
        self.dtype = torch.float32

    def forward(self, x, prompt, batch_img_metas=None):
        return self.model(x)
    

def load_model(args, **kwargs):
    """ 
    default large model: dinov2, swinv2, mae clip
    default base model: dino
    """
    assert args.model_type in ['dino', 'clip', 'swin', 'mae', 'swinv2_192', 'convnext_v2', 'swinv2_256', 'mae_fine_tune', 'dinov2', 'beitv3']
    if args.model_type == 'dino':
        model = CustomDINO(args, **kwargs)  # only backbone
    elif args.model_type == 'clip':
        model = CustomCLIP(args, **kwargs)  # only visiual backbone
    elif args.model_type in ['swin', 'swinv2_192', 'swinv2_256']:
        model = CustomSwin(args, **kwargs)
    elif args.model_type in ['mae', 'mae_fine_tune']:  # total model: enoder + decoder
        model = CustomMAE(args, **kwargs)
    elif args.model_type == 'convnext_v2':
        model = CusttomConvNextV2(args, **kwargs)
    elif args.model_type == 'dinov2':
        model = CustomDINOv2(args, **kwargs)
    # elif args.model_type == 'beitv3':
    #     model = CustomBEITv3(args, **kwargs)
    else:
        raise NotImplementedError

    return model


class BaseExtractor(nn.Module):
    def __init__(self, args, use_checkpoint=True, **kwargs) -> None:
        super().__init__()
        self._load_pretrained_model(args, **kwargs)
        self.hidden_dim = self.backbone.hidden_dim
        self.use_checkpoint = use_checkpoint
        # freeze backbone
        for name, param in self.backbone.named_parameters():
            # param = param.float()
            param.requires_grad_(False)

    def _load_pretrained_model(self, args, **kwargs):
        self.backbone = load_model(args, **kwargs)
    

    def single_forward(
        self,
        inputs: torch.Tensor,
        prompt=None,
        batch_img_metas=None
    ):
        # TODO:
        # return multi-resolution features
        out_list = self.backbone(inputs, prompt=prompt, batch_img_metas=batch_img_metas)
        if not isinstance(out_list, list):
            out_list = [out_list]
        return out_list # return a list of features
    

    def forward(
        self,
        inputs: torch.Tensor,
        prompt=None,
        batch_img_metas = None
    ):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self.single_forward, inputs, prompt, batch_img_metas)
        else:
            return self.single_forward(inputs, prompt, batch_img_metas)
        

# ======================================================================
# we use the following models in our experiments
# dino: input: 224*224, output: [n, 768]
# dinov2: input: 518*518, output: [n, 1024]
# mae: input: 224*224,, output: [n, 1024]  # IN-1K model
# mae_fine_tune: input: 224*224,, output: [n, 1024]  # IN-1K model 
# swinv2: input: 192*192, output: [n, 1536]  # IN-21K model
# swinv2: input: 256*256, output: [n, 1536]  # finetuned in IN-1K model
# convnext_v2: input: 224*224,, output: [n, 1536]
# clip: input: 224*224,, output: [n, 768]
# ======================================================================
# we do not use the following models in our experiments
# swin: input: 224*224,, output: [n, 1536]
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='convnext_v2', help='model type')
    args = parser.parse_args()
    model = BaseExtractor(args).to('cuda')
    inputs = torch.randn(2, 3, 224, 224).to('cuda')
    out = model(inputs, ['a', 'b'])
    print(out[0].shape)