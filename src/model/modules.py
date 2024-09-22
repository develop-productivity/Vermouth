
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import torchvision
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import warnings
from abc import ABC, abstractmethod
import os
from torchvision.transforms.functional import InterpolationMode, resize

from third_party.dino.vision_transformer import vit_base as dino_vitb16
from third_party.dinov2.models.vision_transformer import vit_large as dinov2_vitl14
from third_party.dinov2.models.vision_transformer import vit_base as dinov2_vitb14


class FuseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stage2=False):
        super(FuseBlock, self).__init__()        
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_dim),
            # nn.BatchNorm2d(in_dim // 4),
            nn.SiLU(),
        )
        self.stage2 = None
        if stage2:
            self.stage2 = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=1, padding=1 if kernel_size==3 else 0, bias=False),
                nn.GroupNorm(32, out_dim),
                # nn.BatchNorm2d(out_dim),
                nn.SiLU(),
            )
            self.act = nn.SiLU()
    
    def forward(self,x):
        out = self.stage1(x)
        if self.stage2:
            out += self.stage2(out)
            out = self.act(out)
        return out


class BNFuseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stage2=False):
        super(BNFuseBlock, self).__init__()
        if stage2:
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim // 4),
                nn.SiLU(),
            )

            self.stage2 = nn.Sequential(
                nn.Conv2d(in_dim // 4, out_dim, kernel_size=kernel_size, stride=1, padding=1 if kernel_size==3 else 0, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(),
            )
            self.act = nn.SiLU()
        else:
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(),
            )
            self.stage2 = None
            self.act = nn.SiLU()

    def forward(self,x):
        out = self.stage1(x)
        if self.stage2:
            out = self.stage2(out)
            out = self.act(out)
        return out
           

class SimpleFuse(nn.Module):
    def __init__(self, in_dims, arch='tiny', fuse_method='low2high'):
        super(SimpleFuse, self).__init__()
        is_stage2 = False
        out_dims = [1024, 768, 512, 256]
        if arch == 'tiny':
            out_dims = [1024, 768, 512, 256]
        elif arch == 'pico':
            out_dims = [256, 256, 256, 256]
        elif arch == 'small':
            out_dims = [1024, 768, 512, 256]
            is_stage2 = True
        else:
            raise NotImplementedError
        self.out_dims = out_dims
        if fuse_method == 'low2high':
            in_dims[1:] = [in_dim + in_dim_bias for in_dim, in_dim_bias in zip(in_dims[1:], out_dims[:-1])]
        elif fuse_method == 'high2low':
            in_dims[:-1] = [in_dim + in_dim_bias for in_dim, in_dim_bias in zip(in_dims[:-1], out_dims[1:])]
        else:
            in_dims = in_dims  # no fuse do nothing

        self.low_res_conv =FuseBlock(in_dims[0], out_dims[0], kernel_size=1, stage2=is_stage2)
        self.mid_res_conv = FuseBlock(in_dims[1], out_dims[1],  kernel_size=1,stage2=is_stage2)
        self.high_res_conv = FuseBlock(in_dims[2], out_dims[2], kernel_size=3, stage2=is_stage2)
        self.highest_res_conv = FuseBlock(in_dims[3], out_dims[3], kernel_size=3, stage2=is_stage2)
        self.out_res = 64
        self.init_weight()

    @abstractmethod
    def forward(self, out_list):
        raise NotImplementedError
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

class SegSimpleFuse(nn.Module):
    def __init__(self, in_dims, arch='tiny', fuse_method='low2high'):
        super(SegSimpleFuse, self).__init__()
        is_stage2 = False
        out_dims = [256, 256, 256, 256]
        # if arch == 'tiny':
        #     out_dims = [2048, 1024, 512, 256]
        # elif arch == 'pico':
        #     out_dims = [256, 256, 256, 256]
        # elif arch == 'small':
        #     out_dims = [1024, 768, 512, 256]
        #     is_stage2 = True
        # else:
        #     raise NotImplementedError
        self.out_dims = out_dims
        if fuse_method == 'low2high':
            in_dims[1:] = [in_dim + in_dim_bias for in_dim, in_dim_bias in zip(in_dims[1:], out_dims[:-1])]
        elif fuse_method == 'high2low':
            in_dims[:-1] = [in_dim + in_dim_bias for in_dim, in_dim_bias in zip(in_dims[:-1], out_dims[1:])]
        else:
            in_dims = in_dims  # no fuse do nothing
                            
        self.low_res_conv =nn.Sequential(
                nn.Conv2d(in_dims[0], out_dims[0], kernel_size=1, stride=1, bias=False),
                nn.GroupNorm(32, out_dims[0]),
                nn.SiLU(),
            )
        self.mid_res_conv = nn.Sequential(
                nn.Conv2d(in_dims[1], out_dims[1], kernel_size=1, stride=1, bias=False),
                nn.GroupNorm(32, out_dims[1]),
                nn.SiLU(),
            )
        self.high_res_conv = nn.Sequential(
                nn.Conv2d(in_dims[2], out_dims[2], kernel_size=1, stride=1, bias=False),
                nn.GroupNorm(32, out_dims[2]),
                nn.SiLU(),
            )
        self.highest_res_conv = nn.Sequential(
                nn.Conv2d(in_dims[3], out_dims[3], kernel_size=1, stride=1, bias=False),
                nn.GroupNorm(32, out_dims[3]),
                nn.SiLU(),
            )
        self.out_res = 64
        self.init_weight()

    @abstractmethod
    def forward(self, out_list):
        raise NotImplementedError
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)



class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, group, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x
    

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    

def register_resnet_output(model):
    """
    """
    self = model
    def forward(x: Tensor):
        """
        Apply the model to extract each layer features
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # res: [H/2, W/2]

        x1 = self.layer1(x)  # res: [H/4, W/4] c: 64
        x2 = self.layer2(x1)  # res: [H/8, W/8] c: 128
        x3 = self.layer3(x2)  # res: [H/16, W/16] c: 256
        x4 = self.layer4(x3)  # res: [H/32, W/32] c: 512
        return [x4, x3, x2]
    self.forward = forward

class Adapter(nn.Module):
    def __init__(self, inplanes, reduction):
        super().__init__()
        self.adapter =  nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, bias=False, stride=1),
        )
    
    def forward(self, x):
        return self.adapter(x)

class ResNetAdapterExtrator(nn.Module):
    """
    """
    def __init__(self, out_dims, weigh_path, frozen=True):
        super().__init__()
        self.extracter = torchvision.models.resnet18()
        self.extracter.load_state_dict(torch.load(weigh_path, map_location='cpu'))
        self.extracter.train()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1,1)
        register_resnet_output(self.extracter)
        # we frozen the resnet model
        if frozen:
            for params in self.extracter.parameters():
                params.requires_grad=False
        
        in_dims = [512, 256, 128]
        self.mappers = nn.ModuleList([
                    Adapter(in_dims[0], reduction=4),
                    Adapter(in_dims[1], reduction=4),
                    Adapter(in_dims[2], reduction=4),
                    ])
        # self.mappers.apply(self._init_weight)

    def forward(self, inputs: Tensor):
        # first,we un-normalize tne inputs because the resnet model and the diffusion model's normalization is un-aligned
        inputs = inputs * 0.5 + 0.5
        inputs = (inputs - self.mean.to(inputs.device)) / self.std.to(inputs.device)
        # then, we extract the features from the desired layers
        features_list = self.extracter(inputs)  # from low to high resolution
        mapped_features_list = []
        for i, mapper in enumerate(self.mappers):
            mapped_features_list.append(mapper(features_list[i]))

        return mapped_features_list
    
    def _init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.zero_()
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.zero_()
    


class DINOv2AdapterExtrator(nn.Module):
    def __init__(self, weigh_path, out_indices=[2, 5, 8, 11], scales=[2, 1, 0.5, 0.25]) -> None:
        """
        A simle implementation to get multi-scale features from DINO model.

        Specifically, we use the pretrained DINO model to extract the multi-resolution features. first, we extract the features from the desired layers, then we interpolate the features to the desired scales.
        
        :param out_indices: the indices of the output layers
        :param scales: the scales of the output layers
        return: a list of multi-resolution features
        """
        super().__init__()
        # DINO-v2
        self.extracter = dinov2_vitb14(patch_size=14, img_size=518, init_values=1.0)
        state_dict = torch.load(weigh_path, map_location='cpu')  # check for right or not
        self.extracter.load_state_dict(state_dict, strict=False) 
        # we frozen the dino model
        for params in self.extracter.parameters():
                params.requires_grad=False 
            
        self.hidden_dim = 768
        self.out_indices = out_indices
        self.scales = scales
        self.sizes=[32, 16, 8, 4]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1,1)
        self.dtype = torch.float32
        in_dims = [768, 768, 768, 768]
        self.mappers = nn.ModuleList([
                    Adapter(in_dims[0], reduction=4),
                    Adapter(in_dims[1], reduction=4),
                    Adapter(in_dims[2], reduction=4),
                    Adapter(in_dims[3], reduction=4),
                    ])

    def forward(self, inputs: Tensor):
        """return multi-reslution features"""
        # first,we un-normalize tne inputs because the resnet model and the diffusion model's normalization is un-aligned
        inputs = inputs * 0.5 + 0.5
        inputs = (inputs - self.mean.to(inputs.device)) / self.std.to(inputs.device)
        inputs = resize(inputs, (224, 224), interpolation=InterpolationMode.BILINEAR)
        # then, we extract the features from the desired layers

        # TODO: whether keep cls token to produce the multi-reslution features?
        # we only consider the desired layers and remove the cls token
        # DINO-v2: output tokens from the designed blocks, each token is a 768 dim vector
        out_list = self.extracter.get_intermediate_layers(inputs, n=self.out_indices, reshape=True, norm=False)  # [n, B, HW/(16x16), 768]

        # we scale the features to the desired scales and 
        # features_list = [F.interpolate(out, size=32*scale, mode='bilinear', align_corners=False) for out, scale in zip(out_list, self.scales)]  # from low to high resolution
        features_list = [F.interpolate(out, (size, size), mode='bilinear', align_corners=False) for out, size in zip(out_list, self.sizes)]  # from low to high resolution
        features_list = features_list[::-1]
        mapped_features_list = []
        for i, mapper in enumerate(self.mappers):
            mapped_features_list.append(mapper(features_list[i]))

        return mapped_features_list
    
class DINOv1AdapterExtrator(nn.Module):
    def __init__(self, weigh_path, out_indices=[2, 5, 8, 11], scales=[2, 1, 0.5, 0.25]) -> None:
        """
        A simle implementation to get multi-scale features from DINO model.

        Specifically, we use the pretrained DINO model to extract the multi-resolution features. first, we extract the features from the desired layers, then we interpolate the features to the desired scales.
        
        :param out_indices: the indices of the output layers
        :param scales: the scales of the output layers
        return: a list of multi-resolution features
        """
        super().__init__()
        # DINO-v1
        self.extracter = dino_vitb16()
        state_dict = torch.load(weigh_path, map_location='cpu')  # check for right or not
        self.extracter.load_state_dict(state_dict, strict=False)

        # we frozen the dino model
        for params in self.extracter.parameters():
                params.requires_grad=False 
        self.hidden_dim = 768
        self.out_indices = out_indices
        self.scales = scales
        self.sizes=[32, 16, 8, 4]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1,1)
        self.dtype = torch.float32
        in_dims = [768, 768, 768, 768]
        self.mappers = nn.ModuleList([
                    Adapter(in_dims[0], reduction=4),
                    Adapter(in_dims[1], reduction=4),
                    Adapter(in_dims[2], reduction=4),
                    Adapter(in_dims[3], reduction=4),
                    ])

    def forward(self, inputs: Tensor):
        """return multi-reslution features"""
        # first,we un-normalize tne inputs because the resnet model and the diffusion model's normalization is un-aligned
        inputs = inputs * 0.5 + 0.5
        inputs = (inputs - self.mean.to(inputs.device)) / self.std.to(inputs.device)
        inputs = resize(inputs, (224, 224), interpolation=InterpolationMode.BILINEAR)
        # then, we extract the features from the desired layers

        # DINO-v1: output tokens from the `n` last blocks, each token is a 768 dim vector
        # output tokens from the `n` last blocks, each token is a 768 dim vector
        # TODO: whether keep cls token to produce the multi-reslution features?
        # we only consider the desired layers and remove the cls token
        out_list = self.extracter.get_intermediate_layers(inputs, n=12)  # [n, B, HW/(16x16), 768]
        out_list = [out_list[i][:,1:,:] for i in self.out_indices]
        num_patch = out_list[0].shape[1]
        # we reshape the features to [B, 768, sqrt(HW/(16x16)), sqrt(HW/(16x16))]
        out_list = [out.permute(0, 2, 1).reshape(-1, out.shape[-1], int(num_patch ** 0.5), int(num_patch ** 0.5)) for out in out_list]
        features_list = [F.interpolate(out, (size, size), mode='bilinear', align_corners=False) for out, size in zip(out_list, self.sizes)]  # from low to high resolution
        features_list = features_list[::-1]
        mapped_features_list = []
        for i, mapper in enumerate(self.mappers):
            mapped_features_list.append(mapper(features_list[i]))

        return mapped_features_list



if __name__ == "__main__":
    pass


