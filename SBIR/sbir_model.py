import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import sys 
sys.path.append('/data/sydong/diffusion/diffusion_features')
from src.model.feature_extractors import LDMFeatureExtractor
from src.model.modules import SimpleFuse, AttentionPool2d
from src.model.other_extractor import BaseExtractor
import clip
from src.model.modules import ResNetAdapterExtrator, DINOv1AdapterExtrator, DINOv2AdapterExtrator


class SemanticMap(nn.Module):
    """把提取的特征转变为CILP空间的特征"""
    def __init__(self, in_feature=2048, out_feature=768):
        super(SemanticMap, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        reduction=4
        self.map =  nn.Sequential(
            nn.Conv2d(self.in_feature, self.in_feature // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.in_feature // reduction, self.in_feature, kernel_size=1, stride=1, padding=0, bias=False),
        )
    def forward(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        out = self.map(inputs)
        return out

class ADDModel(nn.Module):
    def __init__(self, in_dims, out_dim) -> torch.Any:
        super().__init__()
        self.low_res_conv = nn.Conv2d(in_dims[0], out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.mid_res_conv = nn.Conv2d(in_dims[1], out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.high_res_conv = nn.Conv2d(in_dims[2], out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.highest_res_conv = nn.Conv2d(in_dims[3], out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_res = 4
        self.out_dim=out_dim
        # self.attn_pool = AttentionPool2d(self.out_res, out_dim, num_heads=8, output_dim=self.clip_feature_dim)
        

    def forward(self, out_list):
        low_res, mid_res, high_res, highest_res = out_list
        highest_res = self.highest_res_conv(highest_res)
        high_res = self.high_res_conv(high_res)
        mid_res = self.mid_res_conv(mid_res)
        low_res = self.low_res_conv(low_res)
        highest_res = TF.resize(highest_res, (self.out_res, self.out_res))
        high_res = TF.resize(high_res, (self.out_res, self.out_res))
        mid_res = TF.resize(mid_res, (self.out_res, self.out_res))
        # low_res = TF.resize(low_res, (self.out_res, self.out_res))
        fuse_feature = (highest_res + high_res + mid_res + low_res) / 4
        
        return fuse_feature


class FuseModel(SimpleFuse):
    def __init__(self, in_dims, out_res=None, arch='tiny', do_fuse=True):
        """
        :param in_dims: list of int, input dimensions   low to high
        :param out_dims: list of int, output dimensions  low to high
        :param out_res: int, output resolution
        """
        if do_fuse:
            fuse_method = 'high2low'
        else:
            fuse_method = 'no_fuse'
        super(FuseModel, self).__init__(in_dims, arch=arch, fuse_method=fuse_method)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.do_fuse = do_fuse
        # self.drop = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_res, out_res))
        self.out_res=out_res

    def forward(self, out_list):
        low_res, mid_res, high_res, highest_res = out_list
        if self.do_fuse:
            highest_res = self.highest_res_conv(highest_res)
            highest_res = self.avg_pool(highest_res)
            # highest_res = self.drop(highest_res)

            high_res = torch.cat([high_res, highest_res], dim=1)
            high_res = self.high_res_conv(high_res)
            high_res = self.avg_pool(high_res)
            # highest_res = self.drop(highest_res)

            mid_res = torch.cat([mid_res, high_res], dim=1)
            mid_res = self.mid_res_conv(mid_res)
            mid_res = self.avg_pool(mid_res)
            # highest_res = self.drop(highest_res)
            
            low_res = torch.cat([low_res, mid_res], dim=1)
            low_res = self.low_res_conv(low_res)
        else:
            low_res = self.low_res_conv(low_res)

        if low_res.shape[-1] != self.out_res:
            low_res = self.adaptive_pool(low_res)

        return low_res


class DiffusionSBIR(nn.Module):
    def __init__(self, time_steps, generator, args, num_classes=1000):
        super().__init__()
        # use_checkpoint :memory_efficient
        self.feature_extractor = LDMFeatureExtractor(args=args, use_checkpoint=True, attn_selector=args.attn_selector)
        self.criterion = nn.CrossEntropyLoss()
        self.logits_scale = 4.6052
        self.temperature = args.temperature
        if hasattr(self.feature_extractor.unet, 'use_unet_block_dims'):
            in_dims = self.feature_extractor.unet.use_unet_block_dims
        elif hasattr(self.feature_extractor.unet.unet, 'use_unet_block_dims'):
            in_dims = self.feature_extractor.unet.unet.use_unet_block_dims
        else:
            in_dims = [1280, 1280, 1280, 960]  # for ['mid', 'up']
        # TODO: modify in_dims adaptively
        # in_dims = self.feature_extractor.unet.use_unet_block_dims
        # in_dims = [1280, 1280, 640, 320]  # for ['up']  ['down'] and ['mid', 'down']
        # in_dims = [1280, 1280, 1280, 960]  # for ['mid', 'up']

        if args.expert == 'resnet':
            self.expert = ResNetAdapterExtrator(out_dims=None, weigh_path='/data/sydong/backbones/checkpoint/resnet18-5c106cde.pth', frozen=args.frozen_expert)
            expert_biases = [512, 256, 128]
            in_dims[1:] = [in_dim + expert_bias for in_dim, expert_bias in zip(in_dims[1:], expert_biases)]  # for resnet expert
        elif args.expert == 'dinov1':
            expert_biases = [768, 768, 768, 768]
            in_dims = [in_dim + expert_bias for in_dim, expert_bias in zip(in_dims, expert_biases)]  # for dino expert 
            self.expert = DINOv1AdapterExtrator(weigh_path='/data/sydong/backbones/checkpoint/dino_vitbase16_pretrain.pth')
        elif args.expert == 'dinov2':
            expert_biases = [768, 768, 768, 768]
            in_dims = [in_dim + expert_bias for in_dim, expert_bias in zip(in_dims, expert_biases)]  # for dino expert 
            self.expert = DINOv2AdapterExtrator(weigh_path='/data/sydong/backbones/checkpoint/dinov2_vitb14_pretrain.pth')
        else:
            self.expert=None
        self.time_steps, _ = torch.sort(time_steps, descending=True)
        self.generator = generator
        self.clip_feature=768
        self.device = args.device
        # frozen parameters
        if args.frozen_backbobe:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad_(False)

        if args.cross_attn:
            in_dims[1] += 77
            in_dims[2] += 77
            if args.max_attn_size == 32:
                in_dims[-1] += 77            
        if args.fuse_arch == 'add':
            self.fuse = ADDModel(in_dims=in_dims, out_dim=512)
            self.attn_pool = AttentionPool2d(4, 512, 8, self.clip_feature)
        else:
            self.fuse = FuseModel(in_dims=in_dims, arch=args.fuse_arch, out_res=4, do_fuse=args.do_fuse)
            self.attn_pool = AttentionPool2d(4, self.fuse.out_dims[0], 8, self.clip_feature)

    def get_global_text_embedding(self, prompt):
        text_input = self.feature_extractor.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.feature_extractor.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        result = self.feature_extractor.text_encoder(text_input.input_ids.to(self.device))
        pooler_output = result.text_embeds
        return F.normalize(pooler_output, dim=-1)  # 提前规范化，节省后面的计算
    
    def register_text_features(self, prompt):
        text_features = self.get_global_text_embedding(prompt)
        self.register_buffer('text_features',text_features)


    def extract_features(self, inputs, prompt):
        out_list = self.feature_extractor(inputs, prompt=prompt, time_steps=self.time_steps, generator=self.generator)
        # we use expert to extract features to fuse with the diff features
        if self.expert is not None:
            expert_out_list = self.expert(inputs)
            # resnet
            # out_list[1:] = [torch.cat([diff_feature, expert_feature], dim=1) for diff_feature, expert_feature in zip(out_list[1:], expert_out_list)]

            # dinov2
            out_list = [torch.cat([diff_feature, expert_feature], dim=1) for diff_feature, expert_feature in zip(out_list, expert_out_list)]
        fused_features = self.fuse(out_list)
        pooled_feature = self.attn_pool(fused_features)
        return pooled_feature
    
    def forward(self, inputs, prompt, labels):
        pooled_feature = self.extract_features(inputs, prompt)
        clip_space_feature = F.normalize(pooled_feature, dim=-1)        
        pre_logits = torch.mm(clip_space_feature, self.text_features.T) * self.logits_scale
        loss_ce = self.criterion(pre_logits / self.temperature, labels)
        # loss_l1 = self.smooth_l1_loss(clip_space_feature, clip_image_embedding)
        # loss_total = loss_ce + self.l1_loss_weight * loss_l1
        loss_total = loss_ce
        return pre_logits, {'ce': loss_ce, 'total': loss_total}


class BackboneSBIR(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.feature_extractor = BaseExtractor(args=args, use_checkpoint=False)
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.logits_scale = 4.6052
        self.temperature = args.temperature
        # frozen parameters
        for name, param in self.feature_extractor.named_parameters():
            param = param.float()
            param.requires_grad_(False)

        reduction = 4
        self.adapter = nn.Sequential(
            nn.Linear(self.feature_extractor.hidden_dim, self.feature_extractor.hidden_dim // reduction , bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(self.feature_extractor.hidden_dim // reduction, 768, bias=False),
        )

        self.head = nn.Linear(self.feature_extractor.hidden_dim, num_classes)

    def get_global_text_embedding(self, prompt):
        text_input = clip.tokenize(
            prompt
        )
        model, _ = clip.load("ViT-L/14", download_root='/data/sydong/backbones/checkpoint/CLIP', device=self.device)
        text_features = model.encode_text(text_input.to(self.device))
        return F.normalize(text_features, dim=-1)  # 提前规范化，节省后面的计算
    
    def register_text_features(self, prompt):
        text_features = self.get_global_text_embedding(prompt)
        self.register_buffer('text_features', text_features)
        self.text_features = self.text_features.detach()

    def extract_features(self, inputs, prompt):
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        features = self.feature_extractor(inputs, prompt=prompt)
        adapted_features = self.adapter(features[0])
        return adapted_features
    
    def forward(self, inputs, prompt, labels):
        features = self.extract_features(inputs, prompt)
        features = F.normalize(features, dim=-1)
        pre_logits = torch.mm(features, self.text_features.T) * self.logits_scale
        loss_ce = self.criterion(pre_logits / self.temperature, labels)
        loss_total = loss_ce
        return pre_logits, {'ce': loss_ce, 'total': loss_total}


        

