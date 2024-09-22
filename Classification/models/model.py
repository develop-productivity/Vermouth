import torch.nn as nn
import torch.nn.functional as F
import torch
from .classifier import Linear_prob, Adapter, FuseClassifier
from Classification.evaluate import accuracy, cls_acc

import sys 

from src.model.feature_extractors import LDMFeatureExtractor
from src.model.other_extractor import BaseExtractor
from src.model.modules import ResNetAdapterExtrator, AttentionPool2d


class FeatureExtractor(nn.Module):
    def __init__(self, args, time_steps, generator, num_classes=1000):
        super().__init__()
        self.feature_extractor = LDMFeatureExtractor(args=args, use_checkpoint=True)
        self.time_steps = time_steps
        self.generator = generator
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))

        for name, param in self.feature_extractor.named_parameters():
        # if 'norm' in name and 'unet' in name:
        #     param.requires_grad_(True)
        # else:
            param.requires_grad_(False)

    def forward(self, inputs, prompt=None):
        batch_size = len(inputs)
        if prompt is None:
            prompt = [""]*len(inputs)
        out_list = self.feature_extractor(inputs, prompt=prompt, time_steps=self.time_steps, generator=self.generator)
        x = self.adap_pool(out_list[0])
        x = x.view(batch_size, -1)
        return x
    

class DiffusionClassifier(nn.Module):
    def __init__(self, time_steps, generator, args, num_classes=1000):
        super().__init__()
        # use_checkpoint :memory_efficient
        self.feature_extractor = LDMFeatureExtractor(args=args, use_checkpoint=True, attn_selector=args.attn_selector)
        # self.expert = DINOAdapterExtrator()
        if args.expert == 'resnet':
            self.expert = ResNetAdapterExtrator(out_dims=[1357, 1357, 960], weigh_path='/data/sydong/backbones/checkpoint/resnet18-5c106cde.pth', frozen=args.frozen_expert)
        else:
            self.expert = None
        self.time_steps, _ = torch.sort(time_steps, descending=True)
        self.generator = generator
        self.clip_feature = 768
        self.classifier_type = args.classifier
        self.second_last_layer = args.second_last_layer
        # frozen parameters
        for name, param in self.feature_extractor.named_parameters():
                if 'norm' in name and 'unet' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        
        # TODO: modify in_dims adaptively
        if hasattr(self.feature_extractor.unet, 'use_unet_block_dims'):
            in_dims = self.feature_extractor.unet.use_unet_block_dims
        elif hasattr(self.feature_extractor.unet.unet, 'use_unet_block_dims'):
            in_dims = self.feature_extractor.unet.unet.use_unet_block_dims
        else:
            in_dims = [1280, 1280, 1280, 960]  # for ['mid', 'up']
        # in_dims = self.feature_extractor.unet.use_unet_block_dims
        # in_dims = [1280, 1280, 640, 320]  # for ['up']  ['down'] and ['mid', 'down']
        # in_dims = [1280, 1280, 1280, 960]  # for ['mid', 'up']
        # in_dims = [1280, 1280, 1280, 960]  # for ['mid', 'up', 'down]
        if args.cross_attn:
            in_dims[1] += 77
            in_dims[2] += 77
            if args.max_attn_size == 32:
                in_dims[-1] += 77
        if args.classifier == 'linear':
            self.classifier = Linear_prob(class_num=num_classes)
        elif args.classifier == 'adapter':
            self.classifier = Adapter(in_feature=in_dims[0])
            self.attn_pool = AttentionPool2d(4, in_dims[0], num_heads=8, output_dim=self.clip_feature)
        elif args.classifier in ['fuse', 'fuse_expert']:
            if self.expert:
                expert_biases = [512, 256, 128]
                in_dims[1:] = [in_dim + expert_bias for in_dim, expert_bias in zip(in_dims[1:], expert_biases)]  # for resnet expert 
            self.classifier = FuseClassifier(in_dims=in_dims, arch=args.fuse_arch, out_res=4, do_fuse=True)
            self.attn_pool = AttentionPool2d(4, self.classifier.out_dims[0], num_heads=8, output_dim=self.clip_feature)
        elif args.classifier == 'no_fuse':
            self.classifier = FuseClassifier(in_dims=in_dims, arch=args.fuse_arch, out_res=4, do_fuse=False)
            self.attn_pool = AttentionPool2d(4, self.classifier.out_dims[0], num_heads=8, output_dim=self.clip_feature)
        else:
            raise NotImplementedError
        
        self.criterion = nn.CrossEntropyLoss()
        # self.smooth_l1_loss = nn.SmoothL1Loss(beta=0.02)
        # self.l1_loss_weight = args.l1_loss_weight
        self.temperature = args.temperature



    def get_global_text_embedding(self, prompts):
        """获取文本的全局特征"""
        for prmpt_per_class in prompts:
            if len(prmpt_per_class) == 1:
                prompts = [p[0] for p in prompts]
                text_input = self.feature_extractor.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=self.feature_extractor.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                result = self.feature_extractor.text_encoder(text_input.input_ids.to(self.feature_extractor.vae.device))
                # text_embeddings = result.pooler_output
                if not self.second_last_layer:
                    text_embeddings = result.text_embeds
                else:
                    # we only use the second last layer's pooled out
                    last_hidden_state = result.last_hidden_state  # result.last_hidden_state: [seq_len, hidden_size]
                    input_ids = text_input.input_ids
                    text_embeddings = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                    ]
                break
            else:
                text_input = self.feature_extractor.tokenizer(
                    prmpt_per_class,
                    padding="max_length",
                    max_length=self.feature_extractor.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                result = self.feature_extractor.text_encoder(text_input.input_ids.to(self.feature_extractor.vae.device))
                # text_embeddings = result.pooler_output
                if not self.second_last_layer:
                    text_embedding = result.text_embeds
                else:
                    # we only use the second last layer's pooled out
                    last_hidden_state = result.last_hidden_state  # result.last_hidden_state: [seq_len, hidden_size]
                    input_ids = text_input.input_ids
                    text_embedding = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                    ]
                if prmpt_per_class == prompts[0]:
                    text_embeddings = text_embedding.mean(dim=0, keepdim=True)
                else:
                    text_embeddings = torch.cat((text_embeddings, text_embedding.mean(dim=0, keepdim=True)), dim=0)
            

        return F.normalize(text_embeddings, dim=-1)  # 提前规范化，节省后面的计算
    
    def registe_text_features(self, prompt):
        text_features = self.get_global_text_embedding(prompt)
        self.register_buffer('text_features',text_features)
        # self.text_features.requires_grad_(False)

    def extract_features(self, inputs, prompt=[""]):
        if prompt==None:
            prompt = [""]*len(inputs)

        # we enable the gradient of inputs to debug the checkpoint
        inputs.requires_grad_(True)
        out_list = self.feature_extractor(inputs, prompt=prompt, time_steps=self.time_steps, generator=self.generator)
        if self.expert:
            expert_out_list = self.expert(inputs)  # from low to high resolution        
            # concat with expert features
            out_list[1:] = [torch.cat([F.normalize(diff_feature, dim=1), F.normalize(expert_feature, dim=1)], dim=1) for diff_feature, expert_feature in zip(out_list[1:], expert_out_list)]
        # print(cross_attns.shape)
        if self.classifier_type in ['fuse', 'no_fuse', 'fuse_expert']:
            features = self.classifier(out_list)
        elif self.classifier_type == 'adapter':
            features = self.classifier(out_list[0])
        else:
            features = self.classifier(out_list[0])
        features = self.attn_pool(features)
        return features
    
    def get_pre(self, inputs, prompt=[""]):
        clip_space_feature = self.extract_features(inputs, prompt)
        clip_space_feature = F.normalize(clip_space_feature, dim=-1)
        pre = torch.mm(clip_space_feature, self.text_features.T)
        pre = pre.softmax(dim=-1)
        return pre

    def forward(self, inputs, labels, prompt=[""]):
        # clip_space_feature, clip_image_embedding = self.extract_features(inputs, prompt)
        clip_space_feature = self.extract_features(inputs, prompt)
        clip_space_feature = F.normalize(clip_space_feature, dim=-1)
        pre = torch.mm(clip_space_feature, self.text_features.T)
        labels = labels.to(pre.device)
        acc1 = cls_acc(pre, labels)
        loss_ce = self.criterion(pre / self.temperature, labels)
        loss_total = loss_ce
        return acc1, {'ce': loss_ce, 'total': loss_total}

class BackboneClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.feature_extractor = BaseExtractor(args=args, use_checkpoint=False)  # have already frozen
        self.device = args.device

        reduction = 4
        if args.classifier == 'adapter':
            self.adapter = nn.Sequential(
                nn.Linear(self.feature_extractor.hidden_dim, self.feature_extractor.hidden_dim // reduction , bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(self.feature_extractor.hidden_dim // reduction, self.feature_extractor.hidden_dim, bias=False),
            )
        else:
            self.adapter = nn.Identity()
        self.head = nn.Linear(self.feature_extractor.hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels, prompt=[""]):
        if inputs.device != self.device:
            inputs = inputs.to(self.device)

        latent = self.feature_extractor(inputs)
        latent = self.adapter(latent[0])

        pre = self.head(latent)
        labels = labels.to(pre.device)
        acc1 = cls_acc(pre, labels)  # tensor
        loss = self.criterion(pre, labels)
        return pre, loss, acc1
    