# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from typing import List, Tuple, Union, Optional, Dict, Any
from torch import Tensor
from mmseg.utils import ConfigType, SampleList

from mmseg.registry import MODELS

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import Upsample, resize



@MODELS.register_module()
class TextFPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        self.global_text_dim = kwargs.get('global_text_dim', 768)
        kwargs.pop('global_text_dim')

        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        self.conv_seg = nn.Conv2d(self.channels, self.global_text_dim, kernel_size=1)

    def predict(self, inputs: Tuple[Tensor], global_text_features, batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs, global_text_features)

        return self.predict_by_feat(seg_logits, batch_img_metas)


    def loss(self, inputs: Tuple[Tensor], global_text_features, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, global_text_features) 
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def forward(self, inputs, global_text_features):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)  # resize and sum together in FPN to output final feature map (b, 128, h, w)

        output = self.cls_seg(output, global_text_features)  # final logits (b, num_classes, h, w)
        eps = 1e-5  # for stable training
        return output
    
    def cls_seg(self, output, global_text_features):
        """Classify each pixel.
        :param output: (b, 128, h, w)
        :param global_text_features: (num_classes, 768)
        """

        if self.dropout is not None:
            output = self.dropout(output)
        # TODO: add global text features and multiply with output
        output = self.conv_seg(output)  # final logits (b, global_text_dim, h, w)
        output = F.normalize(output, p=2, dim=1)  # normalize to unit vector
        output = torch.permute(output, (0, 2, 3, 1))  # (b, h, w, global_text_dim)
        output = torch.matmul(output, global_text_features.T.unsqueeze_(0))  # (b, h, w, num_classes)
        output = torch.permute(output, (0, 3, 1, 2))  # (b, num_classes, h, w)
        return output