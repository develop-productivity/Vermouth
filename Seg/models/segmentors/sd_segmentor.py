import logging
import json

from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from mmengine.config import Config

from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmseg.models.utils import resize
from mmseg.utils import (ForwardResults, ConfigType, OptConfigType, OptMultiConfig, 
                         OptSampleList, SampleList, add_prefix)


from mmseg.models.segmentors.base import BaseSegmentor

import sys
sys.path.append('/data/sydong/diffusion/diffusion_features')
from src.model.feature_extractors import LDMFeatureExtractor
from src.model.modules import SimpleFuse

def get_prompt(meta_file):
    """ from meta_file get prompts"""
    meta_info = json.load(open(meta_file, 'r'))
    templates, classes_dict = meta_info['templates'], meta_info['class_name']
    prompts = []
    for category in classes_dict:
        prompt_per_category = [template.format(category)  for template in templates]
        prompts.append(prompt_per_category)
    return prompts



class FuseModel(SimpleFuse):
    """modify channels"""
    def __init__(self, in_dims, out_dims):
        super(FuseModel, self).__init__(in_dims, out_dims)

    def forward(self, out_list):
        """modify channels to fit ResNet 50 output channels"""
        low_res, mid_res, high_res, highest_res = out_list
        low_res = self.low_res_conv(low_res)
        mid_res = self.mid_res_conv(mid_res)
        high_res = self.high_res_conv(high_res)
        highest_res = self.highest_res_conv(highest_res)
        return [highest_res, high_res, mid_res, low_res]


@MODELS.register_module()
class DiffusionSeg(BaseSegmentor):
    def __init__(self,
                 unet,
                 decode_head,
                 neck=None,
                 auxiliary_head: OptConfigType = None, 
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 generator_seed=8888,
                 time_steps: List = [50],
                 in_dims=[1280, 1357, 1357, 960],
                 out_dims=[2048, 1024, 512, 256]
                 ):
        super(DiffusionSeg, self).__init__(data_preprocessor=data_preprocessor,init_cfg=init_cfg)
        self.backbone = LDMFeatureExtractor(args=unet, use_checkpoint=False)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fuse = FuseModel(in_dims, out_dims)
        self.time_steps = torch.LongTensor(time_steps)
        self.generator = torch.Generator(device='cuda').manual_seed(generator_seed)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        prompts = get_prompt(self.train_cfg.meta_file)
        self.registe_text_features(prompts)
        assert self.with_decode_head


    def get_global_text_embedding(self, prompts):
        """获取文本的全局特征"""
        for prmpt_per_class in prompts:
            if len(prmpt_per_class) == 1:
                prompts = [p[0] for p in prompts]
                text_input = self.backbone.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=self.backbone.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                result = self.backbone.text_encoder(text_input.input_ids.to(self.backbone.vae.device))
                text_embeddings = result.pooler_output

                # we only use the second last layer's pooled out
                # text_embeddings = result.hidden_states[1][:, 0, :]  # result.hidden_states: [layer_nums, seq_len, hidden_size]
                break
            else:
                text_input = self.backbone.tokenizer(
                    prmpt_per_class,
                    padding="max_length",
                    max_length=self.backbone.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                result = self.backbone.text_encoder(text_input.input_ids.to(self.backbone.vae.device))
                text_embedding = result.pooler_output

                # we only use the second last layer's pooled out
                # text_embeddings = result.hidden_states[1][:, 0, :]  # result.hidden_states: [layer_nums, seq_len, hidden_size]
                
                if prmpt_per_class == prompts[0]:
                    text_embeddings = text_embedding.mean(dim=0, keepdim=True)
                else:
                    text_embeddings = torch.cat((text_embeddings, text_embedding.mean(dim=0, keepdim=True)), dim=0)
            

        return F.normalize(text_embeddings, dim=-1)  # 提前规范化，节省后面的计算
    
    def registe_text_features(self, prompt):
        text_features = self.get_global_text_embedding(prompt)
        self.register_buffer('text_features',text_features)
        self.text_features.requires_grad = False


    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs, prompts, batch_img_metas=None) -> List[Tensor]:
        """Extract features from images."""
        if prompts is None:
            prompts=[""]*len(inputs)
        # with autocast():
        x = self.backbone(inputs, prompts, time_steps=self.time_steps, generator=self.generator, batch_img_metas=batch_img_metas)
        x = self.fuse(x)
        if self.with_neck:
            x = self.neck(x)
        x = tuple([xx.float() for xx in x])
        return x
    
    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict], prompts=None) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs, prompts=prompts, batch_img_metas=batch_img_metas)
        seg_logits = self.decode_head.predict(x, self.text_features, batch_img_metas,
                                              self.test_cfg)

        return seg_logits
    
    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None, prompts=None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, prompts)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, prompts)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, prompts)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')
    
    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, self.text_features, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList, prompts) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        x = self.extract_feat(inputs, prompts=prompts, batch_img_metas=batch_img_metas)
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
    
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                prompt=None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas, prompt)

        return self.postprocess_result(seg_logits, data_samples)
    
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None, prompts=None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        x = self.extract_feat(inputs, prompts=prompts, batch_img_metas=batch_img_metas)
        return self.decode_head.forward(x, self.text_features)
    
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict],
                        prompt=None) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits
    
    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict],
                        prompts) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas, prompts)

        return seg_logits
    
    def inference(self, inputs: Tensor, batch_img_metas: List[dict], prompts=None) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas, prompts)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas, prompts)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward(x, img_metas, self.test_cfg)
        return seg_logits
    
    
    def forward_dummy(self, inputs, data_samples, prompts):
        """Dummy forward function."""
        seg_logit = self.encode_decode(inputs, None, prompts)

        return seg_logit
    

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


    
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    cfg = Config.fromfile('Seg/configs/config_mmcv.py')
    segmentation_head = DiffusionSeg(cfg['model'])
    print(segmentation_head)
    print(get_parameter_number(segmentation_head))