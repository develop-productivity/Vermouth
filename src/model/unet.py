import torch
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import sys

sys.path.append('/data/sydong/diffusion/diffusion_features')
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config





logger = logging.getLogger(__name__) # pylint: disable=invalid-name



class UnetFeatureExtractor(UNet2DConditionModel):
    # TODO:

		
	def reset_dim_stride(self, place_in_unet):
		"""Besides return dim and stride

        Returns:
            feature_dims: list of feature dimensions
            feature_strides: list of feature strides
        """
		self.place_in_unet = place_in_unet
		unet_block_indices = ()
		if 'down' in place_in_unet:
			unet_block_indices += (0, 1, 2, 3)
		if 'mid' in place_in_unet:
			unet_block_indices += (4,)
		if 'up' in place_in_unet:
			unet_block_indices += (5, 6, 7, 8)
		self.unet_block_indices = unet_block_indices
		ori_unet_block = []
		for block in self.down_blocks:
			ori_unet_block.append(block)
		ori_unet_block.append(self.mid_block)
		for block in self.up_blocks:
			ori_unet_block.append(block)

		assert set(self.unet_block_indices).issubset(set(range(len(ori_unet_block))))
		ori_unet_block_res = [[32, 32], [16,16], [8,8], [8,8], [8,8], [16,16], [32,32], [64,64], [64,64]]
		ori_unet_block_dims = [320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320]
		self.use_unet_blocks = [ori_unet_block[i] for i in unet_block_indices]
		self.use_unet_block_res = [ori_unet_block_res[i] for i in unet_block_indices]

		if self.place_in_unet == ['down'] or self.place_in_unet == ['up'] or set(self.place_in_unet) == set(['down', 'mid']):
			self.use_unet_block_dims = [1280, 1280, 640, 320] 
		elif set(self.place_in_unet) == set(['mid', 'up']):
			self.use_unet_block_dims = [1280, 1280, 1280, 960]
		elif set(self.place_in_unet) == set(['down', 'mid', 'up']):
			self.use_unet_block_dims = [1280, 1920, 1600, 960]
		elif self.place_in_unet == ['mid']:
			self.use_unet_block_dims = [1280]


	def upsample(self, features, scale_factor=2):
		# TODO:
		# reshape or permute
		"""upsample must be the size of [N, C, H, W, ...]"""
		features = torch.nn.functional.interpolate(features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
		return features
	
	def downsample(self, features, scale_factor=0.5):
		features = torch.nn.functional.interpolate(features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
		return features

	def post_process(self, features_list):
		""" align the feature list size to the [8, 16, 32, 64]
		"""
		if self.place_in_unet == ['down']:
			# features_list dim: [320, 640, 1280, 1280],res: [[32, 32], [16,16], [8,8], [8,8]]
			features_list = features_list[::-1]
			features_list[1:] = [self.upsample(features) for features in features_list[1:]]
			# for idx in range(1, len(features_list)):
			# 	features_list[idx] = self.upsample(features_list[idx])

		elif self.place_in_unet == ['mid']:
			assert isinstance(features_list, list) and len(features_list) == 1
			# feature_list dim: [1280], res: [[8, 8]]
			
			# upsample single resolution to 2x, 4x, 8x respectively
			# for scale_factor in [2, 4, 8]:
			# 	features_list.append(self.upsample(features_list[0], scale_factor=scale_factor))

			return features_list
		
		elif self.place_in_unet == ['up']:
			# feature_list dim: [1280, 1280, 640, 320], res: [[16,16], [32,32], [64,64], [64,64]]
			features_list[:-1] = [self.downsample(features) for features in features_list[:-1]]
			# for idx in range(len(features_list)-1):
			# 	features_list[idx] = self.downsample(features_list[idx])
			

		elif set(self.place_in_unet) == set(['down', 'mid']):
			# features_list dim: [320, 640, 1280, 1280, 1280],res: [[32, 32], [16,16], [8,8], [8,8], [8,8]]
			features_list = features_list[::-1]
			features_list[2:] = [self.upsample(features) for features in features_list[2:]]
			# features_list[1] = (features_list[1] + features_list[0])/2
			del features_list[1]
			

		elif set(self.place_in_unet) == set(['mid', 'up']):
			# feature_list dim: [1280 1280, 1280, 640, 320], res: [[8, 8], [16,16], [32,32], [64,64], [64,64]]
			features_list[-2] = torch.cat([features_list[-2], features_list[-1]], dim=1)
			del features_list[-1]

		elif set(self.place_in_unet) == set(['down', 'mid', 'up']):
			# feature list dim: [320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320] 
			# feature list res: [[32, 32], [16,16], [8,8], [8,8], [8,8], [16,16], [32,32], [64,64], [64,64]]
			features_list[4] = (features_list[2] + features_list[3] + features_list[4])/3  # dim=1280, res=[8,8]
			features_list[5] = torch.cat([features_list[1], features_list[5]], dim=1)  # dim=1920, res=[16,16]
			features_list[6] = torch.cat([features_list[0], features_list[6]], dim=1)  # dim=1600, res=[32,32]
			features_list[7] = torch.cat([features_list[7], features_list[8]], dim=1)  # dim=960, res=[64,64]
			del features_list[:4]
			del features_list[-1]
			
		else:
			raise ValueError(f"place_in_unet{self.place_in_unet}")
		
		return features_list
	
	# rewrite their forward function to output mid-layer values as feature
	def forward(
		self,
		sample: torch.FloatTensor,
		timestep: Union[torch.Tensor, float, int],
		encoder_hidden_states: torch.Tensor,
		class_labels: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		mid_block_additional_residual: Optional[torch.Tensor] = None,
		return_dict: bool = True,
		batch_img_metas=None,
	) -> Union[UNet2DConditionOutput, Tuple]:
		r"""
		Args:
			sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
			timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
			encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
			batch_img_metas (`dict`, *optional*, defaults to `None`): batch image meta information
			return_dict (`bool`, *optional*, defaults to `True`):
				Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
				

		Returns:
			[`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
			[`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
			returning a tuple, the first element is the sample tensor.
		"""
		# By default samples have to be AT least a multiple of the overall upsampling factor.
		# The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
		# However, the upsampling interpolation output size can be forced to fit any upsampling size
		# on the fly if necessary.
		default_overall_up_factor = 2**self.num_upsamplers

		# upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
		forward_upsample_size = False
		upsample_size = None

		if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
			logger.info("Forward upsample size to force interpolation output size.")
			forward_upsample_size = True

		# prepare attention_mask
		if attention_mask is not None:
			attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
			attention_mask = attention_mask.unsqueeze(1)

		# 0. center input if necessary
		if self.config.center_input_sample:
			sample = 2 * sample - 1.0

		# 1. time
		timesteps = timestep
		if not torch.is_tensor(timesteps):
			# TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
			# This would be a good case for the `match` statement (Python 3.10+)
			is_mps = sample.device.type == "mps"
			if isinstance(timestep, float):
				dtype = torch.float32 if is_mps else torch.float64
			else:
				dtype = torch.int32 if is_mps else torch.int64
			timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
		elif len(timesteps.shape) == 0:
			timesteps = timesteps[None].to(sample.device)

		# broadcast to batch dimension in a way that's compatible with ONNX/Core ML
		timesteps = timesteps.expand(sample.shape[0])

		t_emb = self.time_proj(timesteps)
		# timesteps does not contain any weights and will always return f32 tensors
		# but time_embedding might actually be running in fp16. so we need to cast here.
		# there might be better ways to encapsulate this.
		t_emb = t_emb.to(dtype=self.dtype)
		emb = self.time_embedding(t_emb)

		if self.class_embedding is not None:
			if class_labels is None:
				raise ValueError("class_labels should be provided when num_class_embeds > 0")

			if self.config.class_embed_type == "timestep":
				class_labels = self.time_proj(class_labels)

			class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
			emb = emb + class_emb

		# 2. pre-process
		# [N, 4, H, W] -- [N, 320, H, W]
		sample = self.conv_in(sample)  # [N, 320, H, W]

		out_list =[]
		# 3. down
		down_block_res_samples = (sample,)
        # encoder_hidden_states: context
        # sample: ([N, 320, 64, 64])--([N, 640, 32, 32]) -- ([N, 1280, 16, 16]) -- ([N, 1280, 8, 8]) -- ([N, 1280, 8, 8])
		for downsample_block in self.down_blocks:
			if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
				sample, res_samples = downsample_block(
					hidden_states=sample,
					temb=emb,
					encoder_hidden_states=encoder_hidden_states,
					attention_mask=attention_mask,
				)
			else:
				sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
			down_block_res_samples += res_samples
			if downsample_block in self.use_unet_blocks:
				out_list.append(sample)

		# 4. mid
		# sample [N, 1280, 8, 8] -- [N, 1280, 8, 8]
		
		sample = self.mid_block(
			sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
		)
		if mid_block_additional_residual is not None:
			sample = sample + mid_block_additional_residual
		if self.mid_block in self.use_unet_blocks:
				out_list.append(sample)

		# out_list.append(sample)
        # 5. up
		# sample [N, 1280, 8, 8] -- [N, 1280, 16, 16] -- [N, 1280, 32, 32] -- [N, 640, 64, 64] -- [N, 320, 64, 64]
		for i, upsample_block in enumerate(self.up_blocks):
			is_final_block = i == len(self.up_blocks) - 1

			res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
			down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

			# if we have not reached the final block and need to forward the
			# upsample size, we do it here
			if not is_final_block and forward_upsample_size:
				upsample_size = down_block_res_samples[-1].shape[2:]

			if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
				sample = upsample_block(
					hidden_states=sample,
					temb=emb,
					res_hidden_states_tuple=res_samples,
					encoder_hidden_states=encoder_hidden_states,
					# cross_attention_kwargs=cross_attention_kwargs,
					upsample_size=upsample_size,
					attention_mask=attention_mask,
					# encoder_attention_mask=encoder_attention_mask,
				)
			else:
				sample = upsample_block(
					hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
				)
			if upsample_block in self.use_unet_blocks:
				out_list.append(sample)

		# concatenate the last and second last block
		out_list = self.post_process(out_list)

        # 6. post-process
		# sample [N, 320, 64, 64] -- [N, 4, 64, 64]
		if self.conv_norm_out:
			sample = self.conv_norm_out(sample)
			sample = self.conv_act(sample)
		sample = self.conv_out(sample)
	
		if not return_dict:
			return (sample, out_list,)
		
		return UNet2DConditionOutput(sample=sample, out_list=out_list)
	