import logging
import abc
import math
import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from inspect import isfunction
import torch.nn as nn
import clip

import torch
import torchvision
from torchvision.transforms import Compose, Resize, Normalize
from torch import nn, einsum, Tensor
import numpy as np
from einops import rearrange, repeat

from .unet import UnetFeatureExtractor
from .utils import ddim_inversion, ddpm_inversion

import sys
sys.path.append('/data/sydong/diffusion/diffusion_features')

from diffusers import (StableDiffusionPipeline, 
                       EulerDiscreteScheduler, 
                       DDPMScheduler,
                       DDIMInverseScheduler)
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from diffusers import StableDiffusionImg2ImgPipeline



logger = logging.getLogger(__name__) # pylint: disable=invalid-name


CLIP_PRPCESS_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_PRPCESS_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])



def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_scheduler_config(args):
    assert args.version in {'1-4', '1-5','2-1'}
    if args.version.startswith('1'):
        config = {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.7.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False
        }
    elif args.version == '2-1':
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,  # todo
            "trained_betas": None
        }
    else:
        raise NotImplementedError

    return config

# following diffusers' implementation
# (https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py)
# (or: anaconda3/envs/xxx/lib/python3.9/site-packages/diffusers/models/unet_2d_condition.py)

def get_sd_model(args, **kwargs):
    """
    Args:
    Returns:
        vae (AutoencoderKL): 
        tokenizer (CLIPTokenizer): 
        text_encoder (CLIPTextModel): 
        unet (UNet2DConditionModel): 
        scheduler (SchedulerMixin): PNDMScheduler  by default
    """
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError
    if args.version.startswith('1'):
        model_path = args.model_path
        ldm_stable = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
        # ldm_stable.enable_xformers_memory_efficient_attention()
        vae = ldm_stable.vae
        # modify in the future
        unet = UnetFeatureExtractor.from_pretrained(model_path,
                                                    subfolder="unet", torch_dtype=dtype)
        # custom function
        unet.reset_dim_stride(args.place_in_unet)  

        scheduler_config = get_scheduler_config(args)
        if args.inv_method == 'ddpm_schedule':
            scheduler = DDPMScheduler(num_train_timesteps=scheduler_config['num_train_timesteps'],
                                    beta_start=scheduler_config['beta_start'],
                                    beta_end=scheduler_config['beta_end'],
                                    beta_schedule=scheduler_config['beta_schedule'],
                                    )
        elif args.inv_method == 'ddim_inv':
            scheduler = DDIMInverseScheduler(num_train_timesteps=scheduler_config['num_train_timesteps'],
                                             )
        # print(scheduler)
        tokenizer = ldm_stable.tokenizer
        # text_encoder = CLIPTextModelWithProjection.from_pretrained('../clip_model/clip-vit-large-patch14')

    elif args.version == '2-1':
        model_path = args.model_path
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=dtype)
        pipe.enable_xformers_memory_efficient_attention()
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        # text_encoder = CLIPTextModelWithProjection.from_pretrained('../clip_model/clip-vit-large-patch14')
        # unet = pipe.unet
        unet = UnetFeatureExtractor.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype)

    else:
        raise NotImplementedError
    print(f"Pretrained model is successfully loaded from {model_path}")
    return vae, tokenizer, unet, scheduler
    # return vae, tokenizer, text_encoder, unet, scheduler

def register_attention_control(model, controller):
    """ register model.named_children's sub net forward function and count the num_att_layers"""
    """
    to_q, to_k, to_v method is model.named_children's sub net
    """
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            # context ???
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            # b: batch_size  h: heads, n: n_ctx(token_length), d: dim
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            # q = self.head_to_batch_dim(q)
            # k = self.head_to_batch_dim(k)
            # v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            # head = 8 for v1-4
            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            # __call__ bewteen steps
            # if is_cross:
            #     controller(attn2, is_cross, place_in_unet)
            attn2 = controller(attn2, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ in ['CrossAttention','MemoryEfficientCrossAttention', 'Attention']:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return  self.num_att_layers
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        self.cur_att_layer += 1
        attn = self.forward(attn, is_cross, place_in_unet)

        # We separate conditional and unconditional processes and, therefore, only consider the num_att_layers. but 
        # if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
        # attn = self.forward(attn, is_cross, place_in_unet)
        # return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0



class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        # we do not need control the attenion size
        # control the attention size
        # if attn.shape[1] <= self.max_attn_size[0] * self.max_attn_size[1]:  # avoid memory overhead
        #     self.step_store[key].append(attn)
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, base_size=64, max_attn_size=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.base_size = base_size
        # if max_attn_size is None:
        #     self.max_attn_size = [self.base_size, self.base_size]
        # else:
        #     self.max_attn_size = max_attn_size


class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=True, base_size=512, max_attn_size=None, attn_selector='up_self') -> None:
        super().__init__()
        self.unet = unet
        base_size = int(base_size)
        self.attention_store = AttentionStore(base_size=base_size // 8, max_attn_size=max_attn_size)
        self.size8 = base_size // 64
        self.size16 = base_size // 32
        self.size32 = base_size // 16
        self.size64 = base_size // 8
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        # register_hier_output(unet)
        self.max_attn_size = max_attn_size if max_attn_size is not None else self.size64
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        """return out_list features size: (base_size // 64, base_size // 32, base_size // 16, base_size // 8)
        """
        batch_img_metas = kwargs.get('batch_img_metas', None)
        # TODO: when multi-timesteps only save conditional cross attention
        # if_unconditonal = kwargs.get('if_unconditonal', False)
        # ori_shape = [img_metas.get('ori_shape', None) for img_metas in batch_img_metas]
        latents, out_list = self.unet(*args, **kwargs)  # multi-resolutions
        latents_size = args[0].shape[2:]
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn, latents_size=latents_size)
            out_list[1] = torch.cat([out_list[1], attn16], dim=1)  # base_size // 32
            out_list[2] = torch.cat([out_list[2], attn32], dim=1)  # # base_size // 16

            # commit the following programs and will not consider Attn64 adaptively
            if attn64.shape[2] <= self.max_attn_size:
                out_list[3] = torch.cat([out_list[3], attn64], dim=1)  # base_size // 8
        return latents, out_list
    
    def get_multi_features(self, *args, **kwargs):
        _, multi_scale_features_list = self.unet(*args, **kwargs)  # multi-resolutions
        avg_attn = self.attention_store.get_average_attention()
        mean_attns_dict = self.process_visiualize_attn(avg_attn, latents_size=None)
        return mean_attns_dict

    def process_visiualize_attn(self, avg_attn, latents_size):
        """mean attn
        TODO:
        处理任意大小的attn
        """
        mean_attns = {}
        attns = collections.OrderedDict()
        attns = {self.size8: [], self.size16: [], self.size32: [], self.size64: []}
        sizes_default = [keys for keys in attns.keys()]
        for k in self.attn_selector:
            for i, up_attn in enumerate(avg_attn[k]):
                size = int(math.sqrt(up_attn.shape[1]))
                if size not in sizes_default:
                    size = sizes_default[i]
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
        for key, value in attns.items():
            if value:
                mean_attns[key] = torch.stack(value).mean(0)

        return mean_attns
    
    def process_attn(self, avg_attn, latents_size):
        """mean attn
        TODO:
        处理任意大小的attn
        """
        latent_h, latent_w = latents_size
        attns = collections.OrderedDict({self.size16: [], self.size32: [], self.size64: []})
        sizes_default = [keys for keys in attns.keys()]
        for k in self.attn_selector:
            for i, up_attn in enumerate(avg_attn[k]):
                # we comment the following programs because when meet not the square shape, we will get wrong rearrange
                size = int(math.sqrt(up_attn.shape[1]))
                if size not in sizes_default:
                    size = sizes_default[i//3]
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
                # size = sizes_default[i // 3]
                # reduction = 2 ** (2 - (i // 3))
                # attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=math.ceil(latent_h / reduction) if latent_h is not None else size))
        # attn8 = torch.stack(attns[self.size8]).mean(0)
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        attn64 = torch.stack(attns[self.size64]).mean(0)
        return attn16, attn32, attn64
	

class LDMFeatureExtractor(nn.Module):
    ''' 
    Wrapper to extract features from pretrained Stable diffusion.
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, args, use_checkpoint=True, **kwargs):
        super().__init__()
        attn_selector = kwargs.pop('attn_selector', 'up_cross')
        if not hasattr(args, 'inv_method'):
            args.inv_method='ddpm_schedule'
        self.inv_method=args.inv_method
        self._load_pretrained_model(args, **kwargs)
        self.dtype = self.unet.dtype
        # self.device = self.unet.device
        if args.cross_attn:
            self.unet = UNetWrapper(self.unet, base_size=args.img_size, max_attn_size=args.max_attn_size, attn_selector=attn_selector)
        self.use_checkpoint = use_checkpoint


    def _load_pretrained_model(self, args, **kawrgs):
        vae, tokenizer, unet, scheduler = get_sd_model(args, **kawrgs)
        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.scheduler.set_timesteps(1000)
        self.unet.eval()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained('../clip_model/clip-vit-large-patch14')
        # self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('../clip_model/clip-vit-large-patch14')
        self.clip_process = Compose([Resize((224, 224)), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    # @torch.no_grad()
    def ecode_image(self, inputs):
        # input: [B, C, H, W]
        # batch operation
        inputs = inputs.to(self.dtype)
        x0 = self.vae.encode(inputs).latent_dist.mean
        x0 = x0 * 0.18215
        return x0
    
    # @torch.no_grad()
    def extract_step_features(self, latents, context, t, guidance_scale, batch_img_metas=None):
        """
        extract one step features or other activations
        TODO:
        
        """
        # Start by scaling the input with the initial noise distribution, sigma, 
        # the noise scale value, which is required for improved schedulers like UniPCMultistepScheduler:
        # latents = latents * self.scheduler.init_noise_sigma
        latents = self.scheduler.scale_model_input(latents, timestep=t)
        noise_prediction_text, out_list = self.unet(latents, t, encoder_hidden_states=context[1],return_dict=False, batch_img_metas=batch_img_metas)  # unetwrapper output multi-resolution
        noise_pred_uncond, _ = self.unet(latents, t, encoder_hidden_states=context[0], return_dict=False, batch_img_metas=batch_img_metas)  # unetwrapper output multi-resolution
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        # noise_pred = out_list[0]
        return noise_pred, out_list # return last time steps

    @torch.no_grad()
    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        # image = (image * 255).round().astype("uint8")
        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()
        # image = (image * 255).astype(np.uint8)
        return image
    
    def vision_context(self, inputs):
        # input: [B, C, H, W]
        # batch operation
        # first we resize input to [224, 224] and un-normalize tne inputs to align with CLIP pre-train setting
        inputs = inputs * 0.5 + 0.5
        inputs = self.clip_process(inputs)
        outputs = self.vision_encoder(inputs)
        # return outputs.pooled_output
        return outputs.image_embeds
        # return outputs.last_hidden_state

    @torch.no_grad()
    def single_forward_4_latents(
        self,
        inputs: torch.Tensor,
        prompt: List[str],
        time_steps = torch.LongTensor([50]),
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        batch_img_metas=None
    ):
        """
        return:
            activations (List[torch.Tensor])
        generator: Make sure we use the same noise for each sample and each time steps(proposed on: ODISE and VPD) 
        """
        batch_size = len(inputs)
        # Compute x_t and run denoise
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # result = self.text_encoder(text_input.input_ids.to(self.vae.device))
        # text_embeddings = result.last_hidden_state
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.vae.device)).last_hidden_state
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.vae.device)).last_hidden_state
        context = [uncond_embeddings, text_embeddings]
        # vae encode
        latents = self.ecode_image(inputs.to(self.vae.device))
        # set schedule timesteps
        # NOTE: DDIMInverseScheduler cann't set custom timesteps, e.g.,
        # self.scheduler.set_timesteps(timesteps=time_steps)
        # TODO: replace ddpm schedule noise with DDIM inversion
        if self.inv_method == 'ddim_inv':
            """
            DDIM Inversion: In other words, the diffusion process is performed in the reverse direction,that is z0 → zT instead of zT → z0, where z0 is set to be the encoding of the given real image
            In Null text inversion, guidance_scale is set to be 1 to get a good pivot.
            NOTE: you should set schedule to be DDIMInverseScheduler
            """
            time_steps = reversed(time_steps)
        elif self.inv_method == 'ddpm_schedule':
            # add noise
            noise = torch.randn(latents.shape, generator=generator, dtype=self.vae.dtype, device=self.vae.device)
            latents = self.scheduler.add_noise(latents, noise, time_steps[0])
        else:
            raise NotImplementedError
        # denoise
        # TODO: return multi-resolution features
        latents_list = []
        pred_noise_list = []
        for t in time_steps:
            noise_pred, out_list = self.extract_step_features(latents, context, t, guidance_scale, batch_img_metas=batch_img_metas)
            pred_noise_list.append(noise_pred[0])
            latents = self.scheduler.step(noise_pred, t, latents, generator, return_dict=False)[0]
            latents_list.append(latents)
        # When all time steps are completed, reset the attention_store if hvae attention_store attritube:
        if hasattr(self.unet, 'attention_store'):
            self.unet.attention_store.reset()
        return latents_list, pred_noise_list, out_list # return last time steps

    def forward_4_latents(self,
        inputs: torch.Tensor,
        prompt: List[str],
        time_steps = torch.LongTensor([50]),
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        batch_img_metas = None
    ):
        if self.use_checkpoint:
            # inputs.requires_grad_(True)
            # prompt.requires_grad_(True)
            # time_steps.requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(
                self.single_forward_4_latents, inputs, prompt, time_steps, guidance_scale, generator, batch_img_metas)
        else:
            return self.single_forward_4_latents(inputs, prompt, time_steps, guidance_scale, generator, batch_img_metas)
    

    # @torch.no_grad()
    def single_forward(
        self,
        inputs: torch.Tensor,
        prompt: List[str],
        time_steps = torch.LongTensor([50]),
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        batch_img_metas=None
    ):
        """
        return:
            activations (List[torch.Tensor])
        generator: Make sure we use the same noise for each sample and each time steps(proposed on: ODISE and VPD) 
        """
        batch_size = len(inputs)
        # Compute x_t and run denoise
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # result = self.text_encoder(text_input.input_ids.to(self.vae.device))
        # text_embeddings = result.last_hidden_state
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.vae.device)).last_hidden_state
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.vae.device)).last_hidden_state
        context = [uncond_embeddings, text_embeddings]
        # vae encode
        latents = self.ecode_image(inputs.to(self.vae.device))
        # set schedule timesteps
        # NOTE: DDIMInverseScheduler cann't set custom timesteps, e.g.,
        # self.scheduler.set_timesteps(timesteps=time_steps)
        # TODO: replace ddpm schedule noise with DDIM inversion

        if self.inv_method == 'ddim_inv':
            """
            DDIM Inversion: In other words, the diffusion process is performed in the reverse direction,that is z0 → zT instead of zT → z0, where z0 is set to be the encoding of the given real image
            In Null text inversion, guidance_scale is set to be 1 to get a good pivot.
            NOTE: you should set schedule to be DDIMInverseScheduler
            """
            time_steps = reversed(time_steps)
        elif self.inv_method == 'ddpm_schedule':
            # add noise
            noise = torch.randn(latents.shape, generator=generator, dtype=self.vae.dtype, device=self.vae.device)
            latents = self.scheduler.add_noise(latents, noise, time_steps[0])
        else:
            raise NotImplementedError

        # denoise
        # TODO: return multi-resolution features
        for t in time_steps:
            noise_pred, out_list = self.extract_step_features(latents, context, t, guidance_scale, batch_img_metas=batch_img_metas)
            latents = self.scheduler.step(noise_pred, t, latents, generator, return_dict=False)[0]
        # When all time steps are completed, reset the attention_store
        # if hvae attention_store attritube:
        if hasattr(self.unet, 'attention_store'):
            self.unet.attention_store.reset()
        # return out_list, pooled_output # return last time steps
        # return out_list, latents_list # return last time steps
        return out_list # return last time steps

    def forward(
        self,
        inputs: torch.Tensor,
        prompt: List[str],
        time_steps = torch.LongTensor([50]),
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        batch_img_metas = None
    ):
        if self.use_checkpoint:
            # inputs.requires_grad_(True)
            # prompt.requires_grad_(True)
            # time_steps.requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(
                self.single_forward, inputs, prompt, time_steps, guidance_scale, generator, batch_img_metas)
        else:
            return self.single_forward(inputs, prompt, time_steps, guidance_scale, generator, batch_img_metas)
        

    def extract_step_attn(self, latents, context, t, guidance_scale):
        latents = self.scheduler.scale_model_input(latents, timestep=t)
        out_list = self.unet(latents, t, encoder_hidden_states=context[1], return_dict=False)  # multi-resolutions
        avg_attn = self.unet.attention_store.get_average_attention()
        mean_attns_dict = self.unet.process_visiualize_attn(avg_attn, latents_size=None)
        return mean_attns_dict, out_list

        
    def extract_attn(self, 
                     inputs: torch.Tensor,
                     prompt: List[str],
                     time_steps = torch.LongTensor([50]),
                     guidance_scale: float = 7.5,
                     generator: Optional[torch.Generator] = None):
        
        batch_size = len(inputs)
        # Compute x_t and run denoise
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        result = self.text_encoder(text_input.input_ids.to(self.vae.device))
        # text_embeddings, pooled_output = result.last_hidden_state, result.pooler_output
        text_embeddings = result.last_hidden_state
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.vae.device))[0]
        context = [uncond_embeddings, text_embeddings]
        # vae encode
        latents = self.ecode_image(inputs.to(self.vae.device))
        # add noise
        noise = torch.randn(latents.shape, generator=generator, dtype=self.vae.dtype, device=self.vae.device)
        latents = self.scheduler.add_noise(latents, noise, time_steps[0])
        for t in time_steps:
            attn_dict, _ = self.extract_step_attn(latents, context, t, guidance_scale)
        return attn_dict





