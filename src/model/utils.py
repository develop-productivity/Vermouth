from typing import Union
import torch
import numpy as np
from tqdm import tqdm


def ddpm_inversion(model, scheduler, x0, t):
    pass


def encode_text(tokenizer, text_encoder, prompts):
    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length, 
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = text_encoder(text_input.input_ids.to(text_encoder.device))[0]
    return text_encoding

def next_step(scheduler, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    """
    we sample x_{next_timestep} from x_{timestep} and in ddim_inv the timestep is in reverse order. e.g. sample x_{980} from x_{970}
    :param model: model
    """
    timestep, next_timestep = min(timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction  # sample x_{next_timestep} from x_{timestep}
    return next_sample

def get_noise_pred(unet, latent, t, context, cfg_scale):
    latents_input = torch.cat([latent] * 2)
    noise_pred = unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + cfg_scale * (noise_prediction_text - noise_pred_uncond)
    # latents = next_step(model, noise_pred, t, latent)
    return noise_pred


def ddim_loop(unet, tokenizer, text_encoder, scheduler, w0, prompt, cfg_scale):
    # uncond_embeddings, cond_embeddings = self.context.chunk(2)
    # all_latent = [latent]
    text_embedding = encode_text(tokenizer, text_encoder, prompt)
    uncond_embedding = encode_text(tokenizer, text_encoder, "")
    context = torch.cat([uncond_embedding, text_embedding])
    latent = w0.clone().detach()
    for i in tqdm(range(scheduler.num_inference_steps)):
        t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]  # timesteps in reverse order e.g. [990, 980, ..., 0]
        noise_pred = get_noise_pred(unet, latent, t, context, cfg_scale)
        latent = next_step(scheduler, noise_pred, t, latent)
        # all_latent.append(latent)
    return latent

def ddim_inversion(unet, tokenizer, text_encoder,scheduler, w0, prompt, cfg_scale):
    wT = ddim_loop(unet, tokenizer, text_encoder, scheduler, w0, prompt, cfg_scale)
    return wT