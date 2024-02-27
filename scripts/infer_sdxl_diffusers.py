from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
import torch
import os

weight_dtype = torch.float16
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=weight_dtype,
)

from diffusers.models import UNet2DConditionModel

denoising_unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
    # addition_embed_type=None,
).to(dtype=weight_dtype, device="cuda")



pipeline = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path, unet=denoising_unet, vae=vae, torch_dtype=weight_dtype
)


pipeline = pipeline.to("cuda")

prompt = ""
# prompt = "perfect, extremely detailed, 8k, best quality"
# negative_prompt = 'disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w'
generator = torch.Generator(device=torch.device('cuda')).manual_seed(1)

folder_path = 'results/sd_xl_diffusers_no_add_cond_512x912'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for ind in range(10):
    image = pipeline(prompt, num_inference_steps=25, width=512, height=912, generator=generator).images[0]
    image.save(os.path.join(folder_path, str(ind) + ".png"))

