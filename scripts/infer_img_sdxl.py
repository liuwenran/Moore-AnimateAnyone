import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler, ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_sdxl_control2img_debug import SDXLControl2ImagePipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
from train_stage_1 import Net
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=912)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    # weight_dtype = torch.float16
    # pretrained_model_name_or_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     pretrained_model_name_or_path,
    #     vae=vae,
    #     image_encoder=image_enc,
    #     feature_extractor=CLIPImageProcessor(),
    #     torch_dtype=weight_dtype
    # )
    # pipeline = pipeline.to("cuda")
    # prompt = 'a boy'
    # import ipdb;ipdb.set_trace();
    # from PIL import Image
    # ip_adapter_image = Image.open('/mnt/petrelfs/liuwenran/forks/Moore-AnimateAnyone/data/validate_pair_images/nantong_ref_480.jpg')
    # image = pipeline(prompt, ip_adapter_image=ip_adapter_image, num_inference_steps=25).images[0]

    # controlnet = ControlNetModel.from_pretrained(
    #     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=weight_dtype
    # )

    # canny_weight_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0/snapshots/6c57eef07b4f634ede41bc560e5f3e2a321639ae/diffusion_pytorch_model.safetensors'
    # controlnet_openpose_state_dict = load_file(canny_weight_path)
    # # controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
    # state_dict_to_load = {}
    # for k in controlnet_openpose_state_dict.keys():
    #     if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
    #         new_k = k.replace("controlnet_cond_embedding.", "")
    #         state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
    # miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
    # import ipdb;ipdb.set_trace();
    

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)


    from diffusers.models import UNet2DConditionModel
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
        addition_embed_type=None
    ).to(dtype=weight_dtype, device="cuda")

    # denoising_unet = UNet2DConditionModel.from_pretrained(
    #     config.pretrained_base_model_path,
    #     subfolder="unet",
    #     addition_embed_type=None
    # ).to(dtype=weight_dtype, device="cuda")
    # import ipdb;ipdb.set_trace();

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")
    # import ipdb;ipdb.set_trace();

    # sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    # scheduler = DDIMScheduler(**sched_kwargs)
    from diffusers.schedulers import EulerDiscreteScheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="scheduler",
    )


    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        'openai/clip-vit-large-patch14',
    ).to(dtype=weight_dtype, device="cuda")

    image_enc_2 = CLIPVisionModelWithProjection.from_pretrained(
        'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
    ).to(dtype=weight_dtype, device="cuda")
    # image_enc = None
    # image_enc_2 = None

    # text_enc = CLIPTextModel.from_pretrained(
    #     'openai/clip-vit-large-patch14',
    # ).to(dtype=weight_dtype, device="cuda")

    # text_enc_2 = CLIPTextModelWithProjection.from_pretrained(
    #     'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
    # ).to(dtype=weight_dtype, device="cuda")

    generator = torch.manual_seed(args.seed)


    # load pretrained weights
    # denoising_unet.load_state_dict(
    #     torch.load(config.denoising_unet_path, map_location="cpu"),
    #     strict=False,
    # )
    # reference_unet.load_state_dict(
    #     torch.load(config.reference_unet_path, map_location="cpu"),
    # )
    # pose_guider.load_state_dict(
    #     torch.load(config.pose_guider_path, map_location="cpu"),
    # )

    # trained_net = Net(
    #     reference_unet,
    #     denoising_unet,
    #     pose_guider,
    #     None,
    #     None,
    # )

    # trained_net_state_dict = load_file(config.ckpt_tuned_path)
    # trained_net.load_state_dict(trained_net_state_dict)

    pipe = SDXLControl2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        image_encoder_2=image_enc_2,
        reference_unet=reference_unet,
        unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
        text_encoder=None,
        text_encoder_2=None,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    from controlnet_aux.hed import Network
    from controlnet_aux import HEDdetector
    hed_net = Network('/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96/network-bsds500.pth')
    hed_detector = HEDdetector(hed_net)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/debug_sdxl/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    val_image_lines = open(config.validate_image_file).readlines()
    val_control_image_lines = open(config.validate_control_image_file).readlines()

    width, height = args.W, args.H
    width = 1024
    height = 1024

    pil_images = []
    for ind in range(len(val_image_lines)):
        image = val_image_lines[ind].strip()
        ref_image_pil = Image.open(image).convert("RGB")
        ref_image_width, ref_image_height = ref_image_pil.size
        if ref_image_width > ref_image_height:
            width = args.H
            height = args.W

        control_image_line = val_control_image_lines[ind].strip()
        control_image_pil = Image.open(control_image_line).convert("RGB")
        control_image_pil = hed_detector(control_image_pil)
        control_image_pil = control_image_pil.resize((width, height))

        image = pipe(
            ref_image_pil,
            control_image_pil,
            width=width,
            height=height,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=generator,
        ).images
        # image = image[0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        print(f'image shape {image.shape}')
        res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
        # Save ref_image, src_image and the generated_image
        w, h = res_image_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        ref_image_pil = ref_image_pil.resize((w, h))
        control_image_pil = control_image_pil.resize((w, h))
        canvas.paste(ref_image_pil, (0, 0))
        canvas.paste(control_image_pil, (w, 0))
        canvas.paste(res_image_pil, (w * 2, 0))

        pil_images.append({"name": f"{ind}", "img": canvas})
    
    for sample_id, sample_dict in enumerate(pil_images):
        sample_name = sample_dict["name"]
        img = sample_dict["img"]
        out_file = f"{save_dir}/{sample_id}_{date_str}_{time_str}.png"
        img.save(out_file)



if __name__ == "__main__":
    main()
