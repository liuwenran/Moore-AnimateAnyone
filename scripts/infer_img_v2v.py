import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
from train_stage_1 import Net
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
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

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    # inference_config_path = config.inference_config
    # infer_config = OmegaConf.load(inference_config_path)
    # denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    #     config.pretrained_base_model_path,
    #     config.motion_module_path,
    #     subfolder="unet",
    #     unet_additional_kwargs=infer_config.unet_additional_kwargs,
    # ).to(dtype=weight_dtype, device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

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

    trained_net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        None,
        None,
    )

    trained_net_state_dict = load_file(config.ckpt_tuned_path)
    trained_net.load_state_dict(trained_net_state_dict)

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=trained_net.reference_unet,
        denoising_unet=trained_net.denoising_unet,
        pose_guider=trained_net.pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    from controlnet_aux.hed import Network
    from controlnet_aux import HEDdetector
    hed_net = Network('/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96/network-bsds500.pth')
    hed_detector = HEDdetector(hed_net)

    save_dir = Path(config.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)

    val_control_image_lines = open(config.validate_control_image_file).readlines()

    width, height = args.W, args.H

    for ind in range(len(val_control_image_lines)):
        ref_image_pil = Image.open(config.first_ref_image).convert("RGB")
        ref_image_width, ref_image_height = ref_image_pil.size
        if ref_image_width > ref_image_height:
            width = args.H
            height = args.W

        control_image_line = val_control_image_lines[ind].strip()
        control_image_pil = Image.open(control_image_line).convert("RGB")
        control_image_pil = hed_detector(control_image_pil)
        control_image_pil = control_image_pil.resize((width, height))

        generator = torch.manual_seed(args.seed)
        image = pipe(
            ref_image_pil,
            control_image_pil,
            width,
            height,
            args.steps,
            3.5,
            generator=generator,
        ).images
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        res_image_pil = Image.fromarray((image * 255).astype(np.uint8))

        control_image_name = control_image_line.split('/')[-1]

        res_image_pil.save(os.path.join(config.save_path, control_image_name))


if __name__ == "__main__":
    main()
