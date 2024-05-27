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
from src.pipelines.pipeline_sdxl_control2img import SDXLControl2ImagePipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
from train_stage_1 import Net
from safetensors.torch import load_file
from src.dwpose import DWposeDetector
import cv2
import math

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


## 检测 reference 中的 pose
def detect_character(reference_path):
    dwprocessor = DWposeDetector()
    dwprocessor = dwprocessor.to('cuda')

    image = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)
    array = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    with torch.no_grad():
        detected_map, body_score = dwprocessor(array)
    detected_map = np.array(detected_map)    

    return detected_map


def pose_coord(image):

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化图像（将非黑色像素设为白色）
    _, threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # 计算非零像素的坐标
    non_zero_pixels = np.nonzero(threshold)
    # 获取上边界
    y_min = np.min(non_zero_pixels[0])
    y_max = np.max(non_zero_pixels[0])
    x_min = np.min(non_zero_pixels[1])
    x_max = np.max(non_zero_pixels[1])
    pose_coord = []
    pose_coord.append(x_min)
    pose_coord.append(y_min)
    pose_coord.append(x_max)
    pose_coord.append(y_max)

    pose_coord = [int(num) for num in pose_coord]

    return pose_coord


def align_pose(reference_path, driving_pose_path):
    # driving_pose_npy = cv2.imread(driving_pose_path)
    pose_image_pil = Image.open(driving_pose_path).convert("RGB")
    driving_pose_npy = np.array(pose_image_pil)
    pose_character_npy = detect_character(reference_path)
    pose_driving_coord = pose_coord(driving_pose_npy) # 驱动pose 的 范围
    pose_character_coord = pose_coord(pose_character_npy) # 人物pose 的范围

    reference_img = Image.open(reference_path)
    roi = driving_pose_npy[pose_driving_coord[1]:pose_driving_coord[3], pose_driving_coord[0]:pose_driving_coord[2]]  # driving_pose 裁剪区域
    deta_y_character = pose_character_coord[3] - pose_character_coord[1]
    deta_x_character = pose_character_coord[2] - pose_character_coord[0]
    deta_y_driving = pose_driving_coord[3] - pose_driving_coord[1]

    radio = deta_y_character / deta_y_driving  # 计算 refer人物的pose 与 driving pose 的比例

    roi_scaled = cv2.resize(roi, (math.ceil(roi.shape[1]*radio), deta_y_character)) # driving pose进行缩放 h*w

    driving_pose_target = np.zeros_like(pose_character_npy)

    mid = int((pose_character_coord[2] + pose_character_coord[0]) / 2)

    x_target = int (mid - roi_scaled.shape[1]/2)

    if x_target<0 and x_target + roi_scaled.shape[1] > driving_pose_target.shape[1]: # 左右都超
        driving_pose_target[pose_character_coord[1]:pose_character_coord[3], 0:driving_pose_target.shape[1]] = roi_scaled[0:roi_scaled.shape[0], -x_target: (- x_target + driving_pose_target.shape[1])]
    elif x_target<0:  # 左边超范围
        driving_pose_target[pose_character_coord[1]:pose_character_coord[3], 0: x_target + roi_scaled.shape[1]] = roi_scaled[0:roi_scaled.shape[0], -x_target:roi_scaled.shape[1]]
    elif x_target + roi_scaled.shape[1] > driving_pose_target.shape[1]: # 右边超范围
        driving_pose_target[pose_character_coord[1]:pose_character_coord[3], x_target:driving_pose_target.shape[1]] = roi_scaled[0:roi_scaled.shape[0], 0 : (- x_target + driving_pose_target.shape[1])]    
    else:
        driving_pose_target[pose_character_coord[1]:pose_character_coord[3], x_target:(x_target + roi_scaled.shape[1])] = roi_scaled

    driving_pose = Image.fromarray(driving_pose_target)

    return driving_pose

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


    # image_enc = CLIPVisionModelWithProjection.from_pretrained(
    #     'openai/clip-vit-large-patch14',
    # ).to(dtype=weight_dtype, device="cuda")

    # image_enc_2 = CLIPVisionModelWithProjection.from_pretrained(
    #     'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
    # ).to(dtype=weight_dtype, device="cuda")
    image_enc = None
    image_enc_2 = None

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

    pipe = SDXLControl2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        image_encoder_2=image_enc_2,
        reference_unet=reference_unet,
        unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
        # text_encoder=None,
        # text_encoder_2=None,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    if 'control_type' in config.keys() and config.control_type == 'hed':
        from controlnet_aux.hed import Network
        from controlnet_aux import HEDdetector
        hed_net = Network('/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96/network-bsds500.pth')
        hed_detector = HEDdetector(hed_net)
    

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/debug_sdxl/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    # val_image_lines = open(config.validate_image_file).readlines()
    # val_control_image_lines = open(config.validate_control_image_file).readlines()

    width, height = args.W, args.H
    # width = 1024
    # height = 1024

    data_root_path = '/mnt/petrelfs/liuwenran/repos/HumanAnimation'
    ref_image_paths = [
        f"{data_root_path}/inputs/reference/1.jpg",
        f"{data_root_path}/inputs/reference/2.jpg",
        f"{data_root_path}/inputs/reference/3.jpg",
        f"{data_root_path}/inputs/reference/4.jpg",
        f"{data_root_path}/inputs/reference/5.jpg",
        f"{data_root_path}/inputs/reference/6.jpg",
        f"{data_root_path}/inputs/reference/7.jpg",
    ]
    pose_image_paths = [
        f"{data_root_path}/inputs/pose/p1.jpg",
        f"{data_root_path}/inputs/pose/p2.jpg",
        f"{data_root_path}/inputs/pose/p3.jpg",
        f"{data_root_path}/inputs/pose/p4.jpg",
        f"{data_root_path}/inputs/pose/p5.jpg",
        f"{data_root_path}/inputs/pose/p6.jpg",
        f"{data_root_path}/inputs/pose/p7.jpg",
    ]

    image_embeds = torch.load('results/prompt_embeds/prompt_embeds.pt', map_location='cpu')
    image_embeds = image_embeds.to(device='cuda', dtype=torch.float16)

    pil_images = []
    for ref_image_path in ref_image_paths:
        for pose_image_path in pose_image_paths:
    # for index, (ref_image_path, pose_image_path) in enumerate(zip(ref_image_paths, pose_image_paths)):
            pose_name = pose_image_path.split("/")[-1].replace(".jpg", "")
            ref_name = ref_image_path.split("/")[-1].replace(".jpg", "")
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB") # 输入的pose
            pose_image_align = align_pose(ref_image_path, pose_image_path) # align后的pose

            # image0 = pipe(
            #     ref_image_pil,
            #     pose_image_pil,
            #     width=config.data.train_width,
            #     height=config.data.train_height,
            #     num_inference_steps=20,
            #     guidance_scale=3.5,
            #     generator=generator,
            # ).images

            image1 = pipe(
                ref_image_pil,
                pose_image_align,
                width=width,
                height=height,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=generator,
                image_embeds=image_embeds,
                fusion_type=config.fusion_type,
            ).images


            # image0 = image0[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            # print("image shape:", image0.shape)
            # res_image_pil0 = Image.fromarray((image0 * 255).astype(np.uint8))

            image1 = image1[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            print("image shape:", image1.shape)
            res_image_pil1 = Image.fromarray((image1 * 255).astype(np.uint8))

            # Save ref_image, src_image and the generated_image
            w, h = res_image_pil1.size
            # w, h = ref_image_pil.size
            canvas = Image.new("RGB", (w * 3, h), "white")
            ref_image_pil = ref_image_pil.resize((w, h))
            pose_image_pil = pose_image_pil.resize((w, h)) 
            pose_image_align = pose_image_align.resize((w,h))
            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(pose_image_pil, (w, 0))
            # canvas.paste(pose_image_align, (w * 2, 0))
            # canvas.paste(res_image_pil0, (w * 3, 0))
            canvas.paste(res_image_pil1, (w * 2, 0))

            pil_images.append({"name": f"{ref_name}_{pose_name}", "img": canvas})
            out_file = Path(f"{save_dir}/{ref_name}_{pose_name}.png")      
            canvas.save(out_file)



if __name__ == "__main__":
    main()
