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
from src.utils.util import get_fps, read_frames, save_image_grid
from metrics.utils.loss_utils import ssim
from metrics.lpipsPyTorch import lpips
import json
from tqdm import tqdm
from metrics.utils.image_utils import psnr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=896)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--name", type=str, default='temp')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    W = args.W
    H = args.H
    seed = args.seed
    cfg = args.cfg
    steps = args.steps
    config_path = args.config
    # Load the configuration
    config = OmegaConf.load(config_path)

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

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device="cuda")


    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        device="cuda",
        dtype=weight_dtype,
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(
        device="cuda", dtype=weight_dtype
    )

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = W, H
    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device="cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{seed}--{args.name}-{width}x{height}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)
    image_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()])
    
    ssims = []
    psnrs = []
    lpipss = []
    for repeat_ind in range(args.repeat):
        for ref_image_root in config['test_cases']:
            pose_root = config["test_cases"][ref_image_root][0]

            all_vids = os.listdir(ref_image_root)
            all_vids.sort()
            all_ref_videos = [os.path.join(ref_image_root, f) for f in all_vids]

            for ind, ref_video_path in enumerate(all_ref_videos):
                print(f'ind {ind} in {len(all_ref_videos)}')

                if config.data_type == 'frames':
                    vid = ref_video_path.split('/')[-1]
                    pose_video_path = os.path.join(pose_root, vid)

                    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                    frame_files = os.listdir(ref_video_path)
                    frame_files = [os.path.join(ref_video_path, f) for f in frame_files]
                    frame_files.sort()
                    all_ref_images = []
                    for frame in frame_files:
                        frame_extension = os.path.splitext(frame)[1]
                        if frame_extension in IMAGE_EXTENSIONS:
                            all_ref_images.append(frame)
                    ref_random_idx = np.random.randint(0, len(all_ref_images))
                    ref_image_name = all_ref_images[ref_random_idx]
                    ref_image = Image.open(ref_image_name).convert('RGB')
                    ref_image_tensor= image_transform(ref_image)

                    frame_files = os.listdir(pose_video_path)
                    frame_files = [os.path.join(pose_video_path, f) for f in frame_files]
                    frame_files.sort()
                    all_pose_images = []
                    for frame in frame_files:
                        frame_extension = os.path.splitext(frame)[1]
                        if frame_extension in IMAGE_EXTENSIONS:
                            all_pose_images.append(frame)
                    tgt_random_idx = np.random.randint(0, len(all_pose_images))
                    tgt_random_idx = ref_random_idx
                    pose_image_path = all_pose_images[tgt_random_idx]
                    pose_image = Image.open(pose_image_path).convert('RGB')
                    pose_tensor_list = [image_transform(pose_image)]
                    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (1, c, h, w)

                else:
                    vid = ref_video_path.split('/')[-1].split('.')[0]
                    pose_video_path = os.path.join(pose_root, vid + '.mp4')

                    all_ref_images = read_frames(ref_video_path)
                    #assert len(pose_images) == len(ref_images), f"{len(pose_images) = } != {len(ref_images) = }"
                    ref_random_idx = np.random.randint(0, len(all_ref_images))
                    ref_image = all_ref_images[ref_random_idx]
                    ref_image_tensor= image_transform(ref_image)

                    all_pose_images = read_frames(pose_video_path)
                    tgt_random_idx = np.random.randint(0, len(all_pose_images))
                    tgt_random_idx = ref_random_idx
                    pose_image = all_pose_images[tgt_random_idx]
                    pose_tensor_list = [image_transform(pose_image)]
                    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (1, c, h, w)

                ref_name = str(ref_random_idx)
                pose_name = str(tgt_random_idx)
               
                image = pipe(
                    ref_image,
                    pose_image,
                    width,
                    height,
                    steps,
                    cfg,
                    generator=generator,
                ).images

                image = image.squeeze(2)

                if config.data_type == 'frames':
                    gt_img = Image.open(all_ref_images[tgt_random_idx]).convert('RGB')
                else:
                    gt_img = all_ref_images[tgt_random_idx]
                gt_img = torch.stack([image_transform(gt_img)], dim=0)
                ssims.append(ssim(image, gt_img))
                psnrs.append(psnr(image, gt_img))
                lpipss.append(lpips(image, gt_img, net_type='vgg'))
                image = torch.cat([ref_image_tensor.unsqueeze(0), pose_tensor, image, gt_img], dim=0)
                save_image_grid(
                        image,
                        f"{save_dir}/{repeat_ind}_{vid}_{ref_name}_{pose_name}_{height}x{width}_{int(cfg)}_{time_str}.jpg",
                        n_rows=4,
                )

    ssim_result = torch.tensor(ssims).mean()
    psnr_result = torch.tensor(psnrs).mean()
    lpips_result = torch.tensor(lpipss).mean()
    result_dict = {"SSIM": ssim_result.item(), "PSNR": psnr_result.item(), "LPIPS": lpips_result.item()}
    print("SSIM : {:>12.7f}".format(ssim_result, ".5"))
    print("PSNR : {:>12.7f}".format(psnr_result, ".5"))
    print("LPIPS: {:>12.7f}".format(lpips_result, ".5"))
    with open(os.path.join(save_dir, "results.json"), 'w') as fp:
        json.dump(result_dict, fp, indent=True)

if __name__ == "__main__":
    main()
