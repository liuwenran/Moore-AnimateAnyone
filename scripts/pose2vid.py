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
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=784)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--name", type=str, default='temp')
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

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

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

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--{args.name}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)


    image_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()])
    # for ref_image_path in config["test_cases"].keys():
    #     # Each ref_image may correspond to multiple actions
    #     for pose_video_path in config["test_cases"][ref_image_path]:
    for ref_image_root in config['test_cases']:
        pose_root = config["test_cases"][ref_image_root][0]
        # ref_name = Path(ref_image_path).stem
        # pose_name = Path(pose_video_path).stem.replace("_kps", "")

        # ref_image_pil = Image.open(ref_image_path).convert("RGB")

        all_vids = os.listdir(ref_image_root)
        all_vids.sort()
        all_ref_videos = [os.path.join(ref_image_root, f) for f in all_vids]

        for ind, ref_video_path in enumerate(all_ref_videos):
            print(f'ind {ind} in {len(all_ref_videos)}')
            if config.data_type == 'frames':
                vid = ref_video_path.split('/')[-1]
                ref_video_path = os.path.join(ref_image_root, vid, 'images')
                pose_video_path = os.path.join(pose_root, vid, 'dwpose')

                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                frame_files = os.listdir(ref_video_path)
                frame_files = [os.path.join(ref_video_path, f) for f in frame_files]
                frame_files.sort()
                all_ref_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_ref_images.append(frame)
                # ref_random_idx = np.random.randint(0, len(all_ref_images))
                ref_random_idx = 0
                ref_image_name = all_ref_images[ref_random_idx]
                ref_image = Image.open(ref_image_name).convert('RGB')
                ref_image_pil = ref_image
                ref_image_tensor= image_transform(ref_image)

                frame_files = os.listdir(pose_video_path)
                frame_files = [os.path.join(pose_video_path, f) for f in frame_files]
                frame_files.sort()
                all_pose_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_pose_images.append(frame)
                if len(all_pose_images) > 72:
                    all_pose_images = all_pose_images[::3]
                pose_list = [Image.open(pose_image_path).convert('RGB') for pose_image_path in all_pose_images]
                pose_tensor_list = [image_transform(pose_image) for pose_image in pose_list]
                pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (1, c, h, w)

            else:
                vid = ref_video_path.split('/')[-1].split('.')[0]
                pose_video_path = os.path.join(pose_root, vid + '.mp4')

                all_ref_images = read_frames(ref_video_path)
                #assert len(pose_images) == len(ref_images), f"{len(pose_images) = } != {len(ref_images) = }"
                ref_random_idx = 0
                # ref_random_idx = np.random.randint(0, len(all_ref_images))
                ref_image = all_ref_images[ref_random_idx]
                ref_image_pil = ref_image
                ref_image_tensor= image_transform(ref_image)

                pose_list = read_frames(pose_video_path)
                if len(pose_list) > 72:
                    pose_list = pose_list[::3]
                pose_tensor_list = [image_transform(pose_image) for pose_image in pose_list]
                pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (1, c, h, w)

            # pose_list = []
            # pose_tensor_list = []
            # pose_images = read_frames(pose_video_path)
            # src_fps = get_fps(pose_video_path)
            # print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
            # pose_transform = transforms.Compose(
            #     [transforms.Resize((height, width)), transforms.ToTensor()]
            # )
            # for pose_image_pil in pose_images[: args.L]:
            #     pose_tensor_list.append(pose_transform(pose_image_pil))
            #     pose_list.append(pose_image_pil)

            # ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
                0
            )  # (1, c, 1, h, w)
            ref_image_tensor = repeat(
                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=args.L
            )

            # pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)
            pose_tensor = pose_tensor.unsqueeze(0)

            video = pipe(
                ref_image_pil,
                pose_list,
                width,
                height,
                args.L,
                args.steps,
                args.cfg,
                generator=generator,
            ).videos

            # video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
            video = torch.cat([video], dim=0)
            save_videos_grid(
                video,
                f"{save_dir}/{args.name}_{vid}_{ref_random_idx}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
                n_rows=1,
                fps=args.fps,
            )


if __name__ == "__main__":
    main()
