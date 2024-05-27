import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import PIL
from PIL import Image
from transformers import CLIPImageProcessor
import json
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import jsonlines
import copy
from controlnet_aux import HEDdetector, OpenposeDetector
import cv2
from decord import VideoReader


class AnimationImageCombineDataset(Dataset):
    def __init__(self, 
                 data_config, 
                 img_size=(512, 512),
                 control_type='pose',
                 img_scale=(1.0, 1.0),
                #  img_ratio=(0.55, 0.57),
                 img_ratio=(0.9, 1.0),
                 drop_ratio=0.1):
        super().__init__()
        self.sample_margin = data_config.sample_margin
        self.margin_strategy = data_config.margin_strategy

        self.img_size = img_size
        width, height = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.control_type = control_type
        if self.control_type == 'hed':
            # self.hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
            from controlnet_aux.hed import Network
            hed_net = Network('/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96/network-bsds500.pth')
            self.hed = HEDdetector(hed_net)
  
        self.all_video_path = []
        self.all_pose_path = []
        self.all_data_type = []
        for i in range(len(data_config.vid_file)):
            data_lines = open(data_config.vid_file[i], 'r').read().splitlines()
            video_path_lines = [os.path.join(data_config.video_root_path[i], data_line) for data_line in data_lines]
            pose_path_lines = [os.path.join(data_config.pose_root_path[i], data_line) for data_line in data_lines]
            self.all_video_path.extend(video_path_lines)
            self.all_pose_path.extend(pose_path_lines)

            data_lines_type = [data_config.data_type[i]] * len(data_lines)
            self.all_data_type.extend(data_lines_type)

        print(f'Dataset size: {len(self.all_video_path)}')

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio


    def __getitem__(self, index):
        try:
            video_path = self.all_video_path[index]
            pose_path = self.all_pose_path[index]
            data_type = self.all_data_type[index]

            if data_type == 'video':
                video_reader = VideoReader(video_path)
                kps_reader = VideoReader(pose_path)

                assert len(video_reader) == len(
                    kps_reader
                ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

                video_length = len(video_reader)
            else:
                if data_type == 'frames_tiktok':
                    video_path = os.path.join(video_path, 'images')
                frame_files = os.listdir(video_path)
                frame_files = [os.path.join(video_path, f) for f in frame_files]
                frame_files.sort()
                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_images.append(frame)
                
                if data_type == 'frames_tiktok':
                    pose_path = os.path.join(pose_path, 'dwpose')
                frame_files = os.listdir(pose_path)
                frame_files = [os.path.join(pose_path, f) for f in frame_files]
                frame_files.sort()
                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_pose_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_pose_images.append(frame)
                
                assert len(all_images) == len(
                    all_pose_images
                ), f"{len(all_images) = } != {len(all_pose_images) = } in {video_path}"

                video_length = len(all_images)


            tgt_image_ind = random.randint(0, video_length - 1)

            if self.margin_strategy == 'close':
                reference_image_ind = random.randint(-self.sample_margin, self.sample_margin) + tgt_image_ind
                if reference_image_ind < 0:
                    reference_image_ind = 0
                if reference_image_ind > video_length - 1:
                    reference_image_ind = video_length - 1
            elif self.margin_strategy == 'far':
                if tgt_image_ind + self.sample_margin < video_length:
                    reference_image_ind = random.randint(tgt_image_ind + self.sample_margin, video_length - 1)
                elif tgt_image_ind - self.sample_margin > 0:
                    reference_image_ind = random.randint(0, tgt_image_ind - self.sample_margin)
                else:
                    reference_image_ind = random.randint(0, video_length - 1)
            else:
                import sys
                sys.exit()

            reference_image = Image.open(all_images[reference_image_ind]).convert("RGB")
            tgt_image = Image.open(all_images[tgt_image_ind]).convert("RGB")

            control_image = None
            if self.control_type == 'canny':
                control_image = np.array(tgt_image)
                control_image = cv2.Canny(control_image, 100, 200)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)
            elif self.control_type == 'hed':
                control_image = self.hed(tgt_image)
            elif self.control_type == 'pose':
                control_image = Image.open(all_pose_images[tgt_image_ind]).convert("RGB")
            control_image = control_image.resize(tgt_image.size)

        except Exception:
            return self.__getitem__((index + 1) % len(self.data))

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_image, self.transform, state)
        tgt_pose_img = self.augmentation(control_image, self.cond_transform, state)
        ref_img_vae = self.augmentation(reference_image, self.transform, state)
        clip_image = self.clip_image_processor(
            images=reference_image, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
            ind=index
        )

        return sample

    def __len__(self):
        return len(self.all_video_path)

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)
