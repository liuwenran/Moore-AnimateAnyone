from metrics.utils.loss_utils import ssim
from metrics.lpipsPyTorch import lpips
import json
from tqdm import tqdm
from metrics.utils.image_utils import psnr
from pathlib import Path
from decord import VideoReader
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from src.utils.util import get_fps, read_frames, save_image_grid
import numpy as np
from datetime import datetime

# moore result
# dataset_name = 'pexels-test-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/Moore-AnimateAnyone/output/20240604/2112--video_pexels-v--seed_42-768x768'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-v/videos'

# dataset_name = 'tiktok'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/Moore-AnimateAnyone/output/20240604/2113--video_tiktok--seed_42-768x768'
# gt_video_root_path = '/mnt/hwfile/mm_lol/fangyq/share_data/tiktok_video'

# magic animate result
# dataset_name = 'magic-animate_tiktok'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/magic-animate/samples/tiktok--2024-06-05T16-33-58'
# gt_video_root_path = '/mnt/hwfile/mm_lol/fangyq/share_data/tiktok_video'

# dataset_name = 'magic-animate_pexels-test-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/magic-animate/samples/pexels-h--2024-06-05T16-39-26'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-h/videos'

# dataset_name = 'magic-animate_pexels-test-v'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/magic-animate/samples/pexels-v--2024-06-05T16-57-53'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-v/videos'

# champ result
# dataset_name = 'champ_tiktok'
# res_video_root_path = '/mnt/petrelfs/liuwenran/repos/champ/results/tiktok-2024-06-05T12-05-19'
# gt_video_root_path = '/mnt/hwfile/mm_lol/fangyq/share_data/tiktok_video'

# dataset_name = 'champ_pexels-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/repos/champ/results/pexels-h-2024-06-05T19-08-03'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-h/videos'

dataset_name = 'champ_pexels-v'
res_video_root_path = '/mnt/petrelfs/liuwenran/repos/champ/results/pexels-v-2024-06-05T21-58-38'
gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-v/videos'


setting = 'sample-once'

date_str = datetime.now().strftime("%Y%m%d")
time_str = datetime.now().strftime("%H%M")
save_dir = Path(f"output/images_metrics_result/{date_str}-{time_str}_{dataset_name}")
save_dir.mkdir(exist_ok=True, parents=True)

all_videos = os.listdir(res_video_root_path)
all_videos = [os.path.join(res_video_root_path, f) for f in all_videos]
all_videos.sort()
all_videos = [ video for video in all_videos if os.path.splitext(video)[1] == '.mp4']

ssims = []
psnrs = []
lpipss = []
for ind, video_path in enumerate(all_videos):
    print(f'ind {ind} in {len(all_videos)}')
    res_video_reader = VideoReader(video_path)
    vid = video_path.split('/')[-1].split('.')[0].split('_')[2]
    gt_video_path = os.path.join(gt_video_root_path, f'{vid}.mp4')
    gt_video_reader = VideoReader(gt_video_path)
    evaluate_inds = list(range(0, len(gt_video_reader)))
    if len(evaluate_inds) > 72:
        evaluate_inds = evaluate_inds[::3]

    if setting == 'sample-once':
        sample_ind = np.random.randint(0, len(res_video_reader))

        frame = res_video_reader[sample_ind]
        res_frame = frame.asnumpy()
        res_frame = Image.fromarray(res_frame).convert('RGB')
        width, height = res_frame.size

        eval_ind = evaluate_inds[sample_ind]
        gt_frame = gt_video_reader[eval_ind].asnumpy()
        gt_frame = Image.fromarray(gt_frame).convert('RGB')

        image_transform = transforms.Compose(
                    [transforms.Resize((height, width)), transforms.ToTensor()])
        image = torch.stack([image_transform(res_frame)], dim=0)
        gt_img = torch.stack([image_transform(gt_frame)], dim=0)
        ssims.append(ssim(image, gt_img))
        psnrs.append(psnr(image, gt_img))
        lpipss.append(lpips(image, gt_img, net_type='vgg'))
        image = torch.cat([image, gt_img], dim=0)
        save_image_grid(
                image,
                f"{save_dir}/{vid}_{sample_ind}.jpg",
                n_rows=4,
        )
    elif setting == 'sample-all':
        for frame_ind, frame in enumerate(res_video_reader):
            res_frame = frame.asnumpy()
            res_frame = Image.fromarray(res_frame).convert('RGB')
            width, height = res_frame.size

            eval_ind = evaluate_inds[frame_ind]
            gt_frame = gt_video_reader[eval_ind].asnumpy()
            gt_frame = Image.fromarray(gt_frame).convert('RGB')

            image_transform = transforms.Compose(
                        [transforms.Resize((height, width)), transforms.ToTensor()])
            image = torch.stack([image_transform(res_frame)], dim=0)
            gt_img = torch.stack([image_transform(gt_frame)], dim=0)
            ssims.append(ssim(image, gt_img))
            psnrs.append(psnr(image, gt_img))
            lpipss.append(lpips(image, gt_img, net_type='vgg'))
            image = torch.cat([image, gt_img], dim=0)
            save_image_grid(
                    image,
                    f"{save_dir}/{vid}_{frame_ind}.jpg",
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