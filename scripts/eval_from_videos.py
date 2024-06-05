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
from metrics.utils.fvd import calculate_fvd, load_i3d_pretrained
from datetime import datetime


# moore results
# dataset_name = 'pexels-test-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/Moore-AnimateAnyone/output/20240604/2056--video_pexels-h--seed_42-768x768'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-h/videos'

# dataset_name = 'pexels-test-v'
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
save_dir = Path(f"output/video_metrics_result/{date_str}-{time_str}_{dataset_name}")
save_dir.mkdir(exist_ok=True, parents=True)

all_videos = os.listdir(res_video_root_path)
all_videos = [os.path.join(res_video_root_path, f) for f in all_videos]
all_videos.sort()
all_videos = [ video for video in all_videos if os.path.splitext(video)[1] == '.mp4']

device = torch.device("cuda")
i3d = load_i3d_pretrained(device=device)

video1_all_tensors = []
video2_all_tensors = []
result_all = []
for ind, video_path in enumerate(all_videos):
    print(f'ind {ind} in {len(all_videos)}')
    res_video_reader = VideoReader(video_path)
    vid = video_path.split('/')[-1].split('.')[0].split('_')[2]
    gt_video_path = os.path.join(gt_video_root_path, f'{vid}.mp4')
    gt_video_reader = VideoReader(gt_video_path)
    evaluate_inds = list(range(0, len(gt_video_reader)))
    if len(evaluate_inds) > 72:
        evaluate_inds = evaluate_inds[::3]
    evaluate_inds = evaluate_inds[:24]

    width, height = None, None
    video1_images = []
    for frame_ind, frame in enumerate(res_video_reader):
        res_frame = frame.asnumpy()
        res_frame = Image.fromarray(res_frame).convert('RGB')
        width, height = res_frame.size

        image_transform = transforms.Compose(
                    [transforms.Resize((height, width)), transforms.ToTensor()])
        video1_images.append(image_transform(res_frame))
    video1 = torch.stack(video1_images, dim=0)
    
    video2_images = []
    for eval_ind in evaluate_inds:
        gt_frame = gt_video_reader[eval_ind].asnumpy()
        gt_frame = Image.fromarray(gt_frame).convert('RGB')

        image_transform = transforms.Compose(
                    [transforms.Resize((height, width)), transforms.ToTensor()])
        video2_images.append(image_transform(gt_frame))
    video2 = torch.stack(video2_images, dim=0)

    video1 = video1.unsqueeze(0)
    video2 = video2.unsqueeze(0)


    result = calculate_fvd(video1, video2, device, i3d=i3d, method='styleganv')

    values = list(result['value'].values())
    result_all.extend(values)

fvd_result = torch.tensor(result_all).mean().item()
print(f'FVD: {fvd_result}')
result_dict = {"fvd": fvd_result}
with open(os.path.join(save_dir, "results.json"), 'w') as fp:
    json.dump(result_dict, fp, indent=True)

# with open(os.path.join(save_dir, "results.json"), 'w') as fp:
#     json.dump(result, fp, indent=True)

#     NUMBER_OF_VIDEOS = 8
#     VIDEO_LENGTH = 50
#     CHANNEL = 3
#     SIZE = 64
#     videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     device = torch.device("cuda")
#     # device = torch.device("cpu")

