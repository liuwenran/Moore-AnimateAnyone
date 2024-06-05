import copy
import os, sys
import math
import numpy as np
import cv2
sys.path.append("OpenSeeFace/")
from tracker import Tracker, get_model_base_path

features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]


def face_image(frame, save_path=None):
    height, width, c = frame.shape
    tracker = Tracker(width, height, threshold=None, max_threads=1, max_faces=1, discard_after=10, scan_every=3, silent=False, model_type=3, model_dir=None, 
                      no_gaze=False, detection_threshold=0.4, use_retinaface=0, max_feature_updates=900, static_model=True, try_hard=False)
    faces = tracker.predict(frame)
    frame = np.zeros_like(frame)
    detected = False
    face_lms = None
    for face_num, f in enumerate(faces):
        f = copy.copy(f)
        if f.eye_blink is None:
            f.eye_blink = [1, 1]
        right_state = "O" if f.eye_blink[0] > 0.30 else "-"
        left_state = "O" if f.eye_blink[1] > 0.30 else "-"
        detected = True
        if not f.success:
            pts_3d = np.zeros((70, 3), np.float32)
        if face_num == 0:
            face_lms = f.lms
        for pt_num, (x,y,c) in enumerate(f.lms):
            if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.20):
                continue
            if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.20):
                continue
            x = int(x + 0.5)
            y = int(y + 0.5)

            color = (0, 255, 0)
            if pt_num >= 66:
                color = (255, 255, 0)
            if not (x < 0 or y < 0 or x >= height or y >= width):
                cv2.circle(frame, (y, x), 1, color, -1)
        if f.rotation is not None:
            projected = cv2.projectPoints(f.contour, f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
            for [(x,y)] in projected[0]:
                x = int(x + 0.5)
                y = int(y + 0.5)
                if not (x < 0 or y < 0 or x >= height or y >= width):
                    frame[int(x), int(y)] = (0, 255, 255)
                x += 1
                if not (x < 0 or y < 0 or x >= height or y >= width):
                    frame[int(x), int(y)] = (0, 255, 255)
                y += 1
                if not (x < 0 or y < 0 or x >= height or y >= width):
                    frame[int(x), int(y)] = (0, 255, 255)
                x -= 1
                if not (x < 0 or y < 0 or x >= height or y >= width):
                    frame[int(x), int(y)] = (0, 255, 255)
    if save_path is not None:
        cv2.imwrite(save_path, frame)
    return frame, face_lms

if __name__ == '__main__':
    img_lines = open('/mnt/petrelfs/liuwenran/datasets/moore_test_case/redbook/redbook_pose_half.txt').read().splitlines()
    save_dir = '/mnt/petrelfs/liuwenran/datasets/moore_test_case/redbook/redbook_pose_half_crop'

    for ind, line in enumerate(img_lines):
        print(f'ind {ind} in {len(img_lines)}')
        img_name = line.split('/')[-1]
        img_id = img_name.split('.')[0]
        driving_img = cv2.imread(line)
        frame, face_lms = face_image(driving_img)
        min_loc = np.min(face_lms, axis=0)
        max_loc = np.max(face_lms, axis=0)
        min_x, min_y = int(min_loc[0]), int(min_loc[1])
        max_x, max_y = int(max_loc[0]), int(max_loc[1])
        x_sub = max_x - min_x
        y_sub = max_y - min_y
        x_start = min_x - int((512 - x_sub)/2)
        y_start = min_y - int((512 - y_sub)/2)
        driving_img_crop = driving_img[x_start:x_start+512, y_start:y_start+512, :]
        crop_start_position = np.array([y_start, x_start])
        cv2.imwrite(os.path.join(save_dir, img_name), driving_img_crop)
        np.save(os.path.join(save_dir, img_id + '.npy'), crop_start_position)
