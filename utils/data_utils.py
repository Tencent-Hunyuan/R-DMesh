import torch
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
from PIL import Image
from PIL.Image import Resampling

try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS

def load_png_to_tensor(folder_path: str, num_frames: int = 64, samp_ratio: int = 1, video_width: int = 256) -> torch.Tensor:
    names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    indice_list = [min(i * samp_ratio, len(names) - 1) for i in range(num_frames)]
    names = [names[i] for i in indice_list]
    tensors = []
    for fn in names:
        p = os.path.join(folder_path, fn)
        with Image.open(p) as img:
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (0, 0, 0))
                bg.paste(img, (0, 0), img)
                img = bg
            else:
                img = img.convert('RGB')
            if img.size[0] != img.size[1]:
                img = TF.center_crop(img, min(img.size))
            if img.size != (video_width, video_width):
                img = img.resize((video_width, video_width), resample=LANCZOS_FILTER)
            t = TF.to_tensor(img).sub_(0.5).div_(0.5)
            tensors.append(t.unsqueeze(1))
    return torch.cat(tensors, dim=1)

def load_mp4_to_tensor(video_path: str, num_frames: int = 64, video_width: int = 256):
    """
    Load the first N frames from an MP4 video and process them.

    Returns video tensors: Center-cropped, resized to (video_width, video_width), and normalized.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to load and process.
        video_width (int): Target width and height for the resized tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - resized_video_tensor: Shape (C, F, video_width, video_width).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    resized_frames_tensors = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: video stream ended early, fewer than {num_frames} frames were read.")
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_cropped = TF.center_crop(img, min(img.size))
        img_resized = img_cropped.resize((video_width, video_width), resample=Resampling.LANCZOS)
        resized_frames_tensors.append(TF.to_tensor(img_resized).sub_(0.5).div_(0.5).unsqueeze(1))

    cap.release()

    if not resized_frames_tensors:
        raise ValueError(f"Failed to load any frames from '{video_path}'.")

    return torch.cat(resized_frames_tensors, dim=1)