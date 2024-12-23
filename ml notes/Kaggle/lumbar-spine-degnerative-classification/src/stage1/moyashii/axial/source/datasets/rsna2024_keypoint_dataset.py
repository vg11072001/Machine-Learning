# reference: https://www.kaggle.com/code/hugowjd/rsna2024-lsdc-training-densenet#Define-Dataset
# reference: https://github.com/xingyizhou/CenterNet

import math
from typing import Optional

import os
import re
import glob

import cv2
import pydicom
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations as A

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from source.datasets.dataset_phase import DatasetPhase  # noqa


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class RSNA2024KeypointDatasetTrain(Dataset):
    def __init__(
        self,
        image_root: str,
        train_df: pd.DataFrame,
        phase: DatasetPhase = DatasetPhase.TRAIN,
        transform: A.Compose = None,
        heatmap_size: tuple[int, int] = (20, 20),
        stride: int = 4,
    ) -> None:
        self._image_root = Path(image_root)
        self._train_df = train_df.copy()
        self._phase = phase
        self._transform = transform
        self._heatmap_size = heatmap_size
        self._labels = {
            'spinal_canal': 0,
        }
        self._stride = stride

    def __len__(self):
        return len(self._train_df)

    def _read_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.astype(np.uint8)

    def __getitem__(self, idx):
        target_row = self._train_df.iloc[idx]
        study_id = target_row['study_id']
        image_path = target_row['png_path']
        image = self._read_image((self._image_root / image_path).as_posix())

        keypoints = [[target_row['x'], target_row['y']]]
        class_labels = ['spinal_canal']

        if self._transform is not None:
            transformed = self._transform(image=image, keypoints=keypoints, class_labels=class_labels)
            image = transformed['image']
            keypoints = transformed['keypoints']
            class_labels = transformed['class_labels']

        h, w = image.shape[:2]
        heatmap = np.zeros([len(self._labels), h // self._stride, w // self._stride])
        radius = gaussian_radius([s // self._stride for s in self._heatmap_size])
        radius = max(0, int(radius))
        for keypoint, class_label in zip(keypoints, class_labels):
            index = self._labels[class_label]
            kpt_x = keypoint[0] / self._stride
            kpt_y = keypoint[1] / self._stride
            draw_gaussian(heatmap[index], (kpt_x, kpt_y), radius)

        image = image.astype(np.float32)
        image = image[None, ...]

        ret_keypoints = [(-1.0, -1.0) for _ in range(len(self._labels))]
        for keypoint, class_label in zip(keypoints, class_labels):
            index = self._labels[class_label]
            ret_keypoints[index] = keypoint
        return image, heatmap, study_id, np.asarray(ret_keypoints)


if __name__ == '__main__':
    from source.datasets.rsna2024_keypoint_transforms import build_transforms
    image_root = '/kaggle/explorations/keypoint_detector/ver1/create_datasets/result'

    # train
    train_df = pd.read_csv('/kaggle/explorations/keypoint_detector/ver1/create_datasets/result/train_all_5folds.csv')

    transform = build_transforms(phase=DatasetPhase.TRAIN, image_size=(224, 224))
    train_dataset = RSNA2024KeypointDatasetTrain(image_root, train_df, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
    for batch in train_dataloader:
        break
