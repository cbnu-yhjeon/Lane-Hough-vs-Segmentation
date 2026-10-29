"""BDD100K 학습용 Dataset: 영상 + 3-class 주행가능영역 + 차선 마스크 (640x360)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lanebench.data.bdd import drivable_label, load_frame, rasterize_polylines
from lanebench.prediction import EVAL_SIZE

TRAIN_LANE_THICKNESS = 3   # 640x360 기준. 원본에서 약 6 px


def read_image(path: str, size=EVAL_SIZE) -> np.ndarray:
    """BGR 영상을 size로. 1280x720이면 JPEG 디코딩 단계에서 바로 1/2로 줄입니다."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img.shape[1] != size[0] or img.shape[0] != size[1]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(img_bgr[..., ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)


class BDDLaneDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, train: bool = True, size=EVAL_SIZE):
        self.rows = manifest[["image", "label"]].to_numpy()
        self.train = train
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img_path, label_path = self.rows[i]
        img = read_image(img_path, self.size)
        fr = load_frame(label_path)
        drivable = drivable_label(fr, self.size)
        lane = rasterize_polylines(fr.lanes, self.size, TRAIN_LANE_THICKNESS)
        if self.train:
            if np.random.rand() < 0.5:
                img, drivable, lane = img[:, ::-1], drivable[:, ::-1], lane[:, ::-1]
            # 밝기 · 대비 지터 (야간 · 역광 다양성)
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return (to_tensor(img),
                torch.from_numpy(np.ascontiguousarray(drivable)).long(),
                torch.from_numpy(np.ascontiguousarray(lane)).float()[None])
