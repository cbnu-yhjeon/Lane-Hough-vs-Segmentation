"""
주행가능영역 · 자차 차로 IoU

조건별로 다시 묶을 수 있게 이미지마다 교집합 · 합집합 픽셀 수를 남기고,
집계 때 합산합니다 (micro IoU).
"""
from __future__ import annotations

import numpy as np


def iou_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    """이진 마스크 → (교집합, 합집합) 픽셀 수."""
    p, g = pred.astype(bool), gt.astype(bool)
    return int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g))


def confusion(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    """다중 클래스 라벨 맵 → (n,n) 혼동 행렬. 행 = GT, 열 = 예측."""
    idx = gt.astype(np.int64).ravel() * n_classes + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def miou_from_confusion(cm: np.ndarray) -> tuple[float, np.ndarray]:
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    ious = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(ious)), ious


def ego_bg_miou(inter: float, union: float, n_pixels: float) -> float:
    """자차 차로/배경 2-class mIoU. 배경 IoU는 같은 합계에서 계산됩니다."""
    ego = inter / union if union else np.nan
    bg_inter = n_pixels - union
    bg_union = n_pixels - inter
    bg = bg_inter / bg_union if bg_union else np.nan
    return float(np.nanmean([ego, bg]))
