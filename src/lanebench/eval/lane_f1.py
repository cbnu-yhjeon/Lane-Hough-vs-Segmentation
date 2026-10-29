"""
픽셀 허용 오차 내 차선 F1

GT 차선은 두께 1 중심선으로 래스터화합니다. 예측은 방법마다 형식이 달라도
(고전 = 선, 분할 = 띠) 같은 규칙으로 셉니다.

    precision = 예측 픽셀 중 GT 중심선에서 τ px 안에 있는 비율
    recall    = GT 중심선 픽셀 중 예측에서 τ px 안에 있는 비율
"""
from __future__ import annotations

import cv2
import numpy as np

TOLERANCES = (2, 5, 10)   # 평가 해상도(640x360) px. 원본 기준 4 · 10 · 20 px


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """mask 픽셀까지의 거리 맵. mask가 비면 무한대."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def lane_counts(pred: np.ndarray, gt: np.ndarray,
                tolerances=TOLERANCES) -> dict[str, int]:
    """조건별 재집계를 위한 픽셀 수. 키: n_pred, n_gt, tp_pred@τ, tp_gt@τ."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    out = {"n_pred": int(pred.sum()), "n_gt": int(gt.sum())}
    d_gt, d_pred = _dist_to(gt), _dist_to(pred)
    for t in tolerances:
        out[f"tp_pred@{t}"] = int(np.count_nonzero(d_gt[pred] <= t))
        out[f"tp_gt@{t}"] = int(np.count_nonzero(d_pred[gt] <= t))
    return out


def prf(tp_pred: float, n_pred: float, tp_gt: float, n_gt: float) -> tuple[float, float, float]:
    p = tp_pred / n_pred if n_pred else 0.0
    r = tp_gt / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
