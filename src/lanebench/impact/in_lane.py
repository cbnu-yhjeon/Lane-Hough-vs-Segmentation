"""
객체의 "차선 안/밖" 판정과 위험 등급 영향

기준점은 두 가지를 봅니다.
  bottom : 박스 아랫변 중앙 (도로와 닿는 점, 권장)
  center : 박스 중심 (v2 is_in_lane 이 쓰던 점)

GT 자차 차로 polygon은 앞차 바로 아래에서 끝나는 경우가 많아서, 차로 마스크를
아래 · 옆으로 조금 넓힌 뒤 조회합니다 (GT와 예측에 똑같이 적용).
예측 마스크가 비어 있으면 v2처럼 "차선 안"으로 봅니다.
"""
from __future__ import annotations

import cv2
import numpy as np

from lanebench.data.bdd import IMG_H, IMG_W, Box
from lanebench.impact.risk_v2 import risk_score

REF_POINTS = ("bottom", "center")
QUERY_DILATE = 5   # 평가 해상도 px (원본 약 10 px)


def query_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * QUERY_DILATE + 1, 2 * QUERY_DILATE + 1))
    return cv2.dilate(mask, k)


def ref_point(box: Box, ref: str) -> tuple[float, float]:
    cx = (box.x1 + box.x2) / 2
    return (cx, box.y2) if ref == "bottom" else (cx, (box.y1 + box.y2) / 2)


def in_lane(qmask: np.ndarray, x: float, y: float, empty_is_in: bool = True) -> bool:
    """qmask: query_mask() 결과 (평가 해상도). (x, y): 원본 좌표."""
    if not qmask.any():
        return empty_is_in
    h, w = qmask.shape
    xi = min(max(int(x * w / IMG_W), 0), w - 1)
    yi = min(max(int(y * h / IMG_H), 0), h - 1)
    return bool(qmask[yi, xi])


def object_rows(boxes: list[Box], gt_q: np.ndarray, pred_q: np.ndarray,
                visibility: float) -> list[dict]:
    """객체 × 기준점마다 GT 판정 / 방법 판정과 그에 따른 위험 등급."""
    rows = []
    for i, b in enumerate(boxes):
        for ref in REF_POINTS:
            x, y = ref_point(b, ref)
            gt_in = in_lane(gt_q, x, y, empty_is_in=False)
            pr_in = in_lane(pred_q, x, y, empty_is_in=True)
            gt_s, gt_l = risk_score(b.x1, b.y1, b.x2, b.y2, IMG_W, IMG_H, visibility, gt_in)
            pr_s, pr_l = risk_score(b.x1, b.y1, b.x2, b.y2, IMG_W, IMG_H, visibility, pr_in)
            rows.append({"obj": i, "category": b.category, "ref": ref, "gt_in": gt_in,
                         "pred_in": pr_in, "gt_level": gt_l, "pred_level": pr_l,
                         "gt_score": round(gt_s, 4), "pred_score": round(pr_s, 4)})
    return rows
