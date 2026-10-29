"""모든 방법(L0 · L1 · L1+ · L2)이 공통으로 내는 출력 형식."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# 평가 해상도 (W, H). 원본 1280x720의 1/2.
EVAL_SIZE = (640, 360)


@dataclass
class Prediction:
    """평가 해상도 기준 출력.

    ego_mask:  자차 차로 영역 (H,W) uint8 {0,1}. 검출 실패 시 전부 0.
    lane_mask: 차선 픽셀 (H,W) uint8 {0,1}. 고전 기법은 두께 1 선, 분할은 예측 영역.
    info:      방법별 디버그 정보 (소실점 출처, 한쪽 차선 외삽 여부 등).
    """
    ego_mask: np.ndarray
    lane_mask: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return bool(self.ego_mask.any())


def scale_points(pts: np.ndarray, src_size: tuple[int, int],
                 dst_size: tuple[int, int]) -> np.ndarray:
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    return np.round(np.asarray(pts, np.float64) * [sx, sy]).astype(np.int32)


def render(ego_polygon: np.ndarray | None, lane_lines: list[np.ndarray],
           src_size: tuple[int, int], out_size: tuple[int, int] = EVAL_SIZE,
           info: dict | None = None) -> Prediction:
    """원본 좌표의 polygon · polyline을 평가 해상도 마스크로 그립니다."""
    ego = np.zeros((out_size[1], out_size[0]), np.uint8)
    lanes = np.zeros_like(ego)
    if ego_polygon is not None and len(ego_polygon) >= 3:
        cv2.fillPoly(ego, [scale_points(ego_polygon, src_size, out_size)], 1)
    if lane_lines:
        cv2.polylines(lanes, [scale_points(p, src_size, out_size) for p in lane_lines],
                      False, 1, 1)
    return Prediction(ego, lanes, info or {})
