"""
FCW v2 위험도 점수 (monocular-forward-collision-warning/Final_exam/src/risk.py)

영향 분석에서는 차선 판정(in_lane)만 바꾸고 나머지 항은 같게 둡니다.
GT 박스에는 특징점 · TTC · 거리 정보가 없으므로 v2에서 그 값이 없을 때의
기본값을 씁니다 (feature_score = 0, ttc_weight = 1, 크기 점수는 박스 면적).
"""
from __future__ import annotations

import numpy as np

SAFE, CAUTION, DANGER = 0, 1, 2
LEVEL_NAMES = {SAFE: "SAFE", CAUTION: "CAUTION", DANGER: "DANGER"}

LANE_WEIGHT_IN = 1.1
LANE_WEIGHT_OUT = 0.85


def risk_score(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int,
               visibility_score: float, in_lane: bool) -> tuple[float, int]:
    """v2 assess_risk 와 같은 식. 반환: (score, level)."""
    area = (x2 - x1) * (y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    size_score = min(area / (img_w * img_h * 0.08), 1.0)
    dx = (cx - img_w / 2) / (img_w / 2)
    dy = (cy - img_h / 2) / (img_h / 2)
    center_score = 1.0 - float(np.clip(np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2), 0.0, 1.0))
    feature_score = 0.0
    vis_weight = 1.0 + 0.3 * (1.0 - visibility_score)
    ttc_weight = 1.0
    lane_weight = LANE_WEIGHT_IN if in_lane else LANE_WEIGHT_OUT

    base = 0.45 * size_score + 0.45 * center_score + 0.10 * feature_score
    score = float(np.clip(base * vis_weight * ttc_weight * lane_weight, 0.0, 1.0))
    level = DANGER if score >= 0.6 else (CAUTION if score >= 0.35 else SAFE)
    return score, level
