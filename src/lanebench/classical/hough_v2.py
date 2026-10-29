"""
L0: FCW v2 원본 차선 검출 (Canny + HoughLinesP + fitLine)

monocular-forward-collision-warning/Final_exam/src/{preprocessing,lane}.py 를
그대로 옮겼습니다. 파라미터와 로직은 바꾸지 않았고, 출력만 공통 Prediction
형식으로 감쌌습니다.
"""
from __future__ import annotations

import cv2
import numpy as np

from lanebench.prediction import EVAL_SIZE, Prediction, render


# ── preprocessing.py ─────────────────────────────────────────────────────────
def compute_visibility_score(gray: np.ndarray) -> float:
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
    contrast_score = min(std_brightness / 64.0, 1.0)
    return float(np.clip(0.5 * brightness_score + 0.5 * contrast_score, 0.0, 1.0))


def preprocess(img: np.ndarray):
    """CLAHE 강화 전처리 (v2). 반환: gray, equalized, blurred, edges, visibility_score"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis_score = compute_visibility_score(gray)
    if vis_score < 0.5:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
    else:
        equalized = gray.copy()
    blurred = cv2.GaussianBlur(equalized, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, equalized, blurred, edges, vis_score


# ── lane.py ──────────────────────────────────────────────────────────────────
def detect_lane_segments(edges: np.ndarray, img_shape: tuple):
    """HoughLinesP → 기울기 부호로 좌/우 선분 분리 (v2 detect_lanes 의 앞부분)."""
    h = img_shape[0]
    roi_y = int(h * 0.45)
    roi_edges = edges.copy()
    roi_edges[:roi_y, :] = 0

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=80, maxLineGap=60)
    left_lines: list[tuple] = []
    right_lines: list[tuple] = []
    if lines is not None:
        # OpenCV 4.x는 (N,1,4), 5.x는 (N,4)를 돌려줍니다
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines


def fit_lane_line(lines: list[tuple], h: int) -> tuple[int, int, int, int] | None:
    """여러 선분 → 하나의 대표 직선 (cv2.fitLine). y = 0.55h ~ h 구간."""
    if not lines:
        return None
    pts = []
    for x1, y1, x2, y2 in lines:
        pts.extend([(x1, y1), (x2, y2)])
    pts = np.array(pts, dtype=np.float32)
    vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten())
    if abs(vy) < 1e-6:
        return None
    y_bot = h
    y_top = int(h * 0.55)
    t_bot = (y_bot - y0) / vy
    t_top = (y_top - y0) / vy
    return (int(x0 + t_top * vx), y_top, int(x0 + t_bot * vx), y_bot)


class HoughV2:
    """L0. 좌/우 직선이 둘 다 있어야 차로 마스크를 만듭니다 (v2와 동일)."""
    name = "L0"

    def predict(self, img: np.ndarray, out_size=EVAL_SIZE) -> Prediction:
        h, w = img.shape[:2]
        *_, edges, vis = preprocess(img)
        left_segs, right_segs = detect_lane_segments(edges, img.shape)
        left, right = fit_lane_line(left_segs, h), fit_lane_line(right_segs, h)

        lines = [np.array([[l[0], l[1]], [l[2], l[3]]]) for l in (left, right) if l]
        polygon = None
        if left and right:
            polygon = np.array([[left[0], left[1]], [right[0], right[1]],
                                [right[2], right[3]], [left[2], left[3]]])
        return render(polygon, lines, (w, h), out_size,
                      info={"visibility": vis, "n_left": len(left_segs),
                            "n_right": len(right_segs)})
