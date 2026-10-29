"""
L1: IPM + 색 · 그래디언트 이진화 + 슬라이딩 윈도우 + 2차 다항식
L1+: L1의 밝기(L) 채널에 CLAHE를 추가 (야간 보완)

BDD100K는 영상마다 카메라 보정 정보가 없습니다. 그래서 IPM 사다리꼴을
소실점(VP) 기준으로 잡습니다. VP는 영상마다 Hough 선분의 좌/우 교점으로
추정하고, 못 찾으면 학습 셋 GT 차선 교점의 중앙값(DEFAULT_VP)을 씁니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lanebench.prediction import EVAL_SIZE, Prediction, render

# 학습 셋 GT 직선 차선 쌍의 교점 중앙값 (1280x720 기준, 비율로 저장)
DEFAULT_VP = (593 / 1280, 340 / 720)


@dataclass
class IPMConfig:
    bev_w: int = 480
    bev_h: int = 640
    bottom_y: float = 0.80           # IPM 사다리꼴 아랫변 (H 비율). BDD 영상 다수에 보닛이 보임
    ego_bottom: float = 0.90         # 차로 polygon을 이 행까지 연장 (학습 셋 GT 하단 중앙값)
    top_gap: float = 0.07            # VP 아래로 이만큼 떨어진 곳이 윗변 (H 비율)
    bottom_half_width: float = 0.6   # 아랫변 반폭 (W 비율). 옆 차로 차선까지 포함
    estimate_vp: bool = True         # False면 항상 DEFAULT_VP
    vp_window: tuple = (0.12, 0.10)  # 추정 VP가 DEFAULT_VP에서 벗어날 수 있는 범위 (W, H 비율)
    vp_max_spread: float = 25.0      # 교점들의 가중 MAD가 이보다 크면 추정을 버림 (px)
    # 이진화
    tophat_width: int = 31           # 이보다 좁은 밝은 세로 구조 = 차선 후보 (BEV px)
    tophat_min: float = 18.0
    sobel_min: float = 40.0
    # 슬라이딩 윈도우
    n_windows: int = 12
    margin: int = 28
    min_pix: int = 20
    min_lane_pix: int = 120
    lane_width_bev: float = 150.0
    max_bend: float = 0.35           # 2차 곡선이 직선에서 차로 폭의 이 비율 넘게 휘면 직선으로    # 기대 차로 폭 (BEV px). 한쪽만 찾으면 이만큼 평행 이동
    clahe: bool = False


class IPMPoly:
    def __init__(self, cfg: IPMConfig | None = None, name: str = "L1"):
        self.cfg = cfg or IPMConfig()
        self.name = name
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ── 1. 소실점 · 원근 변환 ───────────────────────────────────────────────
    def estimate_vp(self, gray: np.ndarray) -> tuple[float, float, bool]:
        h, w = gray.shape
        dx0, dy0 = DEFAULT_VP[0] * w, DEFAULT_VP[1] * h
        if not self.cfg.estimate_vp:
            return dx0, dy0, False
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 150)
        edges[: int(h * 0.5)] = 0
        edges[int(h * self.cfg.bottom_y):] = 0
        segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=20)
        if segs is None:
            return dx0, dy0, False
        segs = segs.reshape(-1, 4).astype(np.float64)
        dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        ok = np.abs(dx) > 1
        slope = np.where(ok, dy / np.where(ok, dx, 1), 0)
        length = np.hypot(dx, dy)
        keep = ok & (np.abs(slope) > 0.35) & (np.abs(slope) < 3.0)
        left = segs[keep & (slope < 0)]
        right = segs[keep & (slope > 0)]
        if len(left) < 2 or len(right) < 2:
            return dx0, dy0, False
        left = left[np.argsort(-length[keep & (slope < 0)])[:15]]
        right = right[np.argsort(-length[keep & (slope > 0)])[:15]]
        xs, ys, ws = [], [], []
        for a in left:
            for b in right:
                p = _intersect(a, b)
                if p is not None:
                    xs.append(p[0])
                    ys.append(p[1])
                    ws.append(np.hypot(a[2] - a[0], a[3] - a[1]) * np.hypot(b[2] - b[0], b[3] - b[1]))
        if not xs:
            return dx0, dy0, False
        xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
        x, y = _weighted_median(xs, ws), _weighted_median(ys, ws)
        spread = max(_weighted_median(np.abs(xs - x), ws), _weighted_median(np.abs(ys - y), ws))
        wx, wy = self.cfg.vp_window
        if (spread > self.cfg.vp_max_spread
                or abs(x - dx0) > wx * w or abs(y - dy0) > wy * h):
            return dx0, dy0, False
        return float(x), float(y), True

    def homography(self, vp: tuple[float, float], w: int, h: int):
        c = self.cfg
        vx, vy = vp
        yb = c.bottom_y * h
        yt = vy + c.top_gap * h
        hw_b = c.bottom_half_width * w
        hw_t = hw_b * (yt - vy) / (yb - vy)
        src = np.float32([[vx - hw_t, yt], [vx + hw_t, yt], [vx + hw_b, yb], [vx - hw_b, yb]])
        dst = np.float32([[0, 0], [c.bev_w, 0], [c.bev_w, c.bev_h], [0, c.bev_h]])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

    # ── 2. 이진화 (HLS 색 + Sobel 그래디언트) ──────────────────────────────
    def binarize(self, bev: np.ndarray, valid: np.ndarray) -> np.ndarray:
        c = self.cfg
        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        hue, light, sat = hls[..., 0], hls[..., 1], hls[..., 2]
        if c.clahe:
            light = self._clahe.apply(light)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (c.tophat_width, 1))
        # 흰 차선: 주변보다 밝은 가는 세로 구조 (조명 변화에 덜 민감한 white top-hat)
        tophat = cv2.morphologyEx(light, cv2.MORPH_TOPHAT, k).astype(np.float32)
        vals = tophat[valid]
        thr = max(c.tophat_min, float(vals.mean() + 3.0 * vals.std())) if vals.size else c.tophat_min
        white = tophat > thr
        # 노란 차선: HLS 색상 범위 + 채도
        yellow = (hue >= 15) & (hue <= 35) & (sat > 90) & (light > 50)
        yellow &= cv2.morphologyEx(sat, cv2.MORPH_TOPHAT, k) > 30
        # 그래디언트: x 방향 Sobel이 강하고 밝은 쪽에 붙은 픽셀
        sobel = np.abs(cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=3))
        grad = (sobel > c.sobel_min) & (tophat > 0.5 * thr)
        binary = (white | yellow | grad) & valid
        return binary.astype(np.uint8)

    # ── 3. 슬라이딩 윈도우 + 2차 다항식 ─────────────────────────────────────
    def fit_lanes(self, binary: np.ndarray):
        c = self.cfg
        H, W = binary.shape
        hist = binary[H // 2:].sum(0).astype(np.float32)
        hist = np.convolve(hist, np.ones(9) / 9, mode="same")
        mid = W // 2
        lo, hi = int(0.15 * c.lane_width_bev), int(1.0 * c.lane_width_bev)
        bases = {}
        for side, (a, b) in {"left": (mid - hi, mid - lo), "right": (mid + lo, mid + hi)}.items():
            seg = hist[max(a, 0):min(b, W)]
            if seg.size and seg.max() >= 8:
                # 자차에 가장 가까운 뚜렷한 봉우리 (옆 차로 차선을 잡지 않도록)
                strong = np.nonzero(seg >= 0.5 * seg.max())[0]
                idx = strong.max() if side == "left" else strong.min()
                bases[side] = max(a, 0) + int(idx)

        ys, xs = np.nonzero(binary)
        win_h = H // c.n_windows
        fits, counts = {}, {}
        for side, x0 in bases.items():
            cur = x0
            take = []
            for i in range(c.n_windows):
                y_hi = H - i * win_h
                y_lo = y_hi - win_h
                inside = ((ys >= y_lo) & (ys < y_hi) & (xs >= cur - c.margin)
                          & (xs < cur + c.margin)).nonzero()[0]
                take.append(inside)
                if len(inside) > c.min_pix:
                    cur = int(xs[inside].mean())
            idx = np.concatenate(take)
            counts[side] = len(idx)
            if len(idx) >= c.min_lane_pix and np.ptp(ys[idx]) > 0.25 * H:
                fits[side] = self._robust_fit(ys[idx], xs[idx], H)
        return fits, counts

    def _robust_fit(self, ys: np.ndarray, xs: np.ndarray, H: int) -> np.ndarray:
        """2차 다항식. 1차 직선과 BEV 높이 안에서 크게 어긋나면(희소 · 잡음 픽셀) 직선으로."""
        quad = np.polyfit(ys, xs, 2)
        lin = np.polyfit(ys, xs, 1)
        grid = np.linspace(0, H - 1, 16)
        if np.abs(np.polyval(quad, grid) - np.polyval(lin, grid)).max() > self.cfg.max_bend * self.cfg.lane_width_bev:
            return np.array([0.0, *lin])
        return quad

    def complete_pair(self, fits: dict, counts: dict):
        """폭이 말이 안 되면 픽셀이 많은 쪽만 남기고, 한쪽만 있으면 기대 폭만큼 평행 이동."""
        c = self.cfg
        y_eval = c.bev_h - 1
        if "left" in fits and "right" in fits:
            width = np.polyval(fits["right"], y_eval) - np.polyval(fits["left"], y_eval)
            width_top = np.polyval(fits["right"], 0) - np.polyval(fits["left"], 0)
            # 차로 폭이 범위 밖이거나, 위아래 폭이 크게 달라 평행하지 않으면 한쪽을 버림
            if (not 0.5 * c.lane_width_bev < width < 1.8 * c.lane_width_bev
                    or not 0.6 < width_top / width < 1.6):
                drop = "left" if counts["left"] < counts["right"] else "right"
                fits.pop(drop)
        extrapolated = None
        if len(fits) == 1:
            side, coef = next(iter(fits.items()))
            other = "right" if side == "left" else "left"
            shift = c.lane_width_bev if other == "right" else -c.lane_width_bev
            fits[other] = coef + np.array([0, 0, shift])
            extrapolated = other
        return fits, extrapolated

    # ── 전체 ────────────────────────────────────────────────────────────────
    def predict(self, img: np.ndarray, out_size=EVAL_SIZE) -> Prediction:
        c = self.cfg
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vx, vy, vp_ok = self.estimate_vp(gray)
        M, Minv = self.homography((vx, vy), w, h)

        bev = cv2.warpPerspective(img, M, (c.bev_w, c.bev_h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (c.bev_w, c.bev_h))
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8)) > 0
        binary = self.binarize(bev, valid)
        fits, counts = self.fit_lanes(binary)
        fits, extrapolated = self.complete_pair(fits, counts)

        info = {"vp_estimated": vp_ok, "vp": (vx, vy), "found": sorted(k for k in fits
                if k != extrapolated), "extrapolated": extrapolated}
        if len(fits) < 2:
            return render(None, [], (w, h), out_size, info)

        # BEV 아래쪽을 ego_bottom 행까지 연장해서 원래 시점으로 되돌림
        y_img_bottom = cv2.perspectiveTransform(
            np.float32([[[vx, c.ego_bottom * h]]]), M)[0, 0, 1]
        ys = np.linspace(0, y_img_bottom, 48)
        left = np.stack([np.polyval(fits["left"], ys), ys], 1)
        right = np.stack([np.polyval(fits["right"], ys), ys], 1)
        to_img = lambda p: cv2.perspectiveTransform(p[None].astype(np.float32), Minv)[0]
        left_img, right_img = to_img(left), to_img(right)
        polygon = np.concatenate([left_img, right_img[::-1]])
        return render(polygon, [left_img, right_img], (w, h), out_size, info)


def _intersect(a: np.ndarray, b: np.ndarray):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, 0.5 * cw[-1])])


def make_l1() -> IPMPoly:
    return IPMPoly(IPMConfig(clahe=False), name="L1")


def make_l1_plus() -> IPMPoly:
    return IPMPoly(IPMConfig(clahe=True), name="L1+")
