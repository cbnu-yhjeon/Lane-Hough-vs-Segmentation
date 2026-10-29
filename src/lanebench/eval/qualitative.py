"""
정성 비교 · 실패 사례 패널 (results/figures/qualitative_*.jpg)

    python -m lanebench.eval.qualitative

조건(야간 · 곡선 · 비/눈 · 맑은 주간 직선)마다 L1과 L2의 IoU 차이가 큰 영상과
평범한 영상을 골라 GT / L0 / L1 / L2를 나란히 그립니다.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from lanebench.classical.hough_v2 import HoughV2
from lanebench.classical.ipm_poly import make_l1
from lanebench.data.bdd import load_frame, rasterize_polygons, rasterize_polylines
from lanebench.eval.by_condition import load_images
from lanebench.paths import FIG_DIR, MANIFEST
from lanebench.prediction import EVAL_SIZE
from lanebench.segmentation.infer import CachedL2

EGO_BGR = (60, 170, 30)     # 자차 차로
LANE_BGR = (0, 220, 255)    # 차선
GT_BGR = (40, 40, 230)

CONDITIONS = {
    "night": lambda d: d.timeofday == "night",
    "curved": lambda d: d.road_shape == "curved",
    "rain_snow": lambda d: d.weather_g.isin(["rainy", "snowy"]),
    "clear_day_straight": lambda d: (d.weather_g == "clear") & (d.timeofday == "daytime")
    & (d.road_shape == "straight"),
}


def overlay(img: np.ndarray, ego: np.ndarray, lane: np.ndarray, color, title: str) -> np.ndarray:
    out = img.copy()
    m = ego > 0
    out[m] = (0.55 * out[m] + 0.45 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(ego.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    out[cv2.dilate(lane, np.ones((2, 2), np.uint8)) > 0] = LANE_BGR
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def panel(name: str, image_path: str, label_path: str, ious: dict) -> np.ndarray:
    full = cv2.imread(image_path)
    img = cv2.resize(full, EVAL_SIZE, interpolation=cv2.INTER_AREA)
    fr = load_frame(label_path)
    tiles = [overlay(img, rasterize_polygons(fr.direct, EVAL_SIZE),
                     rasterize_polylines(fr.lanes, EVAL_SIZE, 1), GT_BGR,
                     f"GT  {fr.weather} / {fr.timeofday}")]
    preds = {"L0": HoughV2().predict(full), "L1": make_l1().predict(full),
             "L2": CachedL2().load(name)[0]}
    for m, p in preds.items():
        tiles.append(overlay(img, p.ego_mask, p.lane_mask, EGO_BGR,
                             f"{m}  IoU {ious.get(m, float('nan')):.2f}"))
    return np.hstack(tiles)


def main(per_condition: int = 3):
    df = load_images()
    man = pd.read_csv(MANIFEST).set_index("name")
    d = df[df.has_direct].copy()
    d["iou"] = d.ego_inter / d.ego_union.where(d.ego_union > 0)
    wide = d.pivot_table(index="name", columns="method", values="iou").dropna()
    cond = d.drop_duplicates("name").set_index("name").loc[wide.index]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for cname, f in CONDITIONS.items():
        sel = wide[f(cond)]
        if sel.empty:
            continue
        gap = (sel["L2"] - sel["L1"]).sort_values()
        picks = list(gap.index[-per_condition + 1:]) + [gap.index[len(gap) // 2]]
        if "clear_day_straight" == cname:   # 고전 기법이 잘 되는 경우도 보여줌
            picks = list(sel["L1"].sort_values().index[-2:]) + [gap.index[len(gap) // 2]]
        rows = [panel(n, man.loc[n, "image"], man.loc[n, "label"], wide.loc[n].to_dict())
                for n in picks]
        grid = np.vstack(rows)
        cv2.imwrite(str(FIG_DIR / f"qualitative_{cname}.jpg"),
                    cv2.resize(grid, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(cname, picks)


if __name__ == "__main__":
    main()
