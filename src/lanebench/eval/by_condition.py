"""
조건별 집계: 날씨 · 시간대 · 장면 · 도로 형태 × 방법

    python -m lanebench.eval.by_condition

results/raw/per_image_*.csv, per_object_*.csv.gz 를 읽어 results/metrics/ 에
CSV와 요약 마크다운(summary.md)을 씁니다. IoU · F1은 이미지별 픽셀 수를
그룹 안에서 합산해 계산합니다 (micro).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from lanebench.data.manifest import CURVE_RATIO
from lanebench.eval.lane_f1 import prf
from lanebench.eval.miou import ego_bg_miou, miou_from_confusion
from lanebench.impact.risk_v2 import DANGER
from lanebench.paths import MANIFEST, METRICS_DIR, RAW_DIR

METHODS = ["L0", "L1", "L1+", "L2"]
WEATHER = {"clear": "clear", "partly cloudy": "clear", "overcast": "overcast",
           "rainy": "rainy", "snowy": "snowy", "foggy": "foggy"}
WEATHER_ORDER = ["clear", "overcast", "rainy", "snowy", "foggy"]
TIME_ORDER = ["daytime", "dawn/dusk", "night"]
TAU = 5   # 대표 허용 오차 (평가 해상도 px)
SHARP_RATIO = 0.05


def load_images() -> pd.DataFrame:
    files = [f for f in glob.glob(str(RAW_DIR / "per_image_*.csv")) if "smoke" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["name", "method"], keep="last")
    return _with_conditions(df)


def load_objects() -> pd.DataFrame:
    files = [f for f in glob.glob(str(RAW_DIR / "per_object_*.csv.gz")) if "smoke" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["name", "method", "obj", "ref"], keep="last")
    # 방법이 그 프레임에서 차로 마스크를 냈는지 (못 냈으면 v2 규칙대로 전부 "안")
    det = load_images()[["name", "method", "detected"]]
    return _with_conditions(df.merge(det, on=["name", "method"], how="left"))


def _with_conditions(df: pd.DataFrame) -> pd.DataFrame:
    man = pd.read_csv(MANIFEST, usecols=["name", "weather", "timeofday", "scene", "road_shape",
                                         "curvature"])
    df = df.merge(man, on="name", how="left")
    df["weather_g"] = df.weather.map(WEATHER).fillna("undefined")
    # 곡률 세분: 완만(0.02~0.05) / 급(>0.05)
    df["curve_bin"] = pd.cut(df.curvature, [-1, CURVE_RATIO, SHARP_RATIO, np.inf],
                             labels=["straight", "mild", "sharp"]).astype(str)
    df.loc[df.curvature.isna(), "curve_bin"] = "unknown"
    return df


def image_metrics(g: pd.DataFrame) -> pd.Series:
    d = g[g.has_direct]
    inter, union = d.ego_inter.sum(), d.ego_union.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        per_img = (d.ego_inter / d.ego_union.where(d.ego_union > 0)).fillna(0)
    out = {"n_images": len(g),
           "detect_rate": d.detected.mean() if len(d) else np.nan,
           "ego_iou": inter / union if union else np.nan,
           "ego_iou_img": per_img.mean() if len(d) else np.nan,
           "miou_ego_bg": ego_bg_miou(inter, union, d.n_pixels.sum()) if len(d) else np.nan}
    for t in (2, 5, 10):
        p, r, f = prf(g[f"tp_pred@{t}"].sum(), g.n_pred.sum(), g[f"tp_gt@{t}"].sum(), g.n_gt.sum())
        out.update({f"lane_p@{t}": p, f"lane_r@{t}": r, f"lane_f1@{t}": f})
    if "cm00" in g and g.cm00.notna().any():
        cm = np.array([[g[f"cm{i}{j}"].sum() for j in range(3)] for i in range(3)])
        out["miou3"], ious = miou_from_confusion(cm)
        out["iou_direct3"], out["iou_alt3"] = ious[1], ious[2]
    return pd.Series(out)


def object_metrics(g: pd.DataFrame) -> pd.Series:
    danger = g[g.gt_level == DANGER]
    return pd.Series({
        "n_objects": len(g),
        "disagree": (g.gt_in != g.pred_in).mean(),
        "false_in": (g.pred_in & ~g.gt_in).mean(),     # 밖인데 안으로 → 과경보 방향
        "false_out": (~g.pred_in & g.gt_in).mean(),    # 안인데 밖으로 → 미경보 방향
        "level_changed": (g.gt_level != g.pred_level).mean(),
        "level_down": (g.pred_level < g.gt_level).mean(),
        "level_up": (g.pred_level > g.gt_level).mean(),
        "danger_missed": (danger.pred_level < DANGER).mean() if len(danger) else np.nan,
        "n_gt_danger": len(danger),
        "fallback_frac": (~g.detected).mean(),
        "disagree_detected": (g.gt_in != g.pred_in)[g.detected].mean(),
        "disagree_fallback": (g.gt_in != g.pred_in)[~g.detected].mean(),
    })


def by(df: pd.DataFrame, keys: list[str], fn) -> pd.DataFrame:
    out = df.groupby(keys, sort=False).apply(fn, include_groups=False).reset_index()
    if "method" in keys:
        out["method"] = pd.Categorical(out.method, METHODS, ordered=True)
        out = out.sort_values(keys)
    return out


def fmt(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return d.to_markdown(index=False)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    img = load_images()
    obj = load_objects()
    obj["detected"] = obj.detected.astype(bool)

    tables = {
        "overall": by(img, ["method"], image_metrics),
        "by_weather": by(img, ["method", "weather_g"], image_metrics),
        "by_time": by(img, ["method", "timeofday"], image_metrics),
        "by_scene": by(img, ["method", "scene"], image_metrics),
        "by_road_shape": by(img, ["method", "road_shape"], image_metrics),
        "by_curve_bin": by(img, ["method", "curve_bin"], image_metrics),
        "by_weather_time": by(img, ["method", "weather_g", "timeofday"], image_metrics),
        "impact_overall": by(obj, ["method", "ref"], object_metrics),
        "impact_by_time": by(obj[obj.ref == "bottom"], ["method", "timeofday"], object_metrics),
        "impact_by_road_shape": by(obj[obj.ref == "bottom"], ["method", "road_shape"],
                                   object_metrics),
        "impact_by_weather": by(obj[obj.ref == "bottom"], ["method", "weather_g"],
                                object_metrics),
    }
    for name, t in tables.items():
        t.to_csv(METRICS_DIR / f"{name}.csv", index=False, float_format="%.4f")

    # L1+ − L1: CLAHE 효과 (시간대별)
    bt = tables["by_time"].set_index(["method", "timeofday"])
    if {"L1", "L1+"} <= set(bt.index.get_level_values(0)):
        delta = (bt.loc["L1+"][["ego_iou", "detect_rate", f"lane_f1@{TAU}"]]
                 - bt.loc["L1"][["ego_iou", "detect_rate", f"lane_f1@{TAU}"]]).reset_index()
        delta.to_csv(METRICS_DIR / "clahe_delta_by_time.csv", index=False, float_format="%.4f")
        tables["clahe_delta_by_time"] = delta

    main_cols = ["n_images", "detect_rate", "ego_iou", "ego_iou_img", "miou_ego_bg",
                 f"lane_p@{TAU}", f"lane_r@{TAU}", f"lane_f1@{TAU}"]
    imp_cols = ["n_objects", "disagree", "false_in", "false_out", "level_changed",
                "level_down", "level_up", "danger_missed", "fallback_frac", "disagree_detected"]
    md = ["# 지표 요약 (자동 생성)", "",
          "## 전체", fmt(tables["overall"], ["method", *main_cols,
                                             *[c for c in ("miou3",) if c in tables["overall"]]]),
          "", "## 시간대별", fmt(tables["by_time"], ["method", "timeofday", *main_cols]),
          "", "## 날씨별", fmt(tables["by_weather"], ["method", "weather_g", *main_cols]),
          "", "## 도로 형태별", fmt(tables["by_road_shape"], ["method", "road_shape", *main_cols]),
          "", "## 곡률 세분 (sagitta/chord: straight ≤ 0.02 < mild ≤ 0.05 < sharp)",
          fmt(tables["by_curve_bin"], ["method", "curve_bin", *main_cols]),
          "", "## 장면별", fmt(tables["by_scene"][tables["by_scene"].n_images >= 100],
                             ["method", "scene", *main_cols]),
          "", "## 위험도 영향 (기준점별)", fmt(tables["impact_overall"], ["method", "ref", *imp_cols]),
          "", "## 위험도 영향 × 시간대 (bottom)",
          fmt(tables["impact_by_time"], ["method", "timeofday", *imp_cols]),
          "", "## 위험도 영향 × 도로 형태 (bottom)",
          fmt(tables["impact_by_road_shape"], ["method", "road_shape", *imp_cols])]
    if "clahe_delta_by_time" in tables:
        md += ["", "## L1+ − L1 (CLAHE 효과)", fmt(tables["clahe_delta_by_time"],
                                                 list(tables["clahe_delta_by_time"].columns))]
    (METRICS_DIR / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:4]))


if __name__ == "__main__":
    main()
