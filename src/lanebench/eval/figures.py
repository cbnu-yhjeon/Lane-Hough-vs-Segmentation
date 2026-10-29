"""
보고서 그림 (results/figures/*.png)

    python -m lanebench.eval.figures

색: 방법 = 범주형 고정 순서(파랑 · 주황 · 청록 · 노랑), 히트맵 = 파랑 단일 색조 순차,
차이(L1+ − L1) = 주황 ↔ 회색 ↔ 파랑 발산.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from lanebench.eval.by_condition import METHODS, TAU, TIME_ORDER, WEATHER_ORDER
from lanebench.paths import FIG_DIR, METRICS_DIR

METHOD_COLOR = {"L0": "#2a78d6", "L1": "#eb6834", "L1+": "#1baf7a", "L2": "#eda100"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#d95926", "#f0efec", "#2a78d6"])
LABEL = {"L0": "L0 Hough v2", "L1": "L1 IPM+poly", "L1+": "L1+ CLAHE", "L2": "L2 Seg"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.grid": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _methods(df):
    return [m for m in METHODS if m in set(df.method)]


def heatmaps(metric: str, title: str, fname: str):
    t = pd.read_csv(METRICS_DIR / "by_weather_time.csv")
    methods = _methods(t)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 3.4),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        d = t[t.method == m]
        grid = d.pivot(index="weather_g", columns="timeofday", values=metric)
        cnt = d.pivot(index="weather_g", columns="timeofday", values="n_images")
        grid = grid.reindex(index=WEATHER_ORDER, columns=TIME_ORDER)
        cnt = cnt.reindex(index=WEATHER_ORDER, columns=TIME_ORDER)
        # 표본이 너무 적은 칸은 비움
        grid = grid.where(cnt >= 30)
        im = ax.imshow(grid.values, cmap=SEQ, vmin=0, vmax=max(0.8, np.nanmax(t[metric])))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v, n = grid.values[i, j], cnt.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n<30", ha="center", va="center", color=MUTED, fontsize=8)
                    continue
                dark = v > 0.45 * max(0.8, np.nanmax(t[metric]))
                ax.text(j, i, f"{v:.2f}\n(n={int(n)})", ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark else TEXT)
        ax.set_xticks(range(len(TIME_ORDER)), ["day", "dusk", "night"])
        ax.set_yticks(range(len(WEATHER_ORDER)), WEATHER_ORDER)
        ax.set_title(LABEL[m])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label=title)
    fig.suptitle(f"{title} — weather × time of day", color=TEXT)
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def grouped_bars(table: str, key: str, order: list[str], metric: str, ylabel: str, fname: str):
    t = pd.read_csv(METRICS_DIR / f"{table}.csv")
    methods = _methods(t)
    order = [o for o in order if o in set(t[key])]
    x = np.arange(len(order))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(order), 3.4), constrained_layout=True)
    for k, m in enumerate(methods):
        d = t[t.method == m].set_index(key).reindex(order)
        bars = ax.bar(x + (k - (len(methods) - 1) / 2) * width, d[metric], width * 0.92,
                      color=METHOD_COLOR[m], label=LABEL[m], edgecolor=SURFACE, linewidth=1)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(x, order)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def impact_bars():
    t = pd.read_csv(METRICS_DIR / "impact_overall.csv")
    t = t[t.ref == "bottom"]
    methods = _methods(t)
    cols = [("false_in", "out→in\n(over-warn side)"), ("false_out", "in→out\n(under-warn side)"),
            ("level_changed", "risk level\nchanged"), ("danger_missed", "GT DANGER\ndowngraded")]
    x = np.arange(len(cols))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    for k, m in enumerate(methods):
        d = t[t.method == m].iloc[0]
        bars = ax.bar(x + (k - (len(methods) - 1) / 2) * width, [100 * d[c] for c, _ in cols],
                      width * 0.92, color=METHOD_COLOR[m], label=LABEL[m], edgecolor=SURFACE)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(x, [c for _, c in cols])
    ax.set_ylabel("% of GT objects\n(last group: % of GT-DANGER)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(FIG_DIR / "impact_overall.png", dpi=150)
    plt.close(fig)


def clahe_delta():
    p = METRICS_DIR / "clahe_delta_by_time.csv"
    if not p.exists():
        return
    t = pd.read_csv(p).set_index("timeofday").reindex(TIME_ORDER)
    cols = ["ego_iou", "detect_rate", f"lane_f1@{TAU}"]
    v = t[cols].values
    lim = max(0.02, np.nanmax(np.abs(v)))
    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)
    im = ax.imshow(v, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:+.3f}", ha="center", va="center", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(cols)), ["ego IoU", "detect rate", f"lane F1@{TAU}"])
    ax.set_yticks(range(len(TIME_ORDER)), TIME_ORDER)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.9, label="L1+ − L1")
    ax.set_title("CLAHE effect (L1+ − L1)")
    fig.savefig(FIG_DIR / "clahe_delta.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    heatmaps("ego_iou", "Ego-lane IoU", "heatmap_ego_iou.png")
    heatmaps(f"lane_f1@{TAU}", f"Lane F1 @{TAU}px", "heatmap_lane_f1.png")
    grouped_bars("by_road_shape", "road_shape", ["straight", "curved", "unknown"], "ego_iou",
                 "Ego-lane IoU", "road_shape_ego_iou.png")
    grouped_bars("by_time", "timeofday", TIME_ORDER, "ego_iou", "Ego-lane IoU",
                 "time_ego_iou.png")
    grouped_bars("by_time", "timeofday", TIME_ORDER, "detect_rate", "Detection rate",
                 "time_detect_rate.png")
    impact_bars()
    clahe_delta()
    print("figures →", FIG_DIR)


if __name__ == "__main__":
    main()
