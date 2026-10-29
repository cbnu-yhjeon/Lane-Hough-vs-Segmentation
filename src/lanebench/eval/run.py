"""
방법별 평가 실행 → 이미지 단위 · 객체 단위 원자료 CSV

    python -m lanebench.eval.run --methods L0 L1 L1+    # 고전 기법 (CPU 병렬)
    python -m lanebench.eval.run --methods L2           # 저장된 L2 예측 사용

이미지 행에는 IoU · F1을 조건별로 다시 합산할 수 있도록 픽셀 수를 남깁니다.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm

from lanebench.classical.hough_v2 import HoughV2, compute_visibility_score
from lanebench.classical.ipm_poly import make_l1, make_l1_plus
from lanebench.data.bdd import drivable_label, load_frame, rasterize_polygons, rasterize_polylines
from lanebench.eval.lane_f1 import lane_counts
from lanebench.eval.miou import confusion, iou_counts
from lanebench.impact.in_lane import object_rows, query_mask
from lanebench.paths import MANIFEST, RAW_DIR
from lanebench.prediction import EVAL_SIZE

CLASSICAL = {"L0": HoughV2, "L1": make_l1, "L1+": make_l1_plus}
_methods: dict = {}


def _init(names: list[str]):
    cv2.setNumThreads(1)
    for n in names:
        if n == "L2":
            from lanebench.segmentation.infer import CachedL2
            _methods[n] = CachedL2()
        else:
            _methods[n] = CLASSICAL[n]()


def evaluate_image(row: dict) -> tuple[list[dict], list[dict]]:
    fr = load_frame(row["label"])
    img = cv2.imread(row["image"]) if any(n != "L2" for n in _methods) else None
    gray = cv2.cvtColor(img if img is not None else cv2.imread(row["image"]), cv2.COLOR_BGR2GRAY)
    visibility = compute_visibility_score(gray)

    gt_ego = rasterize_polygons(fr.direct, EVAL_SIZE)
    gt_lane = rasterize_polylines(fr.lanes, EVAL_SIZE, 1)
    gt_q = query_mask(gt_ego)
    has_direct = bool(fr.direct)

    img_rows, obj_rows = [], []
    for name, method in _methods.items():
        t = time.perf_counter()
        if name == "L2":
            pred, drivable = method.load(fr.name)
        else:
            pred = method.predict(img, EVAL_SIZE)
        ms = (time.perf_counter() - t) * 1000

        inter, union = iou_counts(pred.ego_mask, gt_ego)
        r = dict(name=fr.name, method=name, has_direct=has_direct, detected=pred.detected,
                 ego_inter=inter, ego_union=union, ego_gt=int(gt_ego.sum()),
                 ego_pred=int(pred.ego_mask.sum()), n_pixels=gt_ego.size,
                 visibility=round(visibility, 4), ms=round(ms, 2),
                 **lane_counts(pred.lane_mask, gt_lane))
        if name == "L2":
            cm = confusion(drivable, drivable_label(fr, EVAL_SIZE), 3)
            r.update({f"cm{i}{j}": int(cm[i, j]) for i in range(3) for j in range(3)})
        else:
            r.update(vp_estimated=pred.info.get("vp_estimated"),
                     extrapolated=pred.info.get("extrapolated"))
        img_rows.append(r)

        if has_direct and fr.boxes:
            for o in object_rows(fr.boxes, gt_q, query_mask(pred.ego_mask), visibility):
                obj_rows.append(dict(name=fr.name, method=name, **o))
    return img_rows, obj_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["L0", "L1", "L1+"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default="", help="출력 파일 이름 접미사")
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST)
    rows = df[df.split == args.split].to_dict("records")
    if args.limit:
        rows = rows[:args.limit]

    img_rows, obj_rows = [], []
    with Pool(args.workers, initializer=_init, initargs=(args.methods,)) as pool:
        for a, b in tqdm(pool.imap_unordered(evaluate_image, rows, chunksize=16), total=len(rows)):
            img_rows.extend(a)
            obj_rows.extend(b)

    tag = args.tag or "_".join(m.replace("+", "plus") for m in args.methods)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(img_rows).to_csv(RAW_DIR / f"per_image_{tag}.csv", index=False)
    pd.DataFrame(obj_rows).to_csv(RAW_DIR / f"per_object_{tag}.csv.gz", index=False)
    print(f"{len(img_rows)} image rows, {len(obj_rows)} object rows → {RAW_DIR}")


if __name__ == "__main__":
    main()
