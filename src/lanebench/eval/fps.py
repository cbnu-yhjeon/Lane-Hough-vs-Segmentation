"""
속도 측정 (results/metrics/fps.csv)

    python -m lanebench.eval.fps --n 200

고전 기법은 CPU 단일 스레드, L2는 GPU(batch 1, bf16)와 CPU(4 스레드, fp32)에서
잽니다. 영상 디코딩 시간은 빼고 predict 호출만 잽니다.
"""
from __future__ import annotations

import argparse
import time

import cv2
import pandas as pd
import torch

from lanebench.classical.hough_v2 import HoughV2
from lanebench.classical.ipm_poly import make_l1, make_l1_plus
from lanebench.paths import CKPT_DIR, MANIFEST, METRICS_DIR
from lanebench.prediction import EVAL_SIZE
from lanebench.segmentation.dataset import to_tensor
from lanebench.segmentation.infer import load_model, predict_batch


def bench(fn, items, warmup=5) -> float:
    for x in items[:warmup]:
        fn(x)
    t = time.perf_counter()
    for x in items:
        fn(x)
    return len(items) / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "l2_lanesegnet.pt"))
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST)
    paths = df[df.split == "val"].image.iloc[: args.n].tolist()
    imgs = [cv2.imread(p) for p in paths]
    rows = []

    cv2.setNumThreads(1)
    for m in (HoughV2(), make_l1(), make_l1_plus()):
        rows.append({"method": m.name, "device": "CPU 1 thread", "input": "1280x720",
                     "fps": bench(lambda im, m=m: m.predict(im), imgs)})

    small = [to_tensor(cv2.resize(im, EVAL_SIZE, interpolation=cv2.INTER_AREA))[None]
             for im in imgs]
    if torch.cuda.is_available():
        model = load_model(args.ckpt, "cuda").to(memory_format=torch.channels_last)

        def gpu(x):
            d, l = predict_batch(model, x.cuda().to(memory_format=torch.channels_last))
            d.cpu(), l.cpu()
        rows.append({"method": "L2", "device": torch.cuda.get_device_name(0), "input": "640x360",
                     "fps": bench(gpu, small, warmup=20)})
    torch.set_num_threads(4)
    model = load_model(args.ckpt, "cpu")
    rows.append({"method": "L2", "device": "CPU 4 threads", "input": "640x360",
                 "fps": bench(lambda x: predict_batch(model, x), small[:50])})

    out = pd.DataFrame(rows)
    out["ms"] = 1000 / out.fps
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(METRICS_DIR / "fps.csv", index=False)
    print(out.round(2).to_markdown(index=False))


if __name__ == "__main__":
    main()
