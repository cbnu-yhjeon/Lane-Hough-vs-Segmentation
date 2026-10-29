"""
L2 추론 결과를 PNG로 저장 (평가 스크립트가 다른 방법과 같은 방식으로 읽음)

    python -m lanebench.segmentation.infer --split val

PNG 한 장에 두 출력을 담습니다: 값 = 주행가능영역 클래스(0~2) + 4 × 차선(0/1)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lanebench.data.bdd import DRIVABLE_DIRECT
from lanebench.paths import CKPT_DIR, MANIFEST, RAW_DIR
from lanebench.prediction import EVAL_SIZE, Prediction
from lanebench.segmentation.dataset import read_image, to_tensor
from lanebench.segmentation.model import LaneSegNet

PRED_DIR = RAW_DIR / "l2_pred"
LANE_THRESHOLD = 0.5


def load_model(ckpt: str | Path, device) -> LaneSegNet:
    model = LaneSegNet(pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_batch(model, img: torch.Tensor):
    """(B,3,H,W) → drivable class (B,H,W) uint8, lane (B,H,W) uint8."""
    use_amp = img.is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        d_logits, l_logits = model(img)
    drivable = d_logits.argmax(1).to(torch.uint8)
    lane = (l_logits[:, 0].float().sigmoid() > LANE_THRESHOLD).to(torch.uint8)
    return drivable, lane


def encode(drivable: np.ndarray, lane: np.ndarray) -> np.ndarray:
    return (drivable + 4 * lane).astype(np.uint8)


def decode(png: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return png & 3, (png >> 2) & 1


class CachedL2:
    """저장된 L2 예측을 다른 방법과 같은 인터페이스로 돌려줍니다."""
    name = "L2"

    def __init__(self, pred_dir: Path = PRED_DIR):
        self.pred_dir = Path(pred_dir)

    def load(self, image_name: str) -> tuple[Prediction, np.ndarray]:
        png = cv2.imread(str(self.pred_dir / f"{image_name}.png"), cv2.IMREAD_UNCHANGED)
        drivable, lane = decode(png)
        ego = (drivable == DRIVABLE_DIRECT).astype(np.uint8)
        return Prediction(ego, lane.astype(np.uint8)), drivable


class _Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, path = self.rows[i]
        return to_tensor(read_image(path, EVAL_SIZE)), name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "l2_lanesegnet.pt"))
    ap.add_argument("--out", default=str(PRED_DIR))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device).to(memory_format=torch.channels_last)
    df = pd.read_csv(MANIFEST)
    rows = df[df.split == args.split][["name", "image"]].to_numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(_Images(rows), batch_size=args.batch, num_workers=8)
    for img, names in tqdm(loader, desc="L2 infer"):
        img = img.to(device).to(memory_format=torch.channels_last)
        drivable, lane = predict_batch(model, img)
        for name, d, l in zip(names, drivable.cpu().numpy(), lane.cpu().numpy()):
            cv2.imwrite(str(out / f"{name}.png"), encode(d, l))


if __name__ == "__main__":
    main()
