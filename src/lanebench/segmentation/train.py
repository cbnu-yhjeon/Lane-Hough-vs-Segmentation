"""
L2 학습

    python -m lanebench.segmentation.train --epochs 4

학습 셋에서 minival(기본 1000장)을 떼어 학습 경과만 봅니다. BDD100K val은
최종 평가에만 쓰고, 체크포인트는 마지막 epoch을 그대로 씁니다 (val로 고르지 않음).
"""
from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lanebench.eval.miou import confusion, miou_from_confusion
from lanebench.paths import CKPT_DIR, MANIFEST, METRICS_DIR
from lanebench.segmentation.dataset import BDDLaneDataset
from lanebench.segmentation.model import LaneSegNet

LANE_POS_WEIGHT = 4.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def compute_loss(out, drivable, lane):
    d_logits, l_logits = out
    l_ce = F.cross_entropy(d_logits.float(), drivable)
    l_bce = F.binary_cross_entropy_with_logits(
        l_logits.float(), lane, pos_weight=torch.tensor(LANE_POS_WEIGHT, device=lane.device))
    l_dice = dice_loss(l_logits.float(), lane)
    return l_ce + l_bce + l_dice, (l_ce.item(), l_bce.item(), l_dice.item())


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((3, 3), np.int64)
    lane_i = lane_u = 0
    for img, drivable, lane in loader:
        img = img.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d_logits, l_logits = model(img)
        pred = d_logits.argmax(1).cpu().numpy()
        cm += confusion(pred, drivable.numpy(), 3)
        lp = (l_logits.float().sigmoid() > 0.5).cpu()
        lane_i += int((lp & (lane > 0.5)).sum())
        lane_u += int((lp | (lane > 0.5)).sum())
    model.train()
    miou, ious = miou_from_confusion(cm)
    return {"miou3": miou, "iou_direct": ious[1], "iou_alt": ious[2],
            "lane_iou": lane_i / max(lane_u, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="학습 영상 수 제한 (0 = 전체)")
    ap.add_argument("--minival", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(CKPT_DIR / "l2_lanesegnet.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(MANIFEST)
    train_df = df[df.split == "train"].sample(frac=1.0, random_state=args.seed)
    minival_df, train_df = train_df.iloc[:args.minival], train_df.iloc[args.minival:]
    if args.limit:
        train_df = train_df.iloc[:args.limit]
    print(f"train {len(train_df)}  minival {len(minival_df)}")

    train_loader = DataLoader(BDDLaneDataset(train_df, train=True), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=True,
                              worker_init_fn=lambda w: np.random.seed(args.seed * 100 + w))
    val_loader = DataLoader(BDDLaneDataset(minival_df, train=False), batch_size=32,
                            num_workers=args.workers, pin_memory=True)

    model = LaneSegNet(pretrained=True).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_loader)
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = METRICS_DIR / "l2_train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "ce", "bce", "dice", "miou3",
                                "iou_direct", "iou_alt", "lane_iou", "minutes"])
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        sums = np.zeros(4)
        for i, (img, drivable, lane) in enumerate(train_loader):
            img = img.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            drivable = drivable.to(device, non_blocking=True)
            lane = lane.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(img)
            loss, parts = compute_loss(out, drivable, lane)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            sums += [loss.item(), *parts]
            if i % 200 == 0:
                print(f"ep {epoch} it {i}/{len(train_loader)} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f}min",
                      flush=True)
        m = evaluate(model, val_loader, device)
        avg = sums / len(train_loader)
        print(f"== epoch {epoch} loss {avg[0]:.4f} minival {m}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, *np.round(avg, 5), *(round(v, 4) for v in m.values()),
                                    round((time.time() - t0) / 60, 1)])
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
