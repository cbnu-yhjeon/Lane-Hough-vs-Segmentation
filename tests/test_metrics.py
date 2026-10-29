import numpy as np
import pytest

from lanebench.eval.lane_f1 import lane_counts, prf
from lanebench.eval.miou import confusion, ego_bg_miou, iou_counts, miou_from_confusion


def test_iou_counts():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros_like(a)
    a[:5] = 1
    b[:, :5] = 1
    assert iou_counts(a, b) == (25, 75)


def test_miou_from_confusion_perfect_and_ego_bg():
    lab = np.array([[0, 1], [2, 1]])
    miou, ious = miou_from_confusion(confusion(lab, lab, 3))
    assert miou == 1.0 and np.all(ious == 1)
    assert ego_bg_miou(50, 100, 1000) == pytest.approx((0.5 + 900 / 950) / 2)


def test_lane_f1_tolerance():
    gt = np.zeros((50, 50), np.uint8)
    gt[:, 20] = 1
    pred = np.zeros_like(gt)
    pred[:, 23] = 1                       # 3 px 옆
    c = lane_counts(pred, gt, tolerances=(2, 5))
    assert prf(c["tp_pred@2"], c["n_pred"], c["tp_gt@2"], c["n_gt"])[2] == 0
    assert prf(c["tp_pred@5"], c["n_pred"], c["tp_gt@5"], c["n_gt"])[2] == 1


def test_lane_f1_empty_prediction():
    gt = np.zeros((20, 20), np.uint8)
    gt[5] = 1
    c = lane_counts(np.zeros_like(gt), gt)
    assert c["n_pred"] == 0 and c["tp_gt@5"] == 0
    assert prf(c["tp_pred@5"], c["n_pred"], c["tp_gt@5"], c["n_gt"]) == (0.0, 0.0, 0.0)
