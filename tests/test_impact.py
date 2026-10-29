import numpy as np
import pytest
import torch

from lanebench.data.bdd import Box
from lanebench.impact.in_lane import in_lane, object_rows, query_mask
from lanebench.impact.risk_v2 import CAUTION, risk_score
from lanebench.segmentation.infer import decode, encode
from lanebench.segmentation.model import LaneSegNet


def test_lane_weight_is_the_only_difference():
    s_in, _ = risk_score(500, 300, 780, 500, 1280, 720, 0.8, True)
    s_out, _ = risk_score(500, 300, 780, 500, 1280, 720, 0.8, False)
    assert s_in / s_out == pytest.approx(1.1 / 0.85)


def test_risk_level_thresholds():
    # 화면 중앙의 큰 박스 → 점수 높음
    _, lvl = risk_score(440, 200, 840, 600, 1280, 720, 1.0, True)
    assert lvl >= CAUTION


def test_in_lane_empty_mask_fallback():
    empty = np.zeros((360, 640), np.uint8)
    assert in_lane(empty, 640, 600, empty_is_in=True)
    assert not in_lane(empty, 640, 600, empty_is_in=False)


def test_object_rows_disagreement():
    gt = np.zeros((360, 640), np.uint8)
    gt[200:, 250:390] = 1
    pred = np.zeros_like(gt)
    pred[200:, 0:100] = 1
    rows = object_rows([Box("car", 600, 400, 680, 500)], query_mask(gt), query_mask(pred), 0.8)
    bottom = next(r for r in rows if r["ref"] == "bottom")
    assert bottom["gt_in"] and not bottom["pred_in"]
    assert bottom["gt_score"] > bottom["pred_score"]


def test_png_encoding_roundtrip():
    d = np.array([[0, 1], [2, 1]], np.uint8)
    lane = np.array([[1, 0], [0, 1]], np.uint8)
    d2, l2 = decode(encode(d, lane))
    assert (d2 == d).all() and (l2 == lane).all()


def test_model_output_shapes():
    m = LaneSegNet(pretrained=False).eval()
    with torch.no_grad():
        d, lane = m(torch.rand(1, 3, 360, 640))
    assert d.shape == (1, 3, 360, 640) and lane.shape == (1, 1, 360, 640)
