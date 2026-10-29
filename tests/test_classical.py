"""합성 도로 영상: 소실점으로 모이는 흰 차선 두 줄 + 회색 노면."""
import cv2
import numpy as np
import pytest

from lanebench.classical.hough_v2 import HoughV2
from lanebench.classical.ipm_poly import DEFAULT_VP, IPMConfig, IPMPoly, make_l1
from lanebench.eval.miou import iou_counts
from lanebench.prediction import EVAL_SIZE

W, H = 1280, 720
VP = (DEFAULT_VP[0] * W, DEFAULT_VP[1] * H)
LEFT_BOTTOM, RIGHT_BOTTOM = 330, 900


def _x_at(x_bottom, y):
    return VP[0] + (x_bottom - VP[0]) * (y - VP[1]) / (H - VP[1])


@pytest.fixture(scope="module")
def road():
    img = np.full((H, W, 3), 70, np.uint8)
    img[: int(VP[1])] = (200, 170, 140)   # 하늘
    y_top = int(VP[1] + 25)
    for xb in (LEFT_BOTTOM, RIGHT_BOTTOM):
        cv2.line(img, (int(_x_at(xb, y_top)), y_top), (xb, H), (235, 235, 235), 12)
    gt = np.zeros((H, W), np.uint8)
    y0 = int(0.6 * H)
    cv2.fillPoly(gt, [np.int32([[_x_at(LEFT_BOTTOM, y0), y0], [_x_at(RIGHT_BOTTOM, y0), y0],
                                [_x_at(RIGHT_BOTTOM, 0.9 * H), 0.9 * H],
                                [_x_at(LEFT_BOTTOM, 0.9 * H), 0.9 * H]])], 1)
    return img, cv2.resize(gt, EVAL_SIZE, interpolation=cv2.INTER_NEAREST)


def test_hough_v2_finds_ego_lane(road):
    img, gt = road
    pred = HoughV2().predict(img)
    assert pred.detected
    inter, union = iou_counts(pred.ego_mask, gt)
    assert inter / union > 0.6


def test_ipm_poly_finds_ego_lane(road):
    img, gt = road
    pred = make_l1().predict(img)
    assert pred.detected and pred.info["extrapolated"] is None
    inter, union = iou_counts(pred.ego_mask, gt)
    assert inter / union > 0.6
    assert pred.lane_mask.any()


def test_ipm_roundtrip():
    m = IPMPoly(IPMConfig())
    M, Minv = m.homography(VP, W, H)
    pts = np.float32([[[500, 600], [700, 650]]])
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(pts, M), Minv)
    np.testing.assert_allclose(back, pts, atol=1e-2)


def test_single_side_is_extrapolated_by_lane_width():
    m = make_l1()
    fits, extra = m.complete_pair({"left": np.array([0.0, 0.0, 100.0])}, {"left": 500})
    assert extra == "right"
    assert fits["right"][2] == pytest.approx(100.0 + m.cfg.lane_width_bev)


def test_blank_image_gives_empty_mask():
    pred = make_l1().predict(np.full((H, W, 3), 90, np.uint8))
    assert not pred.detected
