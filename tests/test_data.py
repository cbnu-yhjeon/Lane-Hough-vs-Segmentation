import json

import numpy as np

from lanebench.data.bdd import (
    DRIVABLE_ALT,
    DRIVABLE_DIRECT,
    drivable_label,
    load_frame,
    poly2d_to_points,
    rasterize_polylines,
)
from lanebench.data.manifest import lane_curvature, road_shape


def test_bezier_expands_and_hits_endpoints():
    pts = poly2d_to_points([[0, 0, "L"], [0, 10, "C"], [10, 10, "C"], [10, 0, "C"]], 8)
    assert len(pts) == 1 + 8
    np.testing.assert_allclose(pts[0], [0, 0])
    np.testing.assert_allclose(pts[-1], [10, 0])
    assert pts[:, 1].max() > 5   # 제어점 쪽으로 휨


def test_lines_only_kept_as_is():
    pts = poly2d_to_points([[0, 0, "L"], [5, 5, "L"]])
    np.testing.assert_allclose(pts, [[0, 0], [5, 5]])


def _label(tmp_path, objects, attrs=None):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"name": "x", "attributes": attrs or {"weather": "rainy",
                             "timeofday": "night", "scene": "highway"},
                             "frames": [{"objects": objects}]}))
    return p


def test_load_frame_filters_categories(tmp_path):
    objs = [
        {"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
        {"category": "traffic sign", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
        {"category": "lane/single white", "attributes": {"direction": "parallel"},
         "poly2d": [[0, 700, "L"], [600, 400, "L"]]},
        {"category": "lane/crosswalk", "attributes": {"direction": "parallel"},
         "poly2d": [[0, 700, "L"], [600, 400, "L"]]},
        {"category": "lane/single white", "attributes": {"direction": "vertical"},
         "poly2d": [[0, 700, "L"], [600, 700, "L"]]},
        {"category": "area/drivable", "poly2d": [[0, 720, "L"], [640, 400, "L"], [1280, 720, "L"]]},
        {"category": "area/alternative", "poly2d": [[0, 720, "L"], [0, 400, "L"], [640, 400, "L"]]},
    ]
    fr = load_frame(_label(tmp_path, objs))
    assert (fr.weather, fr.timeofday, fr.scene) == ("rainy", "night", "highway")
    assert [b.category for b in fr.boxes] == ["car"]
    assert len(fr.lanes) == 1 and len(fr.direct) == 1 and len(fr.alternative) == 1

    lab = drivable_label(fr, (640, 360))
    assert lab[350, 320] == DRIVABLE_DIRECT     # 겹치면 direct 우선
    assert lab[220, 20] == DRIVABLE_ALT
    assert rasterize_polylines(fr.lanes, (640, 360), 1).sum() > 100


def test_road_shape():
    straight = [np.array([[0, 700], [300, 400]], float)]
    t = np.linspace(0, 1, 20)[:, None]
    curved = [np.hstack([300 * t + 60 * np.sin(np.pi * t), 700 - 300 * t])]
    assert road_shape(lane_curvature(straight)) == "straight"
    assert road_shape(lane_curvature(curved)) == "curved"
    assert road_shape(lane_curvature([])) == "unknown"
