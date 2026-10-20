"""深度融合与可视化测试。"""

import numpy as np

from robotvision.fusion.navigation import analyze_scene, risk_grid
from robotvision.pipelines.visualization import depth_panel, draw_detections, draw_faces
from robotvision.types import DepthResult, Detection, Face


def _depth_result() -> DepthResult:
    depth = np.full((100, 100), 0.1, dtype=np.float32)
    depth[60:100, 40:60] = 0.9
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    return DepthResult(
        depth=depth, raw=depth.copy(), color=color,
        min_val=float(depth.min()), max_val=float(depth.max()),
    )


def test_risk_grid_shape_and_range():
    grid = risk_grid(_depth_result().depth, rows=3, cols=3)
    assert grid.shape == (3, 3)
    assert grid.min() >= 0.0 and grid.max() <= 1.0
    assert grid[2, 1] > grid[0, 0]


def test_scene_reports_high_risk_and_suggestion():
    det = Detection((35, 50, 65, 95), 0.9, 0, "person")
    result = analyze_scene([det], _depth_result())
    assert result["objects"][0]["band"] == "near"
    assert result["objects"][0]["risk"] == "high"
    assert result["objects"][0]["direction"] == "center"
    assert result["suggestion"]["action"] in {"stop", "turn_left", "turn_right"}
    assert result["counts"]["person"] == 1


def test_scene_without_detections_returns_forward_or_slow():
    result = analyze_scene([], _depth_result())
    assert result["objects"] == []
    assert result["suggestion"]["action"] in {"forward", "slow_down"}
    assert result["depth_stats"]["min"] == 0.1


def test_visualization_shapes():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    det = Detection((10, 20, 40, 60), 0.8, 0, "car")
    annotated = draw_detections(frame, [det], depth=_depth_result().depth, show_band=True)
    assert annotated.shape == frame.shape
    faces = [Face((10, 10, 30, 30), 0.9, np.array([[15, 15]] * 5, dtype=np.float32))]
    face_image = draw_faces(frame, faces)
    assert face_image.shape == frame.shape
    panel = depth_panel(frame, _depth_result())
    assert panel.ndim == 3
    assert panel.shape[0] == frame.shape[0] + 34
