"""检测 + 深度融合，为机器人导航提供可解释的风险提示。

说明：MiDaS 输出的是**相对深度**，在没有相机标定的情况下不能直接换算为米。
因此这里使用归一化接近度与三档相对距离（近/中/远），并把结果作为
风险提示而非精确测距。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.depth_estimator import DepthEstimator
from ..types import DepthResult, Detection

BAND_LABELS = {"near": "近", "mid": "中", "far": "远"}
DIRECTION_LABELS = {"left": "左侧", "center": "正前方", "right": "右侧"}
ACTION_LABELS = {
    "forward": "保持前进",
    "slow_down": "减速观察",
    "turn_left": "向左避让",
    "turn_right": "向右避让",
    "stop": "立即停止",
}


def risk_grid(depth: np.ndarray, rows: int = 3, cols: int = 3) -> np.ndarray:
    """把深度图划分为网格，返回每个格子的平均接近度。"""
    height, width = depth.shape[:2]
    grid = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y1, y2 = int(height * r / rows), int(height * (r + 1) / rows)
        for c in range(cols):
            x1, x2 = int(width * c / cols), int(width * (c + 1) / cols)
            cell = depth[y1:y2, x1:x2]
            grid[r, c] = float(cell.mean()) if cell.size else 0.0
    return grid


def _direction(center_x: float, width: int) -> str:
    ratio = center_x / max(1, width)
    if ratio < 0.38:
        return "left"
    if ratio > 0.62:
        return "right"
    return "center"


def analyze_scene(
    detections: list[Detection],
    depth_result: DepthResult,
    high_risk_classes: tuple[str, ...] = ("person", "car", "bus", "truck", "motorcycle"),
) -> dict[str, Any]:
    """对检测目标做深度关联，并给出场景级风险与动作建议。

    Args:
        detections: 当前帧检测结果。
        depth_result: 当前帧深度结果。
        high_risk_classes: 进入近距区域时判为高风险的目标类别。

    Returns:
        可 JSON 序列化的融合结果。
    """
    depth = depth_result.depth
    height, width = depth.shape[:2]

    fused: list[dict[str, Any]] = []
    for det in detections:
        proximity = DepthEstimator.roi_proximity(depth, det.xyxy)
        band = DepthEstimator.proximity_band(proximity)
        direction = _direction(det.center[0], width)
        is_dynamic = det.class_name in high_risk_classes
        if band == "near" and is_dynamic:
            risk = "high"
        elif band == "near" or (band == "mid" and is_dynamic):
            risk = "medium"
        else:
            risk = "low"
        fused.append(
            {
                **det.to_dict(),
                "proximity": round(proximity, 4),
                "band": band,
                "band_label": BAND_LABELS[band],
                "direction": direction,
                "direction_label": DIRECTION_LABELS[direction],
                "risk": risk,
            }
        )

    grid = risk_grid(depth)
    # 只看靠近图像下半部分的格子：上方多为天空/远处背景
    ground = grid[1:, :]
    column_scores = ground.mean(axis=0)
    center_score = float(column_scores[1])
    left_score = float(column_scores[0])
    right_score = float(column_scores[2])

    center_high = [item for item in fused if item["risk"] == "high" and item["direction"] == "center"]
    any_high = [item for item in fused if item["risk"] == "high"]

    if center_high:
        if left_score + 0.08 < right_score:
            action, reason = "turn_left", "正前方存在近距离目标，左侧空间更开阔"
        elif right_score + 0.08 < left_score:
            action, reason = "turn_right", "正前方存在近距离目标，右侧空间更开阔"
        else:
            action, reason = "stop", "正前方近距离目标且两侧空间不足"
    elif any_high:
        action, reason = "slow_down", "侧前方存在近距离动态目标，建议减速观察"
    elif center_score > 0.68:
        action, reason = "slow_down", "正前方整体接近度较高，可能存在未识别障碍"
    else:
        action, reason = "forward", "前方相对开阔，可保持前进"

    return {
        "objects": fused,
        "grid": [[round(float(v), 4) for v in row] for row in grid],
        "corridor": {
            "left": round(left_score, 4),
            "center": round(center_score, 4),
            "right": round(right_score, 4),
        },
        "suggestion": {
            "action": action,
            "action_label": ACTION_LABELS[action],
            "reason": reason,
        },
        "counts": {
            "person": sum(1 for d in detections if d.class_name == "person"),
            "vehicle": sum(
                1 for d in detections if d.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}
            ),
            "face": 0,
        },
        "depth_stats": depth_result.stats(),
    }
