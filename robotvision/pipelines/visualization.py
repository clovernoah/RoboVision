"""检测、人脸、深度与融合结果的可视化。"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import DepthResult, Detection, Face

# BGR 调色板：行人绿色、车辆橙色系、两轮车青色
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (80, 220, 120),
    "bicycle": (220, 210, 70),
    "motorcycle": (70, 200, 230),
    "car": (70, 160, 255),
    "bus": (60, 120, 240),
    "truck": (40, 90, 210),
}
DEFAULT_COLOR = (200, 180, 250)
BAND_COLORS = {
    "near": (60, 70, 255),
    "mid": (40, 190, 250),
    "far": (220, 200, 80),
}
BAND_TEXT = {"near": "NEAR", "mid": "MID", "far": "FAR"}


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """绘制带半透明底色的文字标签。"""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y = max(th + 4, y)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - th - 5), (x + tw + 6, y + baseline - 1), color, -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.putText(
        image, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    show_track: bool = True,
    depth: np.ndarray | None = None,
    show_band: bool = False,
) -> np.ndarray:
    """在原图上绘制检测框、类别、置信度、跟踪 ID 与可选深度档位。"""
    canvas = frame.copy()
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"{det.class_name} {det.score:.2f}"
        if show_track and det.track_id is not None:
            label = f"#{det.track_id} {label}"
        _label(canvas, label, (x1, y1 - 4), color)

        if show_band and depth is not None:
            from ..models.depth_estimator import DepthEstimator

            value = DepthEstimator.roi_proximity(depth, det.xyxy)
            band = DepthEstimator.proximity_band(value)
            band_color = BAND_COLORS[band]
            text = f"{BAND_TEXT[band]} {value:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            bx1 = x1
            by1 = min(y2 + th + 8, canvas.shape[0] - 2)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (bx1, by1 - th - 6), (bx1 + tw + 8, by1 + 2), band_color, -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
            cv2.putText(
                canvas, text, (bx1 + 4, by1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
    return canvas


def draw_faces(frame: np.ndarray, faces: list[Face]) -> np.ndarray:
    """绘制人脸框、置信度与 5 个关键点。"""
    canvas = frame.copy()
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = (int(round(v)) for v in face.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 230, 170), 2, cv2.LINE_AA)
        _label(canvas, f"face {index} {face.score:.2f}", (x1, y1 - 4), (60, 170, 120))
        if face.landmarks is not None:
            for px, py in face.landmarks:
                cv2.circle(canvas, (int(px), int(py)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    return canvas


def draw_hud(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 12)) -> np.ndarray:
    """在左上角绘制多行状态信息。"""
    canvas = frame.copy()
    x, y = origin
    for line in lines:
        _label(canvas, line, (x, y + 10), (40, 40, 40), 0.5, 1)
        y += 24
    return canvas


def draw_band_legend(frame: np.ndarray) -> np.ndarray:
    """在右上角绘制近/中/远图例。"""
    canvas = frame.copy()
    x = canvas.shape[1] - 118
    y = 14
    for band in ("near", "mid", "far"):
        color = BAND_COLORS[band]
        cv2.rectangle(canvas, (x, y), (x + 16, y + 16), color, -1)
        cv2.putText(
            canvas, BAND_TEXT[band], (x + 22, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (240, 240, 240), 1, cv2.LINE_AA,
        )
        y += 22
    return canvas


def depth_panel(
    original: np.ndarray,
    depth_result: DepthResult,
    titles: tuple[str, str] = ("RGB", "Depth (MiDaS)"),
) -> np.ndarray:
    """生成“原图 | 深度伪彩色”对比图，并附带深度色条。"""
    height = original.shape[0]
    aspect = original.shape[1] / max(1, original.shape[0])
    width = max(160, int(height * aspect))
    left = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(depth_result.color, (width, height), interpolation=cv2.INTER_AREA)

    bar_w = 58
    canvas = np.zeros((height + 34, width * 2 + bar_w, 3), dtype=np.uint8)
    canvas[:] = (24, 26, 30)
    canvas[34:34 + height, :width] = left
    canvas[34:34 + height, width:width * 2] = right

    cv2.putText(canvas, titles[0], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, titles[1], (width + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    # 右侧竖直色条：上近下远
    bx = width * 2 + 12
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    gradient = np.repeat(gradient, 14, axis=1)
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)
    canvas[34:34 + height, bx:bx + 14] = bar
    cv2.putText(canvas, "near", (bx + 18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "far", (bx + 18, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    """将两张图像缩放到同一高度后横向拼接。"""
    height = max(left.shape[0], right.shape[0])
    lw = int(left.shape[1] * height / left.shape[0])
    rw = int(right.shape[1] * height / right.shape[0])
    left_r = cv2.resize(left, (lw, height), interpolation=cv2.INTER_AREA)
    right_r = cv2.resize(right, (rw, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, lw + rw + gap, 3), 24, dtype=np.uint8)
    canvas[:, :lw] = left_r
    canvas[:, lw + gap:] = right_r
    return canvas
