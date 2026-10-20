"""通用工具：NMS、IoU、图像编解码、视频源与计时。"""

from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .types import Detection


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """NumPy 实现的非极大值抑制。

    Args:
        boxes: (N, 4) xyxy。
        scores: (N,) 置信度。
        iou_threshold: IoU 阈值。

    Returns:
        保留下来的索引列表（按分数降序）。
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_threshold]
    return keep


def nms_detections(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    """按类别分别执行 NMS。"""
    if not detections:
        return []
    kept: list[Detection] = []
    for class_id in {d.class_id for d in detections}:
        subset = [d for d in detections if d.class_id == class_id]
        boxes = np.array([d.xyxy for d in subset], dtype=np.float32)
        scores = np.array([d.score for d in subset], dtype=np.float32)
        for idx in nms(boxes, scores, iou_threshold):
            kept.append(subset[idx])
    return sorted(kept, key=lambda d: d.score, reverse=True)


def encode_image_bgr(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> bytes:
    """将 BGR 图像编码为字节流。"""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"图像编码失败: {ext}")
    return buffer.tobytes()


def encode_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """将 BGR 图像编码为 data URL，供前端直接展示。"""
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(encode_image_bgr(image, ext)).decode("ascii")
    return f"data:{mime};base64,{payload}"


def decode_image(data: bytes) -> np.ndarray:
    """从字节流解码 BGR 图像。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码图像，请检查文件格式")
    return image


def parse_source(source: str | int) -> str | int:
    """将命令行 source 解析为摄像头索引或路径。"""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


def open_video_source(source: str | int) -> cv2.VideoCapture:
    """打开摄像头或视频文件，并尽量设置合理的采集参数。"""
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@contextmanager
def timer() -> Iterator[callable]:
    """计时上下文：with timer() as elapsed: ... ; elapsed() 返回毫秒。"""
    start = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - start) * 1000.0

    yield elapsed


def sliding_average(values: list[float], window: int) -> list[float]:
    """简单滑动平均，长度与输入一致。"""
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    acc = 0.0
    for i, value in enumerate(values):
        acc += value
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out
