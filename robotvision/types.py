"""跨模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    """一个目标检测结果（xyxy 像素坐标）。"""

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    track_id: int | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyxy": [round(float(v), 2) for v in self.xyxy],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "track_id": None if self.track_id is None else int(self.track_id),
        }


@dataclass
class Face:
    """人脸检测结果，landmarks 为 5×2 的关键点像素坐标。"""

    xyxy: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyxy": [round(float(v), 2) for v in self.xyxy],
            "score": round(float(self.score), 4),
            "landmarks": None
            if self.landmarks is None
            else [[round(float(x), 2), round(float(y), 2)] for x, y in self.landmarks],
        }


@dataclass
class DepthResult:
    """深度估计结果。

    depth: 归一化相对接近度，取值 [0, 1]，越大表示越近。
    raw: 网络原始逆深度输出。
    color: 伪彩色可视化（BGR）。
    """

    depth: np.ndarray
    raw: np.ndarray
    color: np.ndarray
    min_val: float = 0.0
    max_val: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> dict[str, float]:
        return {
            "min": round(float(self.min_val), 4),
            "max": round(float(self.max_val), 4),
            "mean": round(float(self.depth.mean()), 4),
        }
