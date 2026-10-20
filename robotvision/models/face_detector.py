"""基于 OpenCV YuNet 的人脸检测器。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import Face


class FaceDetector:
    """YuNet 轻量人脸检测，输出人脸框、置信度与 5 个关键点。"""

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到 YuNet 人脸模型：{self.model_path}\n"
                "请先运行：python scripts/download_models.py"
            )
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("当前 OpenCV 版本不支持 FaceDetectorYN，请升级 opencv-python>=4.8")

        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        """检测 BGR 图像中的人脸。"""
        height, width = frame.shape[:2]
        size = (int(width), int(height))
        if self._input_size != size:
            self.detector.setInputSize(size)
            self._input_size = size

        _, raw_faces = self.detector.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        faces: list[Face] = []
        for row in raw_faces:
            x, y, w, h = (float(v) for v in row[:4])
            # YuNet 输出顺序：box(4) + 5 个关键点(10) + score(1)
            landmarks = row[4:14].reshape(5, 2).astype(np.float32)
            score = float(row[14])
            faces.append(
                Face(
                    xyxy=(x, y, x + w, y + h),
                    score=score,
                    landmarks=landmarks,
                )
            )
        return sorted(faces, key=lambda f: f.score, reverse=True)
