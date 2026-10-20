"""基于 Ultralytics YOLOv8 的目标检测器。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..types import Detection


class YoloDetector:
    """YOLOv8 检测封装。

    只负责单帧推理与后处理，跟踪由 :class:`IoUTracker` 独立完成，
    便于在报告中拆分讲解检测与跟踪。
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Sequence[int] | None = None,
        max_det: int = 100,
        device: str | int = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到 YOLO 权重：{self.model_path}\n"
                "请先运行：python scripts/download_models.py"
            )
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes else None
        self.max_det = int(max_det)
        self.device = self._resolve_device(device)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 ultralytics：pip install ultralytics") from exc

        self.model = YOLO(str(self.model_path))
        self.names: dict[int, str] = self.model.names  # type: ignore[assignment]

    @staticmethod
    def _resolve_device(device: str | int) -> str | int:
        if device != "auto":
            return device
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """对 BGR 图像执行检测，返回按置信度降序的结果。"""
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for box, score, class_id in zip(xyxy, scores, class_ids):
            detections.append(
                Detection(
                    xyxy=tuple(float(v) for v in box),
                    score=float(score),
                    class_id=int(class_id),
                    class_name=str(self.names.get(int(class_id), str(class_id))),
                )
            )
        return detections

    @property
    def device_name(self) -> str:
        return f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
