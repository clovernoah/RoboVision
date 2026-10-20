"""多任务感知流水线：目标检测 + 人脸 + 深度 + 融合。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import display_path, get, load_config, resolve_path
from ..fusion.navigation import analyze_scene
from ..models.depth_estimator import DepthEstimator
from ..models.face_detector import FaceDetector
from ..models.tracker import IoUTracker
from ..models.yolo_detector import YoloDetector
from ..types import DepthResult, Detection, Face
from .visualization import (
    draw_band_legend,
    draw_detections,
    draw_faces,
    draw_hud,
    depth_panel,
)

ALL_TASKS = ("detect", "face", "depth")


class PerceptionPipeline:
    """按需懒加载三个感知模型，并串接成统一分析接口。

    模型加载较慢且占用显存/内存，因此只在第一次使用时初始化；
    推理过程加锁，保证 Flask 多线程环境下的安全。
    """

    def __init__(self, config: dict[str, Any] | None = None, tasks: Iterable[str] = ALL_TASKS) -> None:
        self.config = config or load_config()
        self.root = Path(self.config.get("_project_root", "."))
        self.tasks = tuple(t for t in tasks if t in ALL_TASKS)
        self._lock = threading.RLock()

        self._detector: YoloDetector | None = None
        self._face: FaceDetector | None = None
        self._depth: DepthEstimator | None = None

        det_cfg = self.config.get("detection", {})
        self.tracker = IoUTracker(
            iou_threshold=float(det_cfg.get("tracker_iou", 0.3)),
            max_age=int(det_cfg.get("tracker_max_age", 30)),
        )

    # ------------------------------------------------------------------
    # 懒加载
    # ------------------------------------------------------------------
    @property
    def detector(self) -> YoloDetector:
        if self._detector is None:
            cfg = self.config["detection"]
            self._detector = YoloDetector(
                model_path=resolve_path(cfg["model"], self.root),
                conf=cfg["conf"],
                iou=cfg["iou"],
                imgsz=cfg["imgsz"],
                classes=cfg.get("classes") or None,
                max_det=cfg["max_det"],
                device=cfg.get("device", "auto"),
            )
        return self._detector

    @property
    def face_detector(self) -> FaceDetector:
        if self._face is None:
            cfg = self.config["face"]
            self._face = FaceDetector(
                model_path=resolve_path(cfg["model"], self.root),
                score_threshold=cfg["score_threshold"],
                nms_threshold=cfg["nms_threshold"],
                top_k=cfg.get("top_k", 5000),
            )
        return self._face

    @property
    def depth_estimator(self) -> DepthEstimator:
        if self._depth is None:
            cfg = self.config["depth"]
            self._depth = DepthEstimator(
                model_path=resolve_path(cfg["model"], self.root),
                input_size=cfg["input_size"],
                colormap=cfg["colormap"],
                num_threads=cfg.get("num_threads", 4),
                clip_percentile=tuple(cfg.get("clip_percentile", [2.0, 98.0])),
            )
        return self._depth

    # ------------------------------------------------------------------
    # 单项能力
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, track: bool = False) -> list[Detection]:
        with self._lock:
            detections = self.detector.predict(frame)
            if track and self.config["detection"].get("track", True):
                detections = self.tracker.update(detections)
            return detections

    def detect_faces(self, frame: np.ndarray) -> list[Face]:
        with self._lock:
            return self.face_detector.detect(frame)

    def estimate_depth(self, frame: np.ndarray) -> DepthResult:
        with self._lock:
            return self.depth_estimator.infer(frame)

    def reset_tracker(self) -> None:
        self.tracker.reset()

    # ------------------------------------------------------------------
    # 组合分析
    # ------------------------------------------------------------------
    def analyze(
        self,
        frame: np.ndarray,
        tasks: Iterable[str] | None = None,
        track: bool = False,
    ) -> dict[str, Any]:
        """执行多任务分析并返回结构化结果。

        Returns:
            dict，包含 detections/faces/depth/fusion/timings_ms 等键。
        """
        chosen = tuple(tasks) if tasks is not None else self.tasks
        result: dict[str, Any] = {
            "detections": [],
            "faces": [],
            "depth": None,
            "fusion": None,
            "timings_ms": {},
            "size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        }
        total_start = time.perf_counter()

        if "detect" in chosen:
            start = time.perf_counter()
            result["detections"] = self.detect(frame, track=track)
            result["timings_ms"]["detect"] = round((time.perf_counter() - start) * 1000, 2)

        if "face" in chosen:
            start = time.perf_counter()
            result["faces"] = self.detect_faces(frame)
            result["timings_ms"]["face"] = round((time.perf_counter() - start) * 1000, 2)

        if "depth" in chosen:
            start = time.perf_counter()
            result["depth"] = self.estimate_depth(frame)
            result["timings_ms"]["depth"] = round((time.perf_counter() - start) * 1000, 2)

        if result["depth"] is not None and result["detections"]:
            start = time.perf_counter()
            result["fusion"] = analyze_scene(result["detections"], result["depth"])
            if result["faces"]:
                result["fusion"]["counts"]["face"] = len(result["faces"])
            result["timings_ms"]["fusion"] = round((time.perf_counter() - start) * 1000, 2)
        elif result["depth"] is not None:
            result["fusion"] = analyze_scene([], result["depth"])

        result["timings_ms"]["total"] = round((time.perf_counter() - total_start) * 1000, 2)
        return result

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, result: dict[str, Any], hud: list[str] | None = None) -> dict[str, np.ndarray]:
        """把分析结果渲染为若干 BGR 图像，供 CLI / Web 展示。"""
        detections: list[Detection] = result.get("detections", [])
        faces: list[Face] = result.get("faces", [])
        depth: DepthResult | None = result.get("depth")

        images: dict[str, np.ndarray] = {}
        annotated = draw_detections(
            frame,
            detections,
            depth=None if depth is None else depth.depth,
            show_band=depth is not None,
        )
        if depth is not None:
            annotated = draw_band_legend(annotated)
        if hud:
            annotated = draw_hud(annotated, hud)
        images["annotated"] = annotated

        if faces:
            images["faces"] = draw_faces(frame, faces)
        if depth is not None:
            images["depth"] = depth.color
            images["depth_panel"] = depth_panel(frame, depth)
        if result.get("fusion") and detections:
            fused = draw_detections(frame, detections, depth=depth.depth, show_band=True)
            fused = draw_band_legend(fused)
            suggestion = result["fusion"].get("suggestion", {})
            hud_lines = [f"risk suggestion: {suggestion.get('action', '-')}"]
            images["fused"] = draw_hud(fused, hud_lines)
        return images

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        checks = {}
        for name, cfg_key in (("detection", "detection"), ("face", "face"), ("depth", "depth")):
            path = resolve_path(self.config[cfg_key]["model"], self.root)
            checks[name] = {"model": display_path(path, self.root), "exists": path.exists()}
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:  # pragma: no cover
            pass
        checks["device"] = device
        checks["tasks"] = list(self.tasks)
        return checks
