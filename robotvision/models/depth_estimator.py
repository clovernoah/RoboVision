"""基于 MiDaS-small（ONNX）的单目深度估计。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import DepthResult

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator:
    """MiDaS 单目相对深度估计器。

    MiDaS 输出的是**逆深度**（数值越大越近）。本类将其归一化为 [0, 1] 的
    “相对接近度”，并生成伪彩色图，便于与检测结果融合。
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 256,
        colormap: str = "inferno",
        num_threads: int = 4,
        clip_percentile: tuple[float, float] = (2.0, 98.0),
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到 MiDaS 深度模型：{self.model_path}\n"
                "请先运行：python scripts/download_models.py"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 onnxruntime：pip install onnxruntime") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(num_threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = int(input_size)
        self.clip_percentile = (float(clip_percentile[0]), float(clip_percentile[1]))
        self.colormap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)

    # ------------------------------------------------------------------
    # 预处理 / 推理
    # ------------------------------------------------------------------
    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB → 归一化 → NCHW。"""
        size = self.input_size
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))
        return np.ascontiguousarray(image[None, ...], dtype=np.float32)

    def _normalize(self, raw: np.ndarray) -> tuple[np.ndarray, float, float]:
        low, high = np.percentile(raw, self.clip_percentile)
        if high - low < 1e-6:
            normalized = np.zeros_like(raw, dtype=np.float32)
        else:
            normalized = (raw - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32), float(low), float(high)

    def infer(self, frame_bgr: np.ndarray) -> DepthResult:
        """对单帧执行深度估计，返回与输入同尺寸的结果。"""
        height, width = frame_bgr.shape[:2]
        tensor = self.preprocess(frame_bgr)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        raw = np.asarray(raw).squeeze()
        if raw.ndim != 2:
            raise RuntimeError(f"MiDaS 输出维度异常：{raw.shape}")

        normalized, low, high = self._normalize(raw)
        depth = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_CUBIC)
        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

        color = cv2.applyColorMap(np.uint8(depth * 255.0), self.colormap)
        return DepthResult(
            depth=depth,
            raw=raw,
            color=color,
            min_val=low,
            max_val=high,
            meta={"input_size": self.input_size, "model": self.model_path.name},
        )

    # ------------------------------------------------------------------
    # 下游应用
    # ------------------------------------------------------------------
    @staticmethod
    def roi_proximity(depth: np.ndarray, box: tuple[float, float, float, float], lower_ratio: float = 0.55) -> float:
        """取检测框下部区域的中位接近度，代表目标所在位置。

        车辆/行人等目标的底部更接近地面，深度更稳定，因此只取框的下半部分。
        """
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = box
        x1 = int(np.clip(round(x1), 0, width - 1))
        x2 = int(np.clip(round(x2), x1 + 1, width))
        y1 = int(np.clip(round(y1), 0, height - 1))
        y2 = int(np.clip(round(y2), y1 + 1, height))
        cut = int(y1 + (y2 - y1) * (1.0 - lower_ratio))
        roi = depth[cut:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.median(roi))

    @staticmethod
    def proximity_band(value: float) -> str:
        """将相对接近度映射为近/中/远三档。"""
        if value >= 0.62:
            return "near"
        if value >= 0.38:
            return "mid"
        return "far"

    def to_pointcloud(self, frame_bgr: np.ndarray, depth: np.ndarray, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """用简化针孔模型生成相对三维点云（不依赖 Open3D）。

        Returns:
            points: (N, 3) float32，单位为任意相对尺度；
            colors: (N, 3) uint8，RGB。
        """
        height, width = frame_bgr.shape[:2]
        fx = fy = 0.9 * max(width, height)
        cx, cy = width / 2.0, height / 2.0
        ys, xs = np.mgrid[0:height:stride, 0:width:stride]
        z = 1.0 - depth[ys, xs]  # 越近越小
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        colors = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[ys, xs].reshape(-1, 3)
        return points, colors

    @staticmethod
    def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
        """保存带颜色的 PLY 点云。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(points)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(points, colors):
                fh.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
        return path
