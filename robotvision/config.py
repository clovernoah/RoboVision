"""配置加载与路径解析。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - 仅在未安装 PyYAML 时触发
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

_DEFAULTS: dict[str, Any] = {
    "project": {"name": "RoboVision", "seed": 42},
    "detection": {
        "model": "models/yolov8n.pt",
        "imgsz": 640,
        "conf": 0.35,
        "iou": 0.45,
        "max_det": 100,
        "device": "auto",
        "classes": [0, 1, 2, 3, 5, 7],
        "track": True,
        "tracker_iou": 0.30,
        "tracker_max_age": 30,
    },
    "face": {
        "model": "models/face_detection_yunet_2023mar.onnx",
        "score_threshold": 0.70,
        "nms_threshold": 0.30,
        "top_k": 5000,
    },
    "depth": {
        "model": "models/midas_small.onnx",
        "input_size": 256,
        "colormap": "inferno",
        "num_threads": 4,
        "clip_percentile": [2.0, 98.0],
    },
    "decision": {
        "device": "cpu",
        "cartpole": {"checkpoint": "runs/cartpole/best.pt", "episodes": 500, "max_steps": 500},
        "robotnav": {"checkpoint": "runs/robotnav/best.pt", "episodes": 1000, "max_steps": 100},
    },
    "web": {"host": "127.0.0.1", "port": 5000, "camera": 0, "max_upload_mb": 20},
    "visualization": {"box_thickness": 2, "font_scale": 0.55, "show_depth_level": True},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，override 优先。"""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置并与内置默认值合并。

    Args:
        path: 配置文件路径；为 None 时优先读取 configs/default.yaml。

    Returns:
        合并后的配置字典。
    """
    config = copy.deepcopy(_DEFAULTS)
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if path.exists() and yaml is not None:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        config = _deep_merge(config, loaded)
    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_path"] = str(path)
    return config


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """将配置中的相对路径解析为绝对路径。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(root or PROJECT_ROOT) / p


def ensure_parent(path: str | Path) -> Path:
    """创建文件父目录并返回路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def display_path(path: str | Path, root: str | Path | None = None) -> str:
    """把项目内路径显示为相对路径。

    日志、JSON 与 API 中不直接暴露本机绝对路径，便于分享与部署。
    对项目外路径原样返回。
    """
    p = Path(path)
    base = Path(root or PROJECT_ROOT).resolve()
    try:
        return p.resolve().relative_to(base).as_posix()
    except (ValueError, OSError):
        return str(path)


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """按 "a.b.c" 形式读取配置项。"""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
