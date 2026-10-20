"""下载并校验 RoboVision 所需的三个预训练模型。

用法：
    python scripts/download_models.py
    python scripts/download_models.py --check
    python scripts/download_models.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODELS = {
    "yolov8n.pt": {
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        ],
        "min_size": 5_000_000,
        "description": "YOLOv8n COCO 目标检测",
    },
    "face_detection_yunet_2023mar.onnx": {
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://gh-proxy.com/https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        "min_size": 150_000,
        "description": "YuNet 人脸检测",
    },
    "midas_small.onnx": {
        "urls": [
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
            "https://gh-proxy.com/https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
            "https://ghfast.top/https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        ],
        "min_size": 50_000_000,
        "description": "MiDaS-small 单目深度估计",
    },
}


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _download(url: str, target: Path, expected_min: int) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RoboVision/1.0"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_print = 0.0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print > 0.5:
                speed = downloaded / max(1e-6, now - start) / 1024 / 1024
                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\r  {downloaded / 1024 / 1024:7.1f} / {total / 1024 / 1024:.1f} MB "
                        f"({percent:5.1f}%)  {speed:5.1f} MB/s",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {downloaded / 1024 / 1024:7.1f} MB  {speed:5.1f} MB/s", end="", flush=True)
                last_print = now
    print()
    size = tmp.stat().st_size
    if size < expected_min:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"下载文件过小（{_human(size)}），可能不是有效模型")
    tmp.replace(target)


def ensure_models(force: bool = False, check_only: bool = False) -> bool:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_ready = True
    for filename, spec in MODELS.items():
        target = MODEL_DIR / filename
        min_size = int(spec["min_size"])
        ready = target.exists() and target.stat().st_size >= min_size
        if ready and not force:
            print(f"[跳过] {filename} 已存在（{_human(target.stat().st_size)}）")
            continue
        if check_only:
            print(f"[缺失] {filename}  {spec['description']}")
            all_ready = False
            continue

        print(f"[下载] {filename} - {spec['description']}")
        errors = []
        for url in spec["urls"]:
            try:
                _download(url, target, min_size)
                print(f"[完成] {filename} -> {target.relative_to(PROJECT_ROOT)}")
                errors.clear()
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                print(f"[失败] {url}\n       {type(exc).__name__}: {exc}")
                time.sleep(1)
        else:
            all_ready = False
            print(f"[错误] {filename} 所有下载源均失败：\n  " + "\n  ".join(errors))
    return all_ready


def verify_models() -> None:
    """基础校验：ONNX 可被 onnxruntime 加载，YOLO 权重可被 torch 读取。"""
    try:
        import onnxruntime as ort

        for name in ("face_detection_yunet_2023mar.onnx", "midas_small.onnx"):
            path = MODEL_DIR / name
            if path.exists():
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                shape = session.get_inputs()[0].shape
                print(f"[校验] {name} 输入 {shape}")
    except ImportError:
        print("[提示] 未安装 onnxruntime，跳过 ONNX 校验")
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] ONNX 校验失败：{exc}")

    try:
        import torch

        path = MODEL_DIR / "yolov8n.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            print(f"[校验] yolov8n.pt 可读取，顶层键 {list(checkpoint)[:3]} ...")
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 跳过 YOLO 权重深度校验：{exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 RoboVision 预训练模型")
    parser.add_argument("--check", action="store_true", help="仅检查模型是否齐全")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--verify", action="store_true", help="下载后执行加载校验")
    args = parser.parse_args()

    ok = ensure_models(force=args.force, check_only=args.check)
    if args.verify and not args.check:
        verify_models()
    if args.check:
        print("模型已齐全。" if ok else "存在缺失模型。")
    return 0 if ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
