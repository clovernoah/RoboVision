"""感知模块性能基准：检测 / 人脸 / 深度的延迟与吞吐。

用法：
    python scripts/benchmark.py --iterations 20 --images 4
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robotvision.config import display_path, load_config, resolve_path  # noqa: E402
from robotvision.pipelines.perception import PerceptionPipeline  # noqa: E402
from robotvision.utils import is_image_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboVision 感知性能基准")
    parser.add_argument("--images", type=str, default=str(PROJECT_ROOT / "assets" / "samples"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "runs" / "benchmark"))
    return parser.parse_args()


def summarize(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "std_ms": round(float(arr.std()), 2),
        "fps": round(float(1000.0 / max(1e-9, arr.mean())), 2),
    }


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and is_image_file(p))[: args.limit]
    if not images:
        print(f"[错误] 目录中没有图片：{display_path(image_dir)}\n请先运行 python scripts/download_assets.py")
        return 1

    config = load_config()
    pipeline = PerceptionPipeline(config)
    status = pipeline.status()
    print("模型状态：")
    for key in ("detection", "face", "depth"):
        item = status[key]
        print(f"  {key:10s} {'OK ' if item['exists'] else '缺失'} {item['model']}")
    missing = [k for k in ("detection", "face", "depth") if not status[k]["exists"]]
    if missing:
        print(f"[错误] 缺少模型：{missing}\n请先运行 python scripts/download_models.py")
        return 1

    frames = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append((path, frame))
    if not frames:
        print("[错误] 图片读取失败")
        return 1

    timings: dict[str, list[float]] = {"detect": [], "face": [], "depth": [], "analyze_all": []}
    per_image: list[dict] = []
    print(f"\n对 {len(frames)} 张图片、每项 {args.iterations} 次推理进行测量...\n")

    for path, frame in frames:
        # 预热，避免首次加载/内存分配影响结果
        for _ in range(args.warmup):
            pipeline.detect(frame)
            pipeline.detect_faces(frame)
            pipeline.estimate_depth(frame)

        local = {"image": path.name, "size": f"{frame.shape[1]}x{frame.shape[0]}"}
        for task, fn in (
            ("detect", lambda: pipeline.detect(frame)),
            ("face", lambda: pipeline.detect_faces(frame)),
            ("depth", lambda: pipeline.estimate_depth(frame)),
        ):
            values = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                fn()
                values.append((time.perf_counter() - start) * 1000)
            local[task] = summarize(values)
            local[task]["count"] = len(fn()) if task != "depth" else 1
            timings[task].extend(values)

        values = []
        for _ in range(max(1, args.iterations // 2)):
            start = time.perf_counter()
            pipeline.analyze(frame)
            values.append((time.perf_counter() - start) * 1000)
        local["analyze_all"] = summarize(values)
        timings["analyze_all"].extend(values)
        per_image.append(local)
        print(
            f"{path.name:28s} detect {local['detect']['mean_ms']:7.2f}ms | "
            f"face {local['face']['mean_ms']:9.2f}ms | depth {local['depth']['mean_ms']:7.2f}ms | "
            f"all {local['analyze_all']['mean_ms']:7.2f}ms"
        )

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    for module in ("torch", "ultralytics", "onnxruntime"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            versions[module] = "未安装"

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": status.get("device", "cpu"),
        "versions": versions,
        "iterations": args.iterations,
        "overall": {key: summarize(values) for key, values in timings.items() if values},
        "per_image": per_image,
        "models": {
            name: {
                "path": status[name]["model"],
                "size_mb": round(
                    resolve_path(status[name]["model"], PROJECT_ROOT).stat().st_size / 1024 / 1024, 2
                ),
            }
            for name in ("detection", "face", "depth")
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n总体结果：")
    for key, value in report["overall"].items():
        print(f"  {key:12s} mean {value['mean_ms']:7.2f} ms | p95 {value['p95_ms']:7.2f} ms | {value['fps']:6.1f} FPS")
    print(f"\n已保存：{display_path(out_dir / 'benchmark.json')}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
