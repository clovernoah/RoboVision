"""RoboVision 命令行入口。

示例：
    python -m robotvision analyze --source assets/samples/street_traffic.jpg
    python -m robotvision detect --source 0 --out runs/live
    python -m robotvision train --env robotnav --episodes 1000
    python -m robotvision eval --env robotnav
    python -m robotvision web
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import PROJECT_ROOT, display_path, get, load_config, resolve_path
from .pipelines.perception import PerceptionPipeline
from .utils import is_image_file, is_video_file, open_video_source, parse_source


def _json_result(result: dict[str, Any], source: str) -> dict[str, Any]:
    depth = result.get("depth")
    return {
        "source": source,
        "size": result.get("size"),
        "detections": [d.to_dict() for d in result.get("detections", [])],
        "faces": [f.to_dict() for f in result.get("faces", [])],
        "depth": None if depth is None else {"stats": depth.stats()},
        "fusion": result.get("fusion"),
        "timings_ms": result.get("timings_ms"),
        "counts": {
            "person": sum(1 for d in result.get("detections", []) if d.class_name == "person"),
            "vehicle": sum(
                1
                for d in result.get("detections", [])
                if d.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}
            ),
            "face": len(result.get("faces", [])),
        },
    }


def _save_outputs(
    result: dict[str, Any],
    images: dict[str, np.ndarray],
    out_dir: Path,
    source: str,
    pipeline: PerceptionPipeline | None = None,
    save_pointcloud: bool = False,
    frame: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        cv2.imwrite(str(out_dir / f"{name}.jpg"), image)
    if save_pointcloud and result.get("depth") is not None and frame is not None and pipeline is not None:
        from .models.depth_estimator import DepthEstimator

        points, colors = pipeline.depth_estimator.to_pointcloud(frame, result["depth"].depth)
        DepthEstimator.save_ply(out_dir / "pointcloud.ply", points, colors)
    (out_dir / "result.json").write_text(
        json.dumps(_json_result(result, source), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def process_image(
    pipeline: PerceptionPipeline,
    path: Path,
    out_dir: Path,
    tasks: Iterable[str],
    pointcloud: bool = False,
) -> dict[str, Any]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"无法读取图片：{path}")
    result = pipeline.analyze(frame, tasks=tasks)
    images = pipeline.render(frame, result)
    source_label = display_path(path)
    _save_outputs(
        result, images, out_dir, source_label,
        pipeline=pipeline, save_pointcloud=pointcloud, frame=frame,
    )
    counts = _json_result(result, source_label)["counts"]
    print(
        f"[完成] {path.name} | 目标 {counts['person']} 人 / {counts['vehicle']} 车 | "
        f"人脸 {counts['face']} | 总耗时 {result['timings_ms']['total']:.1f} ms -> {display_path(out_dir)}"
    )
    return result


def process_video(
    pipeline: PerceptionPipeline,
    cap: cv2.VideoCapture,
    out_dir: Path,
    tasks: Iterable[str],
    max_frames: int | None = None,
    display: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frames = 0
    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.analyze(frame, tasks=tasks, track=True)
            images = pipeline.render(frame, result)
            task_set = tuple(tasks)
            if task_set == ("face",) and "faces" in images:
                output = images["faces"]
            elif task_set == ("depth",):
                output = images.get("depth", images["annotated"])
            elif "fused" in images:
                output = images["fused"]
            else:
                output = images["annotated"]
            writer.write(output)
            frames += 1
            if frames % 30 == 0:
                elapsed = time.time() - start
                print(f"\r已处理 {frames} 帧 | {frames / max(1e-6, elapsed):.1f} FPS", end="", flush=True)
            if display:
                cv2.imshow("RoboVision", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and frames >= max_frames:
                break
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
    elapsed = time.time() - start
    print(f"\n[完成] {frames} 帧 -> {display_path(out_dir / 'annotated.mp4')} | 平均 {frames / max(1e-6, elapsed):.1f} FPS")


def run_perception(args: argparse.Namespace, tasks: tuple[str, ...]) -> int:
    config = load_config(args.config)
    pipeline = PerceptionPipeline(config)
    source = parse_source(args.source)
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "runs" / "outputs" / time.strftime("%Y%m%d_%H%M%S")

    if isinstance(source, int):
        cap = open_video_source(source)
        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头 {source}")
            return 1
        print(f"[摄像头] 输入 {source}，按 q 退出")
        process_video(pipeline, cap, out_dir, tasks, args.max_frames, display=True)
        return 0

    path = Path(source)
    if not path.exists():
        print(f"[错误] 路径不存在：{path}")
        return 1
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.is_file() and is_image_file(p))
        if not images:
            print(f"[错误] 目录中没有图片：{path}")
            return 1
        for image_path in images:
            process_image(pipeline, image_path, out_dir / image_path.stem, tasks, args.pointcloud)
        return 0
    if is_image_file(path):
        process_image(pipeline, path, out_dir, tasks, args.pointcloud)
        return 0
    if is_video_file(path):
        cap = open_video_source(str(path))
        if not cap.isOpened():
            print(f"[错误] 无法打开视频：{path}")
            return 1
        process_video(pipeline, cap, out_dir, tasks, args.max_frames, display=args.display)
        return 0
    print(f"[错误] 不支持的文件类型：{path.suffix}")
    return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    return run_perception(args, ("detect", "face", "depth"))


def cmd_detect(args: argparse.Namespace) -> int:
    return run_perception(args, ("detect",))


def cmd_face(args: argparse.Namespace) -> int:
    return run_perception(args, ("face",))


def cmd_depth(args: argparse.Namespace) -> int:
    return run_perception(args, ("depth",))


def cmd_train(args: argparse.Namespace) -> int:
    from .decision.train import train

    summary = train(
        env_name=args.env,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        double_dqn=not args.no_double,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .decision.evaluate import evaluate_checkpoint, evaluate_policy

    checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_path(
        get(load_config(), f"decision.{args.env}.checkpoint")
    )
    result = evaluate_checkpoint(args.env, checkpoint, episodes=args.episodes, gif=not args.no_gif)
    print(
        f"[DQN] 成功率 {result['success_rate']:.1%} | 平均回报 {result['mean_return']:.2f} | "
        f"平均步数 {result['mean_steps']:.1f}"
    )
    if not args.no_baselines:
        baseline = "astar" if args.env == "robotnav" else "heuristic"
        for name in ("random", baseline):
            other = evaluate_policy(args.env, name, episodes=args.episodes, record_gif=False)
            print(
                f"[{name:8s}] 成功率 {other['success_rate']:.1%} | 平均回报 {other['mean_return']:.2f} | "
                f"平均步数 {other['mean_steps']:.1f}"
            )
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .web.app import create_app

    config = load_config(args.config)
    app = create_app(args.config)
    host = args.host or get(config, "web.host", "127.0.0.1")
    port = args.port or int(get(config, "web.port", 5000))
    print(f"RoboVision Web：http://{host}:{port}")
    app.run(host=host, port=port, debug=args.debug, threaded=True, use_reloader=False)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    commands = [[sys.executable, str(PROJECT_ROOT / "scripts" / "download_models.py"), "--verify"]]
    if args.assets:
        commands.append([sys.executable, str(PROJECT_ROOT / "scripts" / "download_assets.py")])
    for command in commands:
        code = subprocess.call(command)
        if code != 0:
            return code
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = PerceptionPipeline(config, tasks=("detect",))
    status = pipeline.status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    for env_name in ("cartpole", "robotnav"):
        path = resolve_path(get(config, f"decision.{env_name}.checkpoint"))
        print(f"{env_name:10s} checkpoint: {'OK' if path.exists() else '缺失'} {display_path(path)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robotvision", description="RoboVision 智能机器人视觉系统")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, help_text in (
        ("analyze", cmd_analyze, "检测 + 人脸 + 深度联合分析"),
        ("detect", cmd_detect, "行人/车辆目标检测"),
        ("face", cmd_face, "人脸检测"),
        ("depth", cmd_depth, "单目深度估计"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--source", required=True, help="图片/视频/目录，或摄像头编号如 0")
        p.add_argument("--out", default=None, help="输出目录")
        p.add_argument("--max-frames", type=int, default=None, help="视频最大处理帧数")
        p.add_argument("--pointcloud", action="store_true", help="保存深度点云 PLY")
        p.add_argument("--display", action="store_true", help="视频处理时显示窗口")
        p.set_defaults(func=func)

    p = sub.add_parser("train", help="训练 DQN/Double DQN")
    p.add_argument("--env", choices=["cartpole", "robotnav"], default="robotnav")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-double", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="评估 DQN 与基线")
    p.add_argument("--env", choices=["cartpole", "robotnav"], default="robotnav")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("web", help="启动 Web 演示")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("download", help="下载模型与示例图片")
    p.add_argument("--assets", action="store_true", help="同时下载示例图片")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("info", help="查看模型与检查点状态")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
