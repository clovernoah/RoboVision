"""生成课程报告所需的真实插图、训练曲线、对比图与 GIF。

用法：
    python scripts/make_report_assets.py
    python scripts/make_report_assets.py --skip-perception
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robotvision.config import display_path, get, load_config, resolve_path  # noqa: E402
from robotvision.decision.dqn import DQNAgent  # noqa: E402
from robotvision.decision.evaluate import evaluate_policy  # noqa: E402
from robotvision.pipelines.perception import PerceptionPipeline  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "docs" / "images"
SAMPLE_DIR = PROJECT_ROOT / "assets" / "samples"

PERCEPTION_JOBS = [
    ("street_traffic.jpg", ("detect", "depth"), {"annotated": "detection_street.png", "depth_panel": "depth_street.png", "fused": "fusion_street.png"}),
    ("street_pedestrian.jpg", ("detect",), {"annotated": "detection_pedestrian.png"}),
    ("bus.jpg", ("detect",), {"annotated": "detection_bus.png"}),
    ("cars.jpg", ("detect",), {"annotated": "detection_cars.png"}),
    ("portrait.jpg", ("face",), {"faces": "face_portrait.png"}),
    ("group.jpg", ("face",), {"faces": "face_group.png"}),
    ("students.jpg", ("face",), {"faces": "face_students.png"}),
    ("city_street.jpg", ("detect", "depth"), {"depth_panel": "depth_city.png", "fused": "fusion_city.png"}),
]


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def fit(image: np.ndarray, max_width: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def generate_perception_assets() -> dict:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = PerceptionPipeline(load_config())
    summary = {}
    for filename, tasks, outputs in PERCEPTION_JOBS:
        path = SAMPLE_DIR / filename
        if not path.exists():
            print(f"[跳过] 缺少示例图 {filename}")
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = pipeline.analyze(frame, tasks=tasks)
        images = pipeline.render(frame, result)
        for key, out_name in outputs.items():
            if key not in images:
                continue
            cv2.imwrite(str(IMAGE_DIR / out_name), fit(images[key]))
        counts = {
            "person": sum(1 for d in result["detections"] if d.class_name == "person"),
            "vehicle": sum(
                1 for d in result["detections"] if d.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}
            ),
            "face": len(result["faces"]),
        }
        summary[filename] = {
            "tasks": list(tasks),
            "counts": counts,
            "timings_ms": result["timings_ms"],
            "suggestion": (result.get("fusion") or {}).get("suggestion"),
            "outputs": list(outputs.values()),
        }
        print(f"[报告插图] {filename} -> {list(outputs.values())} | {counts}")
    (IMAGE_DIR / "perception_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def generate_training_figures() -> None:
    for env in ("cartpole", "robotnav"):
        source = PROJECT_ROOT / "runs" / env / "training_curve.png"
        if source.exists():
            shutil.copyfile(source, IMAGE_DIR / f"{env}_training.png")
            print(f"[训练曲线] {display_path(source)} -> {display_path(IMAGE_DIR / f'{env}_training.png')}")
        else:
            print(f"[跳过] 缺少 {display_path(source)}，请先训练")


def draw_architecture() -> None:
    plt = setup_matplotlib()
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14, 8), dpi=170)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")

    def box(x, y, w, h, title, subtitle="", color="#ffffff", edge="#94a3b8", title_size=12):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.18",
            linewidth=1.2, edgecolor=edge, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.12 if subtitle else 0), title,
                ha="center", va="center", fontsize=title_size, fontweight="bold", color="#0f172a")
        if subtitle:
            ax.text(x + w / 2, y + h / 2 - 0.26, subtitle,
                    ha="center", va="center", fontsize=8.5, color="#475569")

    def arrow(x1, y1, x2, y2, color="#64748b"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2))

    ax.text(0.4, 7.55, "RoboVision 系统总体架构", fontsize=18, fontweight="bold", color="#0f172a")
    ax.text(0.4, 7.18, "多任务感知  →  跨模态融合  →  深度强化学习决策", fontsize=10, color="#64748b")

    box(0.4, 4.7, 2.4, 1.5, "输入", "图像 / 视频 / 摄像头", "#e0f2fe", "#38bdf8")
    box(3.5, 5.55, 2.9, 1.05, "YOLOv8n", "行人 · 车辆 · 80 类 COCO", "#dcfce7", "#34d399")
    box(3.5, 4.2, 2.9, 1.05, "YuNet", "人脸框 + 5 关键点", "#ede9fe", "#a78bfa")
    box(3.5, 2.85, 2.9, 1.05, "MiDaS-small", "单目相对深度（ONNX）", "#fef3c7", "#fbbf24")
    box(7.2, 3.6, 2.7, 2.3, "跨模态融合", "目标框 × 深度图\n近 / 中 / 远 + 风险格网\n8 向射线跟踪", "#f1f5f9", "#64748b")
    box(10.7, 5.1, 2.9, 1.3, "Web 可视化", "Flask + Canvas\n检测 / 深度 / 决策演示", "#ecfeff", "#06b6d4")
    box(10.7, 3.15, 2.9, 1.3, "Double DQN 决策", "RobotNav 11→128→128→4\nCartPole 4→128→128→2", "#fef2f2", "#f87171")
    box(10.7, 1.35, 2.9, 1.15, "评估与基线", "随机 · 启发式 · A*", "#f8fafc", "#94a3b8")

    arrow(2.8, 5.45, 3.5, 6.05)
    arrow(2.8, 5.45, 3.5, 4.72)
    arrow(2.8, 5.45, 3.5, 3.38)
    arrow(6.4, 6.05, 7.2, 5.3)
    arrow(6.4, 4.72, 7.2, 4.75)
    arrow(6.4, 3.38, 7.2, 4.2)
    arrow(9.9, 5.2, 10.7, 5.75)
    arrow(9.9, 4.2, 10.7, 3.8)
    arrow(12.15, 3.15, 12.15, 2.5, "#f87171")

    ax.text(0.4, 0.55, "注：感知模块与决策模块共享统一的配置、类型定义与工程化接口；"
                        "深度为相对深度，未做相机标定。",
            fontsize=8.5, color="#64748b")

    fig.tight_layout()
    output = IMAGE_DIR / "architecture.png"
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[架构图] {display_path(output)}")


def generate_evaluation(episodes: int = 300) -> dict:
    plt = setup_matplotlib()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    summary: dict = {"episodes": episodes, "seed_base": 30000, "results": {}}

    for env in ("robotnav", "cartpole"):
        checkpoint = resolve_path(get(config, f"decision.{env}.checkpoint"))
        env_results = {}
        if checkpoint.exists():
            agent = DQNAgent.load(checkpoint, device=get(config, "decision.device", "cpu"))
            env_results["dqn"] = evaluate_policy(
                env_name=env, policy="dqn", agent=agent, episodes=episodes,
                seed_base=30000, record_gif=False,
            )
            # 单独录制一个成功回合作为 GIF，便于报告展示
            success_seed = next(
                (item["seed"] for item in env_results["dqn"]["details"] if item["success"]), None
            )
            if success_seed is not None:
                evaluate_policy(
                    env_name=env, policy="dqn", agent=agent, episodes=1, seed_base=success_seed,
                    gif_path=IMAGE_DIR / f"{env}_episode.gif", record_gif=True,
                    gif_stride=3 if env == "cartpole" else 1,
                    gif_max_width=400 if env == "cartpole" else 480,
                )
            print(f"[评估] {env} DQN 成功率 {env_results['dqn']['success_rate']:.1%}（{episodes} 回合）")
        else:
            print(f"[跳过] 缺少检查点 {checkpoint}")

        env_results["random"] = evaluate_policy(
            env_name=env, policy="random", episodes=episodes, seed_base=30000, record_gif=False
        )
        baseline = "astar" if env == "robotnav" else "heuristic"
        env_results[baseline] = evaluate_policy(
            env_name=env, policy=baseline, episodes=episodes, seed_base=30000, record_gif=False
        )
        summary["results"][env] = env_results

    # JSON 仅保留聚合指标，避免逐回合明细过大
    for env_results in summary["results"].values():
        for result in env_results.values():
            result.pop("details", None)
    output_json = PROJECT_ROOT / "runs" / "evaluation_summary.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2×2 对比图：成功率与平均回报
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    colors = {"dqn": "#34d399", "random": "#94a3b8", "astar": "#38bdf8", "heuristic": "#a78bfa"}
    env_titles = {"robotnav": "机器人栅格导航", "cartpole": "CartPole 平衡"}
    for row, env in enumerate(("robotnav", "cartpole")):
        results = summary["results"][env]
        names = list(results)
        success = [results[name]["success_rate"] * 100 for name in names]
        returns = [results[name]["mean_return"] for name in names]
        bar_colors = [colors.get(name, "#fbbf24") for name in names]
        axes[row, 0].bar(names, success, color=bar_colors, width=0.55)
        axes[row, 0].set_title(f"{env_titles[env]} · 成功率")
        axes[row, 0].set_ylim(0, 110)
        axes[row, 0].set_ylabel("%")
        for i, value in enumerate(success):
            axes[row, 0].text(i, value + 2, f"{value:.0f}%", ha="center", fontsize=9)
        axes[row, 1].bar(names, returns, color=bar_colors, width=0.55)
        axes[row, 1].set_title(f"{env_titles[env]} · 平均回报")
        low, high = min(returns + [0]), max(returns + [1])
        span = max(1.0, high - low)
        axes[row, 1].set_ylim(low - span * 0.18, high + span * 0.12)
        for i, value in enumerate(returns):
            offset = span * 0.03
            axes[row, 1].text(i, value + offset, f"{value:.1f}", ha="center", fontsize=9)
        for col in range(2):
            axes[row, col].grid(axis="y", alpha=0.25)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"DQN 与基线策略对比（{episodes} 回合，相同随机种子）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(IMAGE_DIR / "decision_comparison.png")
    plt.close(fig)
    print(f"[对比图] {display_path(IMAGE_DIR / 'decision_comparison.png')}")
    return summary


def generate_benchmark_chart() -> None:
    path = PROJECT_ROOT / "runs" / "benchmark" / "benchmark.json"
    if not path.exists():
        print("[跳过] 缺少 benchmark.json，请先运行 scripts/benchmark.py")
        return
    plt = setup_matplotlib()
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report["overall"]
    labels = {"detect": "目标检测", "face": "人脸检测", "depth": "深度估计", "analyze_all": "完整流水线"}
    names = [labels.get(key, key) for key in overall]
    means = [overall[key]["mean_ms"] for key in overall]
    colors = ["#34d399", "#a78bfa", "#fbbf24", "#38bdf8"]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    bars = ax.bar(names, means, color=colors, width=0.55)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f} ms", ha="center", fontsize=10)
    ax.set_ylabel("平均延迟 / ms")
    ax.set_title("感知模块推理延迟（CPU，批次均值）")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "benchmark.png")
    plt.close(fig)
    print(f"[基准图] {display_path(IMAGE_DIR / 'benchmark.png')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成报告插图")
    parser.add_argument("--skip-perception", action="store_true")
    parser.add_argument("--skip-decision", action="store_true")
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    draw_architecture()
    if not args.skip_perception:
        generate_perception_assets()
    generate_training_figures()
    if not args.skip_decision:
        generate_evaluation()
    generate_benchmark_chart()
    print(f"\n全部插图已生成：{display_path(IMAGE_DIR)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
