"""整理 DQN 训练记录，并生成汇总图与训练过程 GIF。

用法：
    python scripts/make_training_assets.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs" / "training"

COPY_FILES = {
    "runs/robotnav/metrics.csv": "robotnav_metrics.csv",
    "runs/cartpole/metrics.csv": "cartpole_metrics.csv",
    "runs/robotnav_vanilla/metrics.csv": "robotnav_vanilla_metrics.csv",
    "runs/train_robotnav.log": "train_robotnav.log",
    "runs/train_cartpole.log": "train_cartpole.log",
    "runs/train_robotnav_vanilla.log": "train_robotnav_vanilla.log",
    "runs/robotnav/training_curve.png": "robotnav_training_curve.png",
    "runs/cartpole/training_curve.png": "cartpole_training_curve.png",
    "runs/robotnav_vanilla/training_curve.png": "robotnav_vanilla_training_curve.png",
    "runs/robotnav/eval_all.json": "robotnav_eval_all.json",
    "runs/cartpole/eval_all.json": "cartpole_eval_all.json",
    "runs/robotnav_vanilla/eval_all.json": "robotnav_vanilla_eval_all.json",
    "runs/robotnav/summary.json": "robotnav_summary.json",
    "runs/cartpole/summary.json": "cartpole_summary.json",
    "runs/robotnav_vanilla/summary.json": "robotnav_vanilla_summary.json",
    "runs/evaluation_summary.json": "evaluation_summary.json",
    "runs/ablation_double_vs_vanilla.json": "ablation_double_vs_vanilla.json",
}

TITLES = {"robotnav": "机器人栅格导航", "cartpole": "CartPole 平衡"}


def load_metrics(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    data = {
        "episode": [int(r["episode"]) for r in rows],
        "return": [float(r["return"]) for r in rows],
        "success": [int(r["success"]) for r in rows],
        "eval_episode": [],
        "eval_success": [],
    }
    for r in rows:
        if r["eval_success"]:
            data["eval_episode"].append(int(r["episode"]))
            data["eval_success"].append(float(r["eval_success"]))
    return data


def smooth(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.convolve(np.asarray(values), np.ones(window) / window, mode="valid")
    return np.asarray(range(window, len(values) + 1)), arr


def generate_dashboard() -> None:
    data = {
        env: load_metrics(OUT / f"{env}_metrics.csv")
        for env in ("robotnav", "cartpole")
    }
    vanilla = load_metrics(OUT / "robotnav_vanilla_metrics.csv")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    for row, env in enumerate(("robotnav", "cartpole")):
        d = data[env]
        window = 50 if env == "robotnav" else 25
        sx, sy = smooth(d["return"], window)
        axes[row, 0].plot(d["episode"], d["return"], color="#9fb4cc", alpha=0.3, linewidth=0.7, label="每回合回报")
        if len(sx):
            axes[row, 0].plot(sx, sy, color="#2f7ed8", linewidth=2, label=f"滑动平均({window})")
        axes[row, 0].set_title(f"{TITLES[env]} · 训练回报")
        axes[row, 0].set_xlabel("回合")
        axes[row, 0].set_ylabel("累计奖励")
        axes[row, 0].legend(frameon=False)
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(d["eval_episode"], np.array(d["eval_success"]) * 100, color="#2aa876", marker="o", markersize=3, label="Double DQN")
        if env == "robotnav":
            axes[row, 1].plot(vanilla["eval_episode"], np.array(vanilla["eval_success"]) * 100, color="#f59e0b", marker="s", markersize=3, label="普通 DQN")
        axes[row, 1].set_ylim(-2, 102)
        axes[row, 1].set_title(f"{TITLES[env]} · 贪心评估成功率")
        axes[row, 1].set_xlabel("回合")
        axes[row, 1].set_ylabel("%")
        axes[row, 1].legend(frameon=False)
        axes[row, 1].grid(alpha=0.25)

    fig.suptitle("RoboVision · DQN 训练与评估汇总", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "training_dashboard.png")
    plt.close(fig)


def generate_gif(env: str, step: int, window: int) -> None:
    data = load_metrics(OUT / f"{env}_metrics.csv")
    episodes = data["episode"]
    returns = data["return"]
    x_all = np.asarray(episodes)
    y_all = np.asarray(returns)
    low = min(-5.0, float(y_all.min()) * 1.05)
    high = max(5.0, float(y_all.max()) * 1.05)
    frames = []

    for end in list(range(step, len(episodes) + 1, step)) + [len(episodes)]:
        end = min(end, len(episodes))
        x = x_all[:end]
        y = y_all[:end]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
        ax.plot(x, y, color="#b8c7d9", alpha=0.45, linewidth=0.7, label="每回合回报")
        if len(y) >= window:
            sx, sy = smooth(y.tolist(), window)
            ax.plot(sx, sy, color="#2f7ed8", linewidth=2, label=f"滑动平均({window})")
        ax.set_xlim(1, len(episodes))
        ax.set_ylim(low, high)
        ax.set_xlabel("回合")
        ax.set_ylabel("累计奖励")
        ax.set_title(f"{TITLES[env]} · 训练过程（第 {episodes[end - 1]} 回合）")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        frames.append(image)
        plt.close(fig)

    if frames:
        frames[0].save(
            OUT / f"{env}_training_progress.gif",
            save_all=True,
            append_images=frames[1:],
            duration=220,
            loop=0,
            optimize=True,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in COPY_FILES.items():
        source = PROJECT_ROOT / src
        if source.exists():
            shutil.copy2(source, OUT / dst)
        else:
            print(f"[跳过] 缺少 {src}")
    generate_dashboard()
    generate_gif("robotnav", step=50, window=50)
    generate_gif("cartpole", step=20, window=25)
    print(f"训练材料已生成：{OUT}")


if __name__ == "__main__":
    main()
