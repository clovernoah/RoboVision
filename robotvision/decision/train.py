"""DQN 训练入口：日志、检查点、评估与训练曲线。"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT, display_path
from .dqn import DQNAgent
from .envs import CartPoleEnv, RobotNavEnv
from .evaluate import evaluate_policy

ENV_BUILDERS = {"cartpole": CartPoleEnv, "robotnav": RobotNavEnv}
ENV_TITLES = {"cartpole": "CartPole", "robotnav": "机器人栅格导航"}


def make_env(env_name: str, max_steps: int | None = None, seed: int | None = None):
    """按名称创建环境。"""
    if env_name == "cartpole":
        return CartPoleEnv(max_steps=max_steps or 500, seed=seed)
    if env_name == "robotnav":
        return RobotNavEnv(max_steps=max_steps or 100, seed=seed)
    raise ValueError(f"未知环境: {env_name}")


def episode_metrics(env_name: str, env, total_return: float, steps: int, info: dict[str, Any]) -> dict[str, Any]:
    if env_name == "cartpole":
        success = steps >= 475
        collision = False
    else:
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
    return {
        "return": round(float(total_return), 3),
        "steps": int(steps),
        "success": bool(success),
        "collision": collision,
    }


def train(
    env_name: str,
    episodes: int = 800,
    seed: int = 42,
    out_dir: str | Path | None = None,
    max_steps: int | None = None,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_size: int = 50_000,
    target_update: int = 200,
    double_dqn: bool = True,
    hidden: tuple[int, ...] = (128, 128),
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.6,
    eval_every: int = 50,
    eval_episodes: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """训练 Double DQN 并返回摘要。

    训练过程每回合记录指标到 ``metrics.csv``，定期进行贪心评估，
    按评估成功率保存 ``best.pt``，结束时保存 ``last.pt`` 与训练曲线。
    """
    if env_name not in ENV_BUILDERS:
        raise ValueError(f"未知环境: {env_name}，可选 {list(ENV_BUILDERS)}")

    root = Path(out_dir) if out_dir else PROJECT_ROOT / "runs" / env_name
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.csv"
    best_path = root / "best.pt"
    last_path = root / "last.pt"

    env = make_env(env_name, max_steps=max_steps, seed=seed)
    agent = DQNAgent(
        obs_dim=env.observation_dim,
        n_actions=env.action_dim,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update=target_update,
        double_dqn=double_dqn,
        device=device,
        seed=seed,
    )

    fieldnames = [
        "episode", "return", "steps", "success", "collision", "epsilon",
        "loss", "eval_success", "eval_return", "eval_steps", "elapsed_s",
    ]
    best_score = -1e9
    best_episode = -1
    start_time = time.time()
    decay_episodes = max(1, int(episodes * epsilon_decay_frac))

    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode - 1) / decay_episodes)
            agent.epsilon = epsilon
            total_return = 0.0
            losses: list[float] = []
            done = False
            info: dict[str, Any] = {}

            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                agent.push(obs, action, reward, next_obs, done)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                obs = next_obs
                total_return += float(reward)

            metrics = episode_metrics(env_name, env, total_return, env.steps, info)
            row = {
                "episode": episode,
                "return": metrics["return"],
                "steps": metrics["steps"],
                "success": int(metrics["success"]),
                "collision": int(metrics["collision"]),
                "epsilon": round(epsilon, 4),
                "loss": round(float(np.mean(losses)) if losses else 0.0, 5),
                "eval_success": "",
                "eval_return": "",
                "eval_steps": "",
                "elapsed_s": round(time.time() - start_time, 1),
            }

            if episode % eval_every == 0 or episode == episodes:
                evaluation = evaluate_policy(
                    env_name=env_name,
                    policy="dqn",
                    agent=agent,
                    episodes=eval_episodes,
                    seed_base=10_000 + episode * 100,
                    max_steps=max_steps,
                )
                row.update(
                    {
                        "eval_success": round(evaluation["success_rate"], 4),
                        "eval_return": round(evaluation["mean_return"], 3),
                        "eval_steps": round(evaluation["mean_steps"], 2),
                    }
                )
                score = evaluation["success_rate"] * 1000 + evaluation["mean_return"]
                if score > best_score:
                    best_score = score
                    best_episode = episode
                    agent.save(
                        best_path,
                        meta={
                            "env": env_name,
                            "episode": episode,
                            "eval": evaluation,
                            "double_dqn": double_dqn,
                            "hidden": list(hidden),
                            "episodes": episodes,
                        },
                    )
                if verbose:
                    print(
                        f"[{env_name}] ep {episode:4d} | train_ret {metrics['return']:+8.2f} "
                        f"| steps {metrics['steps']:3d} | eps {epsilon:.3f} "
                        f"| eval_success {evaluation['success_rate']:.2%} "
                        f"| eval_ret {evaluation['mean_return']:+8.2f}",
                        flush=True,
                    )
            elif verbose and episode % 10 == 0:
                avg_return = total_return
                print(
                    f"[{env_name}] ep {episode:4d} | train_ret {avg_return:+8.2f} "
                    f"| steps {metrics['steps']:3d} | eps {epsilon:.3f} | loss {row['loss']:.4f}",
                    flush=True,
                )

            writer.writerow(row)
            fh.flush()

    agent.save(
        last_path,
        meta={"env": env_name, "episode": episodes, "double_dqn": double_dqn, "hidden": list(hidden)},
    )

    summary = {
        "env": env_name,
        "episodes": episodes,
        "best_episode": best_episode,
        "best_score": best_score,
        "checkpoint": display_path(best_path),
        "last_checkpoint": display_path(last_path),
        "metrics_csv": display_path(metrics_path),
        "elapsed_s": round(time.time() - start_time, 1),
        "double_dqn": double_dqn,
        "hidden": list(hidden),
        "seed": seed,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_training(metrics_path, root / "training_curve.png", env_name)
    return summary


def plot_training(metrics_path: str | Path, output_path: str | Path, env_name: str) -> Path:
    """根据 metrics.csv 绘制回报与评估成功率曲线。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return Path(output_path)

    episodes, returns, successes, eval_eps, eval_success = [], [], [], [], []
    with Path(metrics_path).open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))
            successes.append(int(row["success"]))
            if row["eval_success"]:
                eval_eps.append(int(row["episode"]))
                eval_success.append(float(row["eval_success"]))

    if not episodes:
        return Path(output_path)

    window = max(5, min(50, len(returns) // 10 or 5))
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    smooth_x = episodes[window - 1 :]

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    axes[0].plot(episodes, returns, color="#9fb4cc", alpha=0.35, linewidth=0.8, label="每回合回报")
    axes[0].plot(smooth_x, smooth, color="#2f7ed8", linewidth=2.0, label=f"滑动平均({window})")
    axes[0].set_xlabel("回合")
    axes[0].set_ylabel("累计奖励")
    axes[0].set_title(f"{ENV_TITLES.get(env_name, env_name)} 训练回报")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    if eval_eps:
        axes[1].plot(eval_eps, eval_success, color="#2aa876", linewidth=2.0, marker="o", markersize=3)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("回合")
    axes[1].set_ylabel("贪心策略成功率")
    axes[1].set_title(f"{ENV_TITLES.get(env_name, env_name)} 定期评估")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output
