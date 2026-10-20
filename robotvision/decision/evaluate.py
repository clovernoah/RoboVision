"""策略评估：成功率、回报、步数与回合 GIF。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import display_path
from .agents import AStarAgent, HeuristicCartPoleAgent, RandomAgent
from .dqn import DQNAgent
from .envs import CartPoleEnv, RobotNavEnv

def _make_env(env_name: str, max_steps: int | None = None):
    if env_name == "cartpole":
        return CartPoleEnv(max_steps=max_steps or 500)
    if env_name == "robotnav":
        return RobotNavEnv(max_steps=max_steps or 100)
    raise ValueError(f"未知环境: {env_name}")


def _make_policy(env_name: str, policy: str, agent: DQNAgent | None, env, seed: int):
    if policy == "dqn":
        if agent is None:
            raise ValueError("policy=dqn 时必须提供 agent")
        return lambda obs: agent.act(obs, greedy=True)
    if policy == "random":
        random_agent = RandomAgent(env.action_dim, seed=seed)
        return lambda obs: random_agent.act(obs)
    if policy == "astar":
        if env_name != "robotnav":
            raise ValueError("A* 仅适用于 robotnav")
        astar = AStarAgent()
        return lambda obs: astar.act(obs, env)
    if policy == "heuristic":
        if env_name != "cartpole":
            raise ValueError("启发式策略仅适用于 cartpole")
        heuristic = HeuristicCartPoleAgent()
        return lambda obs: heuristic.act(obs)
    raise ValueError(f"未知策略: {policy}")


def _save_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 60, max_width: int = 480) -> Path | None:
    if not frames:
        return None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None

    prepared = []
    for frame in frames:
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        prepared.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    path.parent.mkdir(parents=True, exist_ok=True)
    prepared[0].save(
        path,
        save_all=True,
        append_images=prepared[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return path


def evaluate_policy(
    env_name: str,
    policy: str = "dqn",
    agent: DQNAgent | None = None,
    episodes: int = 50,
    seed_base: int = 20_000,
    max_steps: int | None = None,
    gif_path: str | Path | None = None,
    gif_episode: int = 0,
    record_gif: bool = True,
    gif_stride: int = 1,
    gif_max_width: int = 480,
) -> dict[str, Any]:
    """评估指定策略，返回聚合指标与逐回合明细。"""
    env = _make_env(env_name, max_steps=max_steps)
    policy_fn = _make_policy(env_name, policy, agent, env, seed_base)

    returns: list[float] = []
    steps_list: list[int] = []
    successes: list[bool] = []
    collisions: list[bool] = []
    timeouts: list[bool] = []
    details: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []

    start = time.perf_counter()
    for index in range(episodes):
        seed = seed_base + index
        obs, _ = env.reset(seed=seed)
        done = False
        total_return = 0.0
        info: dict[str, Any] = {}
        episode_frames: list[np.ndarray] = []
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_return += float(reward)
            if record_gif and index == gif_episode and env.steps % max(1, gif_stride) == 0:
                episode_frames.append(env.render())

        if env_name == "cartpole":
            success = env.steps >= 475
            collision = False
            timeout = bool(truncated)
        else:
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = bool(truncated)

        returns.append(total_return)
        steps_list.append(env.steps)
        successes.append(success)
        collisions.append(collision)
        timeouts.append(timeout)
        details.append(
            {
                "seed": seed,
                "return": round(total_return, 3),
                "steps": env.steps,
                "success": success,
                "collision": collision,
            }
        )
        if index == gif_episode:
            frames = episode_frames

    elapsed = time.perf_counter() - start
    gif_saved = None
    if gif_path and frames:
        saved = _save_gif(frames, Path(gif_path), max_width=gif_max_width)
        gif_saved = str(saved) if saved else None

    return {
        "env": env_name,
        "policy": policy,
        "episodes": int(episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if timeouts else 0.0,
        "elapsed_s": round(elapsed, 3),
        "gif": gif_saved,
        "details": details,
    }


def evaluate_checkpoint(
    env_name: str,
    checkpoint: str | Path,
    episodes: int = 100,
    out_dir: str | Path | None = None,
    device: str = "cpu",
    gif: bool = True,
    gif_stride: int = 1,
    gif_max_width: int = 480,
) -> dict[str, Any]:
    """加载检查点并在同一组随机种子上评估 DQN。"""
    agent = DQNAgent.load(checkpoint, device=device)
    root = Path(out_dir) if out_dir else Path(checkpoint).parent
    result = evaluate_policy(
        env_name=env_name,
        policy="dqn",
        agent=agent,
        episodes=episodes,
        gif_path=(root / "episode.gif") if gif else None,
        record_gif=gif,
        gif_stride=gif_stride,
        gif_max_width=gif_max_width,
    )
    result["checkpoint"] = display_path(checkpoint)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    (root / "eval.json").write_text(payload, encoding="utf-8")
    return result
