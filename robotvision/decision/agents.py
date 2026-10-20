"""基线策略：随机、CartPole 启发式与 A* 导航。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .envs.gridworld import ACTIONS, RobotNavEnv, astar_path


class RandomAgent:
    """均匀随机策略。"""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: Any = None) -> int:
        return int(self.rng.integers(self.n_actions))


class HeuristicCartPoleAgent:
    """经典 CartPole 启发式：根据角度与角速度方向施力。"""

    def act(self, obs: np.ndarray, env: Any = None) -> int:
        theta, theta_dot = float(obs[2]), float(obs[3])
        return 1 if theta + 0.5 * theta_dot > 0 else 0


class AStarAgent:
    """在已知静态地图上做 A*，并把动态障碍视为临时障碍。

    这是导航任务的**强基线**：每步重新规划，能利用全局地图信息；
    DQN 只能看到局部射线，因此比较时需说明信息差异。
    """

    def act(self, obs: np.ndarray, env: RobotNavEnv) -> int:
        if env is None:
            return 0
        blocked = set(env.static) | set(env.dynamic)
        path = astar_path(env.size, blocked, env.pos, env.goal)
        if len(path) >= 2:
            target = path[1]
        else:
            target = self._fallback(env)
        dr, dc = target[0] - env.pos[0], target[1] - env.pos[1]
        for action, (ar, ac) in ACTIONS.items():
            if (ar, ac) == (dr, dc):
                return action
        return self._fallback_action(env)

    def _fallback(self, env: RobotNavEnv) -> tuple[int, int]:
        n = env.size
        best = env.pos
        best_dist = 1 << 30
        for dr, dc in ACTIONS.values():
            nr, nc = env.pos[0] + dr, env.pos[1] + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr, nc) in env.static or (nr, nc) in env.dynamic:
                continue
            dist = abs(nr - env.goal[0]) + abs(nc - env.goal[1])
            if dist < best_dist:
                best_dist, best = dist, (nr, nc)
        return best

    def _fallback_action(self, env: RobotNavEnv) -> int:
        best_action, best_dist = 0, 1 << 30
        for action, (dr, dc) in ACTIONS.items():
            nr, nc = env.pos[0] + dr, env.pos[1] + dc
            if (
                0 <= nr < env.size
                and 0 <= nc < env.size
                and (nr, nc) not in env.static
                and (nr, nc) not in env.dynamic
            ):
                dist = abs(nr - env.goal[0]) + abs(nc - env.goal[1])
                if dist < best_dist:
                    best_dist, best_action = dist, action
        return best_action
