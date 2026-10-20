"""机器人栅格导航环境：局部射线感知 + 动态障碍 + A* 基线。"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any

import cv2
import numpy as np

# 动作：上、下、左、右
ACTIONS: dict[int, tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "上", 1: "下", 2: "左", 3: "右"}
ACTION_ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}

# 8 向射线：上下左右 + 四个对角
RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _neighbors(node: tuple[int, int], size: int):
    r, c = node
    for dr, dc in ACTIONS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def astar_path(
    size: int,
    blocked: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """标准 A*（4 邻域，曼哈顿启发式），返回包含起终点的路径。"""
    if start == goal:
        return [start]
    if goal in blocked:
        return []

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0, 0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        if cost > g_score.get(current, 1 << 30):
            continue
        for nxt in _neighbors(current, size):
            if nxt in blocked:
                continue
            new_cost = cost + 1
            if new_cost < g_score.get(nxt, 1 << 30):
                g_score[nxt] = new_cost
                came_from[nxt] = current
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(open_heap, (new_cost + heuristic, new_cost, nxt))
    return []


def _path_exists(size: int, blocked: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if start in blocked or goal in blocked:
        return False
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for nxt in _neighbors(node, size):
            if nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


class RobotNavEnv:
    """服务机器人局部导航任务。

    机器人只能看到 8 个方向、有限距离内的障碍（类似 2D 激光雷达），
    需要避开静态障碍与随机游走的动态障碍，尽快到达目标点。

    观测（11 维）：
      - 8 个方向归一化射线距离
      - 目标相对位置 (dr/n, dc/n)
      - 归一化曼哈顿距离
    动作：0 上 / 1 下 / 2 左 / 3 右
    """

    observation_dim = 11
    action_dim = 4

    def __init__(
        self,
        grid_size: int = 10,
        obstacle_density: float = 0.12,
        n_dynamic: int = 2,
        max_steps: int = 100,
        ray_range: int = 6,
        seed: int | None = None,
        dynamic_randomness: float = 0.35,
    ) -> None:
        self.size = int(grid_size)
        self.obstacle_density = float(obstacle_density)
        self.n_dynamic = int(n_dynamic)
        self.max_steps = int(max_steps)
        self.ray_range = int(ray_range)
        self.dynamic_randomness = float(dynamic_randomness)
        self.rng = np.random.default_rng(seed)

        self.static: set[tuple[int, int]] = set()
        self.dynamic: list[tuple[int, int]] = []
        self.start: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (self.size - 1, self.size - 1)
        self.pos: tuple[int, int] = self.start
        self.trail: list[tuple[int, int]] = [self.start]
        self.steps = 0
        self.done = False
        self.last_action = 0
        self.last_reward = 0.0
        self.last_info: dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------------
    # 布局生成
    # ------------------------------------------------------------------
    def _sample_layout(self) -> None:
        n = self.size
        all_cells = [(r, c) for r in range(n) for c in range(n)]
        target_static = int(round(self.obstacle_density * n * n))

        for _ in range(300):
            start = all_cells[int(self.rng.integers(len(all_cells)))]
            candidates = [
                cell
                for cell in all_cells
                if abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= max(3, n // 2)
            ]
            if not candidates:
                continue
            goal = candidates[int(self.rng.integers(len(candidates)))]

            protected = {start, goal}
            for r in range(start[0] - 1, start[0] + 2):
                for c in range(start[1] - 1, start[1] + 2):
                    if 0 <= r < n and 0 <= c < n:
                        protected.add((r, c))
            pool = [cell for cell in all_cells if cell not in protected]
            if len(pool) < target_static:
                continue
            indices = self.rng.choice(len(pool), size=target_static, replace=False)
            static = {pool[int(i)] for i in indices}

            if _path_exists(n, static, start, goal):
                self.start, self.goal, self.static = start, goal, static
                return
        # 极端参数下的兜底：无静态障碍
        self.static = set()
        self.start = (0, 0)
        self.goal = (n - 1, n - 1)

    def _sample_dynamic(self) -> None:
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in self.static and (r, c) != self.start and (r, c) != self.goal
        ]
        self.rng.shuffle(free)
        self.dynamic = free[: self.n_dynamic]

    def _move_dynamic(self) -> None:
        n = self.size
        occupied = set(self.dynamic)
        new_positions: list[tuple[int, int]] = []
        for obstacle in self.dynamic:
            occupied.discard(obstacle)
            options = [obstacle]
            for dr, dc in ACTIONS.values():
                nr, nc = obstacle[0] + dr, obstacle[1] + dc
                if (
                    0 <= nr < n
                    and 0 <= nc < n
                    and (nr, nc) not in self.static
                    and (nr, nc) not in occupied
                ):
                    options.append((nr, nc))
            if self.rng.random() < self.dynamic_randomness and len(options) > 1:
                choice = options[int(self.rng.integers(1, len(options)))]
            else:
                choice = obstacle
            new_positions.append(choice)
            occupied.add(choice)
        self.dynamic = new_positions

    # ------------------------------------------------------------------
    # Gym 风格接口
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, **_: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample_layout()
        self._sample_dynamic()
        self.pos = self.start
        self.trail = [self.start]
        self.steps = 0
        self.done = False
        self.last_action = 0
        self.last_reward = 0.0
        self.last_info = {}
        return self._observation(), self.state_dict()

    def _observation(self) -> np.ndarray:
        n = self.size
        rays: list[float] = []
        for dr, dc in RAY_DIRECTIONS:
            distance = 0
            r, c = self.pos
            for step in range(1, self.ray_range + 1):
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < n and 0 <= nc < n):
                    distance = step
                    break
                if (nr, nc) in self.static or (nr, nc) in self.dynamic:
                    distance = step
                    break
                distance = step
            rays.append(distance / self.ray_range)

        dr = (self.goal[0] - self.pos[0]) / n
        dc = (self.goal[1] - self.pos[1]) / n
        manhattan = (abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])) / (2 * n)
        return np.array(rays + [dr, dc, manhattan], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("当前回合已结束，请先调用 reset()")
        action = int(action)
        if action not in ACTIONS:
            raise ValueError("动作必须为 0..3")

        n = self.size
        self.steps += 1
        self.last_action = action
        dr, dc = ACTIONS[action]
        target = (self.pos[0] + dr, self.pos[1] + dc)

        old_distance = abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])
        reward = -0.03
        terminated = False
        truncated = False
        info = {"success": False, "collision": False, "blocked": False, "steps": self.steps}

        if not (0 <= target[0] < n and 0 <= target[1] < n) or target in self.static:
            reward -= 0.4
            info["blocked"] = True
        else:
            self.pos = target
            self.trail.append(target)

        # 动态障碍先移动，再判定碰撞
        self._move_dynamic()
        if self.pos in self.dynamic:
            reward = -10.0
            terminated = True
            info["collision"] = True
        elif self.pos == self.goal:
            reward = 50.0
            terminated = True
            info["success"] = True
        else:
            new_distance = abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])
            reward += 0.8 * (old_distance - new_distance)
            if self.steps >= self.max_steps:
                truncated = True
                reward -= 3.0

        if len(self.trail) > self.max_steps + 2:
            self.trail = self.trail[-self.max_steps - 2 :]

        self.done = bool(terminated or truncated)
        self.last_reward = float(reward)
        self.last_info = info
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # 渲染与状态
    # ------------------------------------------------------------------
    def render(self, cell: int = 44, show_rays: bool = False) -> np.ndarray:
        """渲染栅格环境，返回 BGR 图像。"""
        n = self.size
        pad = 10
        panel_h = 62
        width = n * cell + pad * 2
        height = n * cell + pad * 2 + panel_h
        canvas = np.full((height, width, 3), (28, 30, 36), dtype=np.uint8)

        def to_px(node: tuple[int, int]) -> tuple[int, int]:
            return pad + node[1] * cell, pad + node[0] * cell

        # 网格
        for i in range(n + 1):
            x = pad + i * cell
            y = pad + i * cell
            cv2.line(canvas, (x, pad), (x, pad + n * cell), (48, 52, 60), 1)
            cv2.line(canvas, (pad, y), (pad + n * cell, y), (48, 52, 60), 1)

        # 静态障碍
        for r, c in self.static:
            x, y = to_px((r, c))
            cv2.rectangle(canvas, (x + 3, y + 3), (x + cell - 3, y + cell - 3), (70, 76, 88), -1)

        # 轨迹
        for r, c in self.trail[:-1]:
            x, y = to_px((r, c))
            cv2.circle(canvas, (x + cell // 2, y + cell // 2), 3, (110, 90, 60), -1)

        # 目标
        gx, gy = to_px(self.goal)
        cv2.rectangle(canvas, (gx + 6, gy + 6), (gx + cell - 6, gy + cell - 6), (90, 200, 110), -1)
        cv2.putText(
            canvas, "G", (gx + cell // 3, gy + int(cell * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (20, 40, 25), 2, cv2.LINE_AA,
        )

        # 动态障碍
        for r, c in self.dynamic:
            x, y = to_px((r, c))
            cv2.circle(canvas, (x + cell // 2, y + cell // 2), cell // 3, (60, 90, 230), -1, cv2.LINE_AA)

        # 射线
        if show_rays:
            obs = self._observation()[:8]
            center = (to_px(self.pos)[0] + cell // 2, to_px(self.pos)[1] + cell // 2)
            overlay = canvas.copy()
            for value, (dr, dc) in zip(obs, RAY_DIRECTIONS):
                end = (
                    int(center[0] + dc * value * self.ray_range * cell),
                    int(center[1] + dr * value * self.ray_range * cell),
                )
                cv2.line(overlay, center, end, (70, 150, 110), 1)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # 机器人
        rx, ry = to_px(self.pos)
        center = (rx + cell // 2, ry + cell // 2)
        cv2.circle(canvas, center, cell // 3, (230, 150, 60), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, ACTION_ARROWS.get(self.last_action, "?"),
            (rx + cell // 3, ry + int(cell * 0.66)), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (25, 30, 40), 2, cv2.LINE_AA,
        )

        # 底部信息面板
        panel_y = pad + n * cell + 8
        status = "DONE" if self.done else "RUNNING"
        if self.last_info.get("success"):
            status = "SUCCESS"
        elif self.last_info.get("collision"):
            status = "COLLISION"
        info = f"{status} | step {self.steps}/{self.max_steps} | reward {self.last_reward:+.2f}"
        cv2.putText(canvas, info, (pad, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 214, 222), 1, cv2.LINE_AA)
        dist = abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])
        cv2.putText(
            canvas, f"distance to goal: {dist}", (pad, panel_y + 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 156, 168), 1, cv2.LINE_AA,
        )
        return canvas

    def state_dict(self) -> dict[str, Any]:
        """面向 Web 的状态描述。"""
        return {
            "type": "robotnav",
            "grid_size": self.size,
            "static": [list(map(int, cell)) for cell in sorted(self.static)],
            "dynamic": [list(map(int, cell)) for cell in self.dynamic],
            "robot": list(map(int, self.pos)),
            "goal": list(map(int, self.goal)),
            "start": list(map(int, self.start)),
            "trail": [list(map(int, cell)) for cell in self.trail[-60:]],
            "steps": self.steps,
            "max_steps": self.max_steps,
            "done": self.done,
            "last_reward": round(self.last_reward, 3),
            "last_action": self.last_action,
            "success": bool(self.last_info.get("success", False)),
            "collision": bool(self.last_info.get("collision", False)),
            "distance": abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1]),
            "observation": [round(float(v), 4) for v in self._observation()],
        }

    def action_masks(self) -> list[bool]:
        """返回 4 个动作是否可执行（用于展示）。"""
        masks = []
        for dr, dc in ACTIONS.values():
            r, c = self.pos[0] + dr, self.pos[1] + dc
            masks.append(0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.static)
        return masks
