"""CartPole 平衡环境（按经典动力学公式自实现，不依赖 gym）。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


class CartPoleEnv:
    """小车-倒立摆平衡任务。

    观测：``[x, x_dot, theta, theta_dot]``
    动作：0 向左施力，1 向右施力
    奖励：每存活一步 +1，杆倾角或小车位置越界即终止。
    """

    observation_dim = 4
    action_dim = 2

    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    total_mass = mass_cart + mass_pole
    half_length = 0.5
    pole_mass_length = mass_pole * half_length
    force_mag = 10.0
    tau = 0.02
    theta_threshold = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    def __init__(self, max_steps: int = 500, seed: int | None = None) -> None:
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.done = False

    def reset(self, seed: int | None = None, **_: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        self.done = False
        return self.state.copy(), {"steps": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("当前回合已结束，请先调用 reset()")
        action = int(action)
        if action not in (0, 1):
            raise ValueError("CartPole 动作必须为 0 或 1")

        x, x_dot, theta, theta_dot = (float(v) for v in self.state)
        force = self.force_mag if action == 1 else -self.force_mag
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)

        temp = (force + self.pole_mass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.half_length * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1
        self.done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold
            or self.steps >= self.max_steps
        )
        # 速度/角度越大惩罚越小，保持与原版 CartPole 一致的稀疏奖励
        reward = 1.0
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        info = {"steps": self.steps, "x": x, "theta": theta}
        return self.state.copy(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        """与 RobotNavEnv 统一的观测访问接口。"""
        return self.state.copy()

    def render(self, size: tuple[int, int] = (640, 420)) -> np.ndarray:
        """渲染当前状态，返回 BGR 图像。"""
        width, height = size
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        horizon = int(height * 0.78)
        cv2.line(canvas, (0, horizon), (width, horizon), (120, 120, 120), 2)
        cv2.putText(canvas, "CartPole", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

        x, _, theta, _ = (float(v) for v in self.state)
        scale = 90.0
        cart_x = width / 2 + x * scale
        cart_y = horizon - 26
        cart_w, cart_h = 96, 42

        # 轨道范围提示
        left_track = width / 2 - self.x_threshold * scale
        right_track = width / 2 + self.x_threshold * scale
        cv2.line(canvas, (int(left_track), horizon + 4), (int(right_track), horizon + 4), (80, 80, 80), 3)
        cv2.circle(canvas, (int(left_track), horizon + 4), 4, (90, 90, 220), -1)
        cv2.circle(canvas, (int(right_track), horizon + 4), 4, (90, 90, 220), -1)

        # 小车
        cv2.rectangle(
            canvas,
            (int(cart_x - cart_w / 2), int(cart_y - cart_h)),
            (int(cart_x + cart_w / 2), int(cart_y)),
            (80, 120, 200),
            -1,
        )
        cv2.circle(canvas, (int(cart_x - 28), int(cart_y + 7)), 9, (50, 50, 50), -1)
        cv2.circle(canvas, (int(cart_x + 28), int(cart_y + 7)), 9, (50, 50, 50), -1)

        # 杆
        pole_len = 2 * self.half_length * scale
        pivot = (cart_x, cart_y - cart_h / 2)
        tip = (
            pivot[0] + math.sin(theta) * pole_len,
            pivot[1] - math.cos(theta) * pole_len,
        )
        cv2.line(canvas, (int(pivot[0]), int(pivot[1])), (int(tip[0]), int(tip[1])), (40, 60, 170), 10, cv2.LINE_AA)
        cv2.circle(canvas, (int(pivot[0]), int(pivot[1])), 5, (30, 30, 30), -1)
        cv2.circle(canvas, (int(tip[0]), int(tip[1])), 8, (40, 60, 170), -1)

        # 状态信息
        info = f"step {self.steps:3d}   x={x:+.2f}   theta={theta:+.3f} rad"
        cv2.putText(canvas, info, (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
        return canvas

    def state_dict(self) -> dict[str, Any]:
        """面向 Web 的轻量状态描述。"""
        return {
            "type": "cartpole",
            "x": round(float(self.state[0]), 4),
            "x_dot": round(float(self.state[1]), 4),
            "theta": round(float(self.state[2]), 4),
            "theta_dot": round(float(self.state[3]), 4),
            "steps": self.steps,
            "max_steps": self.max_steps,
            "done": self.done,
        }
