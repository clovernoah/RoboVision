"""Double DQN 的从零实现：Q 网络、经验回放、智能体与训练更新。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """两层 MLP 的 Q 值网络。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers.extend([nn.Linear(last, width), nn.ReLU()])
            last = width
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """固定容量的经验回放池（环形数组）。"""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """Double DQN 智能体。

    Double DQN 用在线网络选动作、目标网络估值，缓解 Q 值过估计：
        y = r + γ * Q_target(s', argmax_a Q_online(s', a))
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update: int = 200,
        double_dqn: bool = True,
        grad_clip: float = 10.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = tuple(int(v) for v in hidden)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.double_dqn = bool(double_dqn)
        self.grad_clip = float(grad_clip)
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.online = QNetwork(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.target = QNetwork(self.obs_dim, self.n_actions, self.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(lr))
        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, seed=seed)
        self.epsilon = 1.0
        self.update_steps = 0
        self.last_loss: float | None = None

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(tensor).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, epsilon: float | None = None, greedy: bool = False) -> int:
        eps = 0.0 if greedy else (self.epsilon if epsilon is None else float(epsilon))
        if np.random.random() < eps:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------
    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.online(obs).gather(1, actions)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions)
            else:
                next_q = self.target(next_obs).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.last_loss = float(loss.item())
        return self.last_loss

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden": list(self.hidden),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_steps": self.update_steps,
                "meta": meta or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", **overrides: Any) -> "DQNAgent":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"未找到策略检查点：{path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            n_actions=int(checkpoint["n_actions"]),
            hidden=tuple(checkpoint.get("hidden", (128, 128))),
            device=device,
            double_dqn=bool(checkpoint.get("meta", {}).get("double_dqn", True)),
            **overrides,
        )
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.update_steps = int(checkpoint.get("update_steps", 0))
        agent.epsilon = 0.0
        return agent
