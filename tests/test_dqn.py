"""DQN 组件测试。"""

import numpy as np
import torch

from robotvision.decision.dqn import DQNAgent, QNetwork, ReplayBuffer


def test_replay_buffer_capacity_and_sample():
    buffer = ReplayBuffer(capacity=5, obs_dim=3, seed=0)
    for i in range(8):
        buffer.push(np.full(3, i), i % 2, float(i), np.full(3, i + 1), i == 7)
    assert len(buffer) == 5
    sample = buffer.sample(4)
    assert sample["obs"].shape == (4, 3)
    assert sample["actions"].shape == (4,)
    assert set(np.unique(sample["actions"])).issubset({0, 1})


def test_q_network_forward():
    net = QNetwork(11, 4, hidden=(32, 32))
    output = net(torch.zeros(2, 11))
    assert output.shape == (2, 4)


def test_agent_update_and_target_sync():
    agent = DQNAgent(5, 3, hidden=(16,), batch_size=8, buffer_size=100, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=5).astype(np.float32)
        next_obs = rng.normal(size=5).astype(np.float32)
        agent.push(obs, int(rng.integers(3)), float(rng.normal()), next_obs, bool(rng.random() < 0.2))
    loss = agent.update()
    assert loss is not None and loss >= 0
    assert agent.update_steps == 1
    with torch.no_grad():
        next(agent.online.parameters()).add_(0.5)
    agent.sync_target()
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(online, target)


def test_agent_save_load_roundtrip(tmp_path):
    agent = DQNAgent(11, 4, hidden=(32, 32), seed=2)
    obs = np.linspace(-1, 1, 11, dtype=np.float32)
    q_before = agent.q_values(obs)
    path = agent.save(tmp_path / "agent.pt", meta={"env": "robotnav"})
    loaded = DQNAgent.load(path)
    assert np.allclose(q_before, loaded.q_values(obs), atol=1e-6)
    assert loaded.obs_dim == 11
    assert loaded.n_actions == 4


def test_greedy_action_matches_argmax():
    agent = DQNAgent(4, 3, seed=3)
    obs = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    q = agent.q_values(obs)
    assert agent.act(obs, greedy=True) == int(np.argmax(q))
