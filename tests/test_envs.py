"""环境动力学与奖励测试。"""

import numpy as np

from robotvision.decision.agents import AStarAgent
from robotvision.decision.envs import CartPoleEnv, RobotNavEnv, astar_path
from robotvision.decision.envs.gridworld import ACTIONS


def test_cartpole_reset_and_step():
    env = CartPoleEnv(max_steps=100, seed=1)
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert info["steps"] == 0
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert next_obs.shape == (4,)
    assert reward == 1.0
    assert isinstance(terminated, bool)
    assert not truncated


def test_cartpole_truncates_at_max_steps():
    env = CartPoleEnv(max_steps=3, seed=2)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps <= 3
    assert env.steps == 3


def test_cartpole_terminates_out_of_bounds():
    env = CartPoleEnv(max_steps=100, seed=3)
    env.reset()
    env.state = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(0)
    assert terminated


def test_robotnav_layout_is_deterministic():
    a = RobotNavEnv(seed=7)
    b = RobotNavEnv(seed=7)
    a.reset(seed=7)
    b.reset(seed=7)
    assert a.start == b.start
    assert a.goal == b.goal
    assert a.static == b.static
    assert a.dynamic == b.dynamic
    assert a._observation().shape == (11,)


def test_robotnav_goal_reward():
    env = RobotNavEnv(grid_size=5, obstacle_density=0.0, n_dynamic=0, max_steps=30, seed=4)
    env.reset(seed=4)
    goal = env.goal
    for action, (dr, dc) in ACTIONS.items():
        candidate = (goal[0] - dr, goal[1] - dc)
        if 0 <= candidate[0] < env.size and 0 <= candidate[1] < env.size:
            env.pos = candidate
            env.trail = [candidate]
            env.steps = 0
            env.done = False
            _, reward, terminated, truncated, info = env.step(action)
            assert info["success"] is True
            assert terminated is True
            assert reward > 40
            return
    raise AssertionError("未找到目标相邻格")


def test_robotnav_blocked_action_keeps_position():
    env = RobotNavEnv(grid_size=5, obstacle_density=0.0, n_dynamic=0, max_steps=30, seed=5)
    env.reset(seed=5)
    env.goal = (4, 4)
    env.pos = (0, 0)
    env.trail = [(0, 0)]
    env.dynamic = []
    env.static = {(0, 1)}
    _, reward, _, _, info = env.step(3)  # 向右被静态障碍挡住
    assert info["blocked"] is True
    assert env.pos == (0, 0)
    assert reward < 0


def test_astar_avoids_blocked_cells():
    blocked = {(1, 0), (1, 1), (1, 2), (1, 3)}
    path = astar_path(5, blocked, (0, 0), (4, 0))
    assert path[0] == (0, 0)
    assert path[-1] == (4, 0)
    assert not (set(path) & blocked)
    assert len(path) == 13  # 需要绕行第 1 行的连续障碍


def test_astar_agent_returns_valid_action():
    env = RobotNavEnv(seed=11)
    env.reset(seed=11)
    agent = AStarAgent()
    action = agent.act(env._observation(), env)
    assert action in ACTIONS
