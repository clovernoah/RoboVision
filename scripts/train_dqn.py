"""训练 DQN/Double DQN。

示例：
    python scripts/train_dqn.py --env cartpole --episodes 500
    python scripts/train_dqn.py --env robotnav --episodes 1000
    python scripts/train_dqn.py --env robotnav --episodes 800 --no-double
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robotvision.decision.train import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 DQN/Double DQN")
    parser.add_argument("--env", choices=["cartpole", "robotnav"], default="robotnav")
    parser.add_argument("--episodes", type=int, default=None, help="默认 cartpole=500, robotnav=1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-frac", type=float, default=0.6)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-double", action="store_true", help="关闭 Double DQN，使用普通 DQN")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes = args.episodes if args.episodes is not None else (500 if args.env == "cartpole" else 1000)
    summary = train(
        env_name=args.env,
        episodes=episodes,
        seed=args.seed,
        out_dir=args.out,
        max_steps=args.max_steps,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update=args.target_update,
        double_dqn=not args.no_double,
        hidden=tuple(args.hidden),
        epsilon_end=args.epsilon_end,
        epsilon_decay_frac=args.epsilon_decay_frac,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        device=args.device,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
