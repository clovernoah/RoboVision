"""评估训练好的 DQN，并与随机 / 启发式 / A* 基线对比（相同随机种子）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robotvision.config import display_path  # noqa: E402
from robotvision.decision.dqn import DQNAgent  # noqa: E402
from robotvision.decision.evaluate import evaluate_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 DQN 与基线策略")
    parser.add_argument("--env", choices=["cartpole", "robotnav"], default="robotnav")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=20_000)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint or str(PROJECT_ROOT / "runs" / args.env / "best.pt")
    out_dir = Path(args.out) if args.out else Path(checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    try:
        agent = DQNAgent.load(checkpoint, device=args.device)
    except FileNotFoundError:
        print(f"[错误] 未找到检查点：{checkpoint}\n请先运行 scripts/train_dqn.py")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 加载检查点失败：{exc}")
        return 1

    results["dqn"] = evaluate_policy(
        env_name=args.env,
        policy="dqn",
        agent=agent,
        episodes=args.episodes,
        seed_base=args.seed_base,
        gif_path=None if args.no_gif else out_dir / "episode.gif",
        record_gif=not args.no_gif,
    )

    if not args.no_baselines:
        results["random"] = evaluate_policy(
            env_name=args.env,
            policy="random",
            episodes=args.episodes,
            seed_base=args.seed_base,
            record_gif=False,
        )
        baseline = "astar" if args.env == "robotnav" else "heuristic"
        results[baseline] = evaluate_policy(
            env_name=args.env,
            policy=baseline,
            episodes=args.episodes,
            seed_base=args.seed_base,
            record_gif=False,
        )

    payload = {
        "env": args.env,
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "checkpoint": display_path(checkpoint),
        "results": results,
    }
    (out_dir / "eval_all.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'策略':<10}{'成功率':>10}{'平均回报':>12}{'平均步数':>12}{'碰撞率':>10}")
    print("-" * 56)
    for name, result in results.items():
        print(
            f"{name:<10}{result['success_rate']:>10.1%}{result['mean_return']:>12.2f}"
            f"{result['mean_steps']:>12.1f}{result['collision_rate']:>10.1%}"
        )
    print(f"\n结果已保存：{display_path(out_dir / 'eval_all.json')}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
