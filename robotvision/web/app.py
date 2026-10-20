"""Flask Web 演示：多任务感知分析 + 机器人 DQN 决策。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from .. import __version__
from ..config import PROJECT_ROOT, display_path, get, load_config, resolve_path
from ..decision.agents import AStarAgent, HeuristicCartPoleAgent, RandomAgent
from ..decision.dqn import DQNAgent
from ..decision.envs import CartPoleEnv, RobotNavEnv
from ..decision.envs.gridworld import ACTION_NAMES
from ..pipelines.perception import PerceptionPipeline
from ..utils import encode_base64, encode_image_bgr, open_video_source

MAX_SIDE = 1600
CARTPOLE_ACTION_NAMES = {0: "左推", 1: "右推"}


# ----------------------------------------------------------------------
# 决策会话
# ----------------------------------------------------------------------
class DecisionSession:
    """保存单个浏览器会话的环境与策略。

    课程演示采用单会话全局状态（加锁），多人同时访问会共享同一回合。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.env: RobotNavEnv | CartPoleEnv | None = None
        self.env_name = "robotnav"
        self.policy = "dqn"
        self.agent: DQNAgent | None = None
        self.random_agent: RandomAgent | None = None
        self.astar = AStarAgent()
        self.heuristic = HeuristicCartPoleAgent()
        self.total_return = 0.0
        self.episode_index = 0
        self.note = ""

    def _load_policy(self, env_name: str, policy: str) -> None:
        checkpoint = resolve_path(
            get(self.config, f"decision.{env_name}.checkpoint", f"runs/{env_name}/best.pt")
        )
        self.policy = policy
        self.note = ""
        self.agent = None
        self.random_agent = None

        if policy == "dqn":
            try:
                self.agent = DQNAgent.load(checkpoint, device=get(self.config, "decision.device", "cpu"))
            except Exception as exc:  # noqa: BLE001
                self.policy = "random"
                self.note = f"DQN 检查点不可用（{type(exc).__name__}），已回退为随机策略"
                self.random_agent = RandomAgent(self.env.action_dim, seed=int(time.time()) % 100000)
        elif policy == "random":
            self.random_agent = RandomAgent(self.env.action_dim, seed=int(time.time()) % 100000)
        elif policy == "astar" and env_name != "robotnav":
            self.policy = "random"
            self.note = "A* 仅适用于机器人导航，已回退为随机策略"
            self.random_agent = RandomAgent(self.env.action_dim, seed=int(time.time()) % 100000)
        elif policy == "heuristic" and env_name != "cartpole":
            self.policy = "random"
            self.note = "启发式策略仅适用于 CartPole，已回退为随机策略"
            self.random_agent = RandomAgent(self.env.action_dim, seed=int(time.time()) % 100000)

    def reset(self, env_name: str, policy: str, seed: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.env_name = env_name
            if env_name == "cartpole":
                self.env = CartPoleEnv(max_steps=int(get(self.config, "decision.cartpole.max_steps", 500)))
            else:
                self.env = RobotNavEnv(max_steps=int(get(self.config, "decision.robotnav.max_steps", 100)))
            self._load_policy(env_name, policy)
            obs, _ = self.env.reset(seed=seed)
            self.total_return = 0.0
            self.episode_index += 1
            return self._payload(action=None, q_values=self._q_values(obs))

    def _q_values(self, obs: np.ndarray) -> list[float] | None:
        if self.agent is None:
            return None
        return [round(float(v), 4) for v in self.agent.q_values(obs)]

    def _choose_action(self, auto: bool, manual_action: int | None) -> int:
        if self.env is None:
            raise RuntimeError("尚未开始回合")
        if not auto:
            if manual_action is None:
                raise ValueError("手动模式下必须提供 action")
            if not (0 <= int(manual_action) < self.env.action_dim):
                raise ValueError(f"动作超出范围：{manual_action}")
            return int(manual_action)

        if self.policy == "dqn" and self.agent is not None:
            return self.agent.act(self.env._observation(), greedy=True)
        if self.policy == "random" and self.random_agent is not None:
            return self.random_agent.act(self.env._observation())
        if self.policy == "astar" and self.env_name == "robotnav":
            return self.astar.act(self.env._observation(), self.env)
        if self.policy == "heuristic" and self.env_name == "cartpole":
            return self.heuristic.act(self.env._observation())
        raise ValueError("当前策略无法自动决策，请使用手动模式")

    def step(self, auto: bool = True, action: int | None = None) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                raise RuntimeError("尚未开始回合")
            if self.env.done:
                raise RuntimeError("当前回合已结束，请开始新回合")

            obs = self.env._observation()
            q_values = self._q_values(obs)
            chosen = self._choose_action(auto, action)
            _, reward, terminated, truncated, _ = self.env.step(chosen)
            self.total_return += float(reward)
            return self._payload(action=chosen, q_values=q_values, reward=reward)

    def _payload(
        self,
        action: int | None,
        q_values: list[float] | None,
        reward: float | None = None,
    ) -> dict[str, Any]:
        assert self.env is not None
        state = self.env.state_dict()
        frame = None
        if self.env_name == "cartpole":
            frame = encode_base64(self.env.render(), ".jpg")
        names = CARTPOLE_ACTION_NAMES if self.env_name == "cartpole" else ACTION_NAMES
        return {
            "state": state,
            "frame": frame,
            "action": action,
            "action_name": names.get(action, "-") if action is not None else "-",
            "reward": None if reward is None else round(float(reward), 3),
            "done": bool(self.env.done),
            "q_values": q_values,
            "policy": self.policy,
            "note": self.note,
            "total_return": round(self.total_return, 3),
            "episode": self.episode_index,
        }


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _prepare_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / longest
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _analysis_to_json(result: dict[str, Any], images: dict[str, np.ndarray]) -> dict[str, Any]:
    depth = result.get("depth")
    payload: dict[str, Any] = {
        "detections": [d.to_dict() for d in result.get("detections", [])],
        "faces": [f.to_dict() for f in result.get("faces", [])],
        "depth": None,
        "fusion": result.get("fusion"),
        "timings_ms": result.get("timings_ms", {}),
        "counts": {
            "person": sum(1 for d in result.get("detections", []) if d.class_name == "person"),
            "vehicle": sum(
                1
                for d in result.get("detections", [])
                if d.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}
            ),
            "face": len(result.get("faces", [])),
            "objects": len(result.get("detections", [])),
        },
    }
    if depth is not None:
        payload["depth"] = {
            "stats": depth.stats(),
            "size": [int(depth.depth.shape[1]), int(depth.depth.shape[0])],
        }
    payload["images"] = {
        name: encode_base64(image, ".jpg") for name, image in images.items()
    }
    return payload


def _sample_images() -> list[dict[str, Any]]:
    sample_dir = PROJECT_ROOT / "assets" / "samples"
    sources_path = sample_dir / "SOURCES.json"
    metadata: dict[str, dict[str, Any]] = {}
    if sources_path.exists():
        for item in json.loads(sources_path.read_text(encoding="utf-8")):
            metadata[item["filename"]] = item
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    for path in sorted(sample_dir.iterdir()) if sample_dir.exists() else []:
        if path.suffix.lower() not in allowed:
            continue
        meta = metadata.get(path.name, {})
        samples.append(
            {
                "filename": path.name,
                "scene": meta.get("scene", "示例图片"),
                "source": meta.get("source", "local"),
                "url": f"/samples/{path.name}",
            }
        )
    return samples


# ----------------------------------------------------------------------
# 应用工厂
# ----------------------------------------------------------------------
def create_app(config_path: str | Path | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    config = load_config(config_path)
    app.config["ROBOVISION"] = config
    app.config["MAX_CONTENT_LENGTH"] = int(get(config, "web.max_upload_mb", 20)) * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.json.ensure_ascii = False

    pipeline = PerceptionPipeline(config)
    session = DecisionSession(config)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", version=__version__)

    @app.get("/api/health")
    def health():
        status = pipeline.status()
        checkpoints = {}
        for env_name in ("cartpole", "robotnav"):
            path = resolve_path(get(config, f"decision.{env_name}.checkpoint"))
            checkpoints[env_name] = {"path": display_path(path), "exists": path.exists()}
        return jsonify(
            {
                "ok": True,
                "version": __version__,
                "models": status,
                "checkpoints": checkpoints,
                "samples": _sample_images(),
                "camera_index": int(get(config, "web.camera", 0)),
            }
        )

    @app.get("/samples/<path:filename>")
    def sample_file(filename: str):
        allowed = {item["filename"] for item in _sample_images()}
        if filename not in allowed:
            return jsonify({"error": "文件不在示例列表"}), 404
        return send_from_directory(PROJECT_ROOT / "assets" / "samples", filename)

    @app.post("/api/analyze")
    def analyze():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error": "请上传图片文件"}), 400
        tasks = request.form.get("tasks", "detect,face,depth")
        task_tuple = tuple(t.strip() for t in tasks.split(",") if t.strip())
        valid = {"detect", "face", "depth"}
        if not task_tuple or not set(task_tuple).issubset(valid):
            return jsonify({"error": f"tasks 必须是 {sorted(valid)} 的子集"}), 400

        data = file.read()
        if not data:
            return jsonify({"error": "上传文件为空"}), 400
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "无法解码图片，请使用 JPG/PNG 格式"}), 400
        frame = _prepare_image(frame)

        try:
            result = pipeline.analyze(frame, tasks=task_tuple)
            images = pipeline.render(frame, result)
            if "annotated" not in images:
                images["annotated"] = frame
            payload = _analysis_to_json(result, images)
            payload["filename"] = file.filename
            payload["tasks"] = list(task_tuple)
            return jsonify(payload)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc), "hint": "运行 python scripts/download_models.py"}), 503
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"分析失败：{type(exc).__name__}: {exc}"}), 500

    @app.get("/video_feed")
    def video_feed():
        mode = request.args.get("mode", "detect")
        if mode not in {"detect", "face", "depth", "all"}:
            return jsonify({"error": "mode 必须为 detect/face/depth/all"}), 400
        camera = int(get(config, "web.camera", 0))
        cap = open_video_source(camera)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return jsonify({"error": "摄像头不可用，请改用图片上传模式"}), 503

        task_map = {
            "detect": ("detect",),
            "face": ("face",),
            "depth": ("depth",),
            "all": ("detect", "depth"),
        }

        def generate():
            fps = 0.0
            last = time.time()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = pipeline.analyze(frame, tasks=task_map[mode], track=True)
                    images = pipeline.render(frame, result)
                    if mode == "depth" and result.get("depth") is not None:
                        output = result["depth"].color
                    elif mode == "face" and "faces" in images:
                        output = images["faces"]
                    elif mode == "all":
                        output = images.get("fused", images["annotated"])
                    else:
                        output = images["annotated"]
                    now = time.time()
                    instant = 1.0 / max(1e-6, now - last)
                    fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
                    last = now
                    from ..pipelines.visualization import draw_hud

                    output = draw_hud(
                        output,
                        [f"FPS {fps:4.1f}", f"objects {len(result.get('detections', []))}"],
                    )
                    jpeg = encode_image_bgr(output, ".jpg", 85)
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(
                        len(jpeg)
                    ).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            finally:
                cap.release()

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/decision/models")
    def decision_models():
        checkpoints = {}
        for env_name in ("cartpole", "robotnav"):
            path = resolve_path(get(config, f"decision.{env_name}.checkpoint"))
            checkpoints[env_name] = path.exists()
        return jsonify(
            {
                "checkpoints": checkpoints,
                "policies": {
                    "robotnav": ["dqn", "astar", "random", "manual"],
                    "cartpole": ["dqn", "heuristic", "random", "manual"],
                },
            }
        )

    @app.post("/api/decision/reset")
    def decision_reset():
        body = request.get_json(silent=True) or {}
        env_name = body.get("env", "robotnav")
        policy = body.get("policy", "dqn")
        if env_name not in {"robotnav", "cartpole"}:
            return jsonify({"error": "env 必须为 robotnav 或 cartpole"}), 400
        if policy not in {"dqn", "random", "astar", "heuristic", "manual"}:
            return jsonify({"error": "未知策略"}), 400
        try:
            return jsonify(session.reset(env_name, policy))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.post("/api/decision/step")
    def decision_step():
        body = request.get_json(silent=True) or {}
        auto = bool(body.get("auto", True))
        action = body.get("action")
        if action is not None:
            try:
                action = int(action)
            except (TypeError, ValueError):
                return jsonify({"error": "action 必须为整数"}), 400
        try:
            return jsonify(session.step(auto=auto, action=action))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": f"文件超过 {get(config, 'web.max_upload_mb', 20)}MB 限制"}), 413

    @app.after_request
    def add_headers(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="启动 RoboVision Web 演示")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config()
    app = create_app()
    host = args.host or get(config, "web.host", "127.0.0.1")
    port = args.port or int(get(config, "web.port", 5000))
    print(f"RoboVision Web 已启动：http://{host}:{port}")
    app.run(host=host, port=port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
