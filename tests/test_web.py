"""Flask 接口测试（不触发大模型推理）。"""

import io

from robotvision.web.app import create_app


def test_index_page():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"RoboVision" in response.data


def test_health_contains_models():
    app = create_app()
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert {"detection", "face", "depth"} <= set(payload["models"])
    assert "samples" in payload


def test_analyze_without_file_returns_400():
    app = create_app()
    client = app.test_client()
    response = client.post("/api/analyze", data={})
    assert response.status_code == 400


def test_decision_random_episode():
    app = create_app()
    client = app.test_client()
    response = client.post("/api/decision/reset", json={"env": "robotnav", "policy": "random"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"]["type"] == "robotnav"
    assert payload["policy"] == "random"

    response = client.post("/api/decision/step", json={"auto": True})
    assert response.status_code == 200
    step = response.get_json()
    assert "state" in step
    assert step["action"] in (0, 1, 2, 3)


def test_decision_manual_step():
    app = create_app()
    client = app.test_client()
    client.post("/api/decision/reset", json={"env": "robotnav", "policy": "manual"})
    response = client.post("/api/decision/step", json={"auto": False, "action": 0})
    assert response.status_code == 200
    assert response.get_json()["action"] == 0


def test_decision_invalid_env():
    app = create_app()
    client = app.test_client()
    response = client.post("/api/decision/reset", json={"env": "unknown", "policy": "random"})
    assert response.status_code == 400


def test_upload_non_image_returns_400():
    app = create_app()
    client = app.test_client()
    data = {"file": (io.BytesIO(b"not an image"), "bad.txt"), "tasks": "detect"}
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
