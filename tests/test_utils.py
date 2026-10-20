"""工具函数、NMS 与 IoU 跟踪测试。"""

import cv2
import numpy as np

from robotvision.config import PROJECT_ROOT, display_path
from robotvision.models.tracker import IoUTracker
from robotvision.types import Detection
from robotvision.utils import box_iou, decode_image, encode_image_bgr, nms, parse_source


def test_display_path_is_relative():
    assert display_path(PROJECT_ROOT / "models" / "yolov8n.pt") == "models/yolov8n.pt"
    assert display_path(PROJECT_ROOT / "runs" / "robotnav" / "best.pt").startswith("runs/")
    assert display_path("some/outside/path.pt") == "some/outside/path.pt"


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    iou = box_iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(iou - 1 / 3) < 1e-5


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.45)
    assert keep == [0, 2]


def test_image_codec_roundtrip():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    data = encode_image_bgr(image, ".png")
    decoded = decode_image(data)
    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image)


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("assets/a.jpg") == "assets/a.jpg"


def test_tracker_keeps_id_for_stable_box():
    tracker = IoUTracker(iou_threshold=0.3, max_age=2)
    det = Detection((10, 10, 50, 80), 0.9, 0, "person")
    first = tracker.update([det])
    assert first[0].track_id == 1
    second = tracker.update([Detection((11, 11, 51, 81), 0.9, 0, "person")])
    assert second[0].track_id == 1


def test_tracker_creates_new_id_for_disjoint_box():
    tracker = IoUTracker(iou_threshold=0.3, max_age=2)
    tracker.update([Detection((0, 0, 20, 20), 0.9, 0, "person")])
    second = tracker.update([Detection((200, 200, 220, 220), 0.9, 0, "person")])
    assert second[0].track_id == 2


def test_tracker_drops_old_tracks():
    tracker = IoUTracker(iou_threshold=0.3, max_age=1)
    tracker.update([Detection((0, 0, 20, 20), 0.9, 0, "car")])
    tracker.update([])
    tracker.update([])
    assert tracker.tracks == []
