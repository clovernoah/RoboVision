"""轻量 IoU 多目标跟踪器（SORT 简化版）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Detection
from ..utils import box_iou


@dataclass
class _Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    class_id: int
    misses: int = 0


class IoUTracker:
    """基于 IoU 贪心匹配的跟踪器。

    对每一帧检测结果与已有轨迹计算 IoU，按 IoU 从高到低贪心分配；
    未匹配的检测新建轨迹，连续 ``max_age`` 帧未匹配的轨迹被删除。
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """更新轨迹并为检测结果写入 ``track_id``（返回新列表）。"""
        if not detections:
            for track in self.tracks:
                track.misses += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
            return []

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                iou = box_iou(track.xyxy, det.xyxy)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: dict[int, int] = {}
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets[di] = ti

        updated: list[Detection] = []
        for di, det in enumerate(detections):
            if di in matched_dets:
                track = self.tracks[matched_dets[di]]
                track.xyxy = det.xyxy
                track.misses = 0
                track_id = track.track_id
            else:
                track_id = self._next_id
                self._next_id += 1
                self.tracks.append(_Track(track_id, det.xyxy, det.class_id))
            updated.append(
                Detection(
                    xyxy=det.xyxy,
                    score=det.score,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    track_id=track_id,
                )
            )

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # 重新分配：删除了旧轨迹后 track_id 仍然保持，无需改动
        return updated
