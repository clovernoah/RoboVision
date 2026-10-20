"""下载用于演示与报告的公开示例图片（Unsplash / Pexels）。

图片仅用于课程演示。来源与作者信息记录在 assets/samples/SOURCES.json。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "assets" / "samples"

ASSETS = [
    {
        "filename": "street_traffic.jpg",
        "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1280&q=85",
        "scene": "城市道路车流：车辆检测",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1449824913935-59a10b8d2000",
    },
    {
        "filename": "street_pedestrian.jpg",
        "url": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=1280&q=85",
        "scene": "城市街景：行人/车辆混合",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1519501025264-65ba15a82390",
    },
    {
        "filename": "bus.jpg",
        "url": "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=1280&q=85",
        "scene": "公交车：车辆检测",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1544620347-c4fd4a3d5957",
    },
    {
        "filename": "cars.jpg",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=1280&q=85",
        "scene": "跑车：车辆检测",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1503376780353-7e6692767b70",
    },
    {
        "filename": "portrait.jpg",
        "url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=1000&q=85",
        "scene": "人物肖像：人脸检测",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1494790108377-be9c29b29330",
    },
    {
        "filename": "portrait_man.jpg",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1000&q=85",
        "scene": "人物肖像：人脸检测",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1507003211169-0a1dd7228f2d",
    },
    {
        "filename": "group.jpg",
        "url": "https://images.pexels.com/photos/1595385/pexels-photo-1595385.jpeg?auto=compress&cs=tinysrgb&w=1280",
        "scene": "多人合影：8 张人脸",
        "source": "Pexels",
        "page": "https://www.pexels.com/photo/1595385/",
    },
    {
        "filename": "city_street.jpg",
        "url": "https://images.unsplash.com/photo-1514924013411-cbf25faa35bb?w=1280&q=85",
        "scene": "城市街道：深度估计与融合",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1514924013411-cbf25faa35bb",
    },
    {
        "filename": "students.jpg",
        "url": "https://images.unsplash.com/photo-1543269865-cbf427effbad?w=1280&q=85",
        "scene": "学生讨论：多人脸检测",
        "source": "Unsplash",
        "page": "https://unsplash.com/photos/1543269865-cbf427effbad",
    },
]


def download(url: str, target: Path, retries: int = 3) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RoboVision"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            if len(data) < 20_000:
                raise RuntimeError("图片过小，可能下载失败")
            target.write_bytes(data)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"下载失败：{last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载示例图片")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    failures = 0
    for asset in ASSETS:
        target = SAMPLE_DIR / asset["filename"]
        if target.exists() and not args.force:
            print(f"[跳过] {asset['filename']}")
        else:
            print(f"[下载] {asset['filename']} - {asset['scene']}")
            try:
                download(asset["url"], target)
            except Exception as exc:  # noqa: BLE001
                print(f"[失败] {asset['filename']}: {exc}")
                failures += 1
                continue
        sources.append({k: v for k, v in asset.items() if k != "url"})

    (SAMPLE_DIR / "SOURCES.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成：{len(sources)} 张图片，失败 {failures} 张，目录 {SAMPLE_DIR.relative_to(PROJECT_ROOT)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
