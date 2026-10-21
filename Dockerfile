FROM python:3.12-slim

LABEL org.opencontainers.image.title="RoboVision" \
      org.opencontainers.image.description="Intelligent robot perception and decision system" \
      org.opencontainers.image.source="https://github.com/clovernoah/RoboVision" \
      org.opencontainers.image.licenses="AGPL-3.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV / PyTorch 运行所需的系统库
# 使用清华 Debian 镜像，避免 deb.debian.org 在国内过慢
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 工具链从清华源安装
RUN python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
     --upgrade pip setuptools wheel

# 只从 PyTorch 官方源下载 torch/torchvision 本体，避免巨量 CUDA 包
RUN python -m pip install --no-cache-dir \
     --index-url https://download.pytorch.org/whl/cpu \
     --no-deps torch torchvision

# torch 的纯 Python 依赖走清华源，避免官方源小文件下载过慢
RUN python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
     filelock typing-extensions sympy networkx jinja2 fsspec numpy pillow

# 依赖单独一层，代码改动时可以复用缓存
COPY requirements.txt .
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 项目代码
COPY robotvision ./robotvision
COPY configs ./configs
COPY assets ./assets
COPY models ./models

# 演示用 DQN 检查点
COPY runs/robotnav/best.pt ./runs/robotnav/best.pt
COPY runs/cartpole/best.pt ./runs/cartpole/best.pt
COPY runs/robotnav_vanilla/best.pt ./runs/robotnav_vanilla/best.pt

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health')" || exit 1

CMD ["python", "-m", "robotvision", "web", "--host", "0.0.0.0", "--port", "5000"]
