# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ffmpeg git libegl1 libgl1 libglib2.0-0 \
    libgles2 libglew2.2 libhdf5-dev libosmesa6 libsm6 libusb-1.0-0-dev \
    libxext6 libxrender1 ninja-build pkg-config python-is-python3 \
    python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja

WORKDIR /workspace/ACG
COPY ACG/ /workspace/ACG/

# Install PyTorch first so CUDA extensions build against the intended version.
RUN python -m pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install -e libs/Isaac-GR00T-N1 && \
    python -m pip install --no-build-isolation --no-deps flash-attn==2.8.3

# RoboCasa and Robomimic pin versions incompatible with GR00T's PyTorch and
# NumPy requirements. Install their source packages without those old pins.
RUN python -m pip install libs/robosuite/ && \
    python -m pip install -e libs/robocasa/ --no-deps && \
    python -m pip install -e libs/robomimic/ --no-deps && \
    python -m pip install -e libs/dexmimicgen/ --no-deps && \
    python -m pip install pygame pynput hidapi lxml tensorboard tensorboardX \
        psutil imageio-ffmpeg && \
    python -m pip install -r requirements.txt

# RoboCasa checks these exact versions at import time. NumPy 1.23.5 also
# satisfies GR00T's >=1.23.5,<2 requirement.
RUN python -m pip install numpy==1.23.5 numba==0.56.4 mujoco==3.2.6 \
        protobuf==4.25.0 && \
    python -c "import torch, gr00t, robosuite, robocasa, robomimic, dexmimicgen, flash_attn; print('ACG imports OK; PyTorch', torch.__version__)"

CMD ["bash"]
