# Project Protoss

> **Project Protoss** is an open-source initiative for building practical, controllable, and reliable video agents.

The long-term vision of Project Protoss is to enable AI agents that can understand, restore, edit, generate, and reason about videos while interacting with humans and production systems.

Unlike conventional video models that perform a single forward pass from input to output, Project Protoss focuses on **agentic video workflows**: observing videos, planning actions, executing tools, evaluating results, and iteratively improving outputs.

---

# Vision

Current video foundation models are becoming increasingly capable, but most of them behave as **single-shot models**:

```
Video + Prompt
        ↓
     Foundation Model
        ↓
       Output
```

Project Protoss explores a different direction.

Instead of asking one model to solve everything at once, an AI agent should:

- Observe the video
- Understand the user's goal
- Plan the workflow
- Execute appropriate tools
- Evaluate the result
- Revise the plan if necessary

The goal is to build practical video agents that can operate under real-world constraints.

---

# Principles

Project Protoss is designed around several core principles.

### Practical

Built for real production workflows rather than research demos.

### Deployable

Uses pretrained models and modular components that can be deployed in diverse environments.

### Cost-effective

Optimizes latency, GPU usage, and API cost by executing only necessary operations.

### Reliable

Continuously validates intermediate and final outputs to improve trustworthiness.

### Controllable

Keeps humans in the loop and allows explicit control over workflows and generation.

---

# Version 1

## Agentic Video Restoration

The first milestone focuses on intelligent video restoration.

Input

```
Video + Natural Language Instruction
```

Output

```
Restored Video
+ Execution Report
```

The agent is expected to:

- analyze video quality
- detect degraded regions
- localize restoration targets
- select appropriate restoration tools
- execute restoration
- evaluate restoration quality
- retry or adjust when necessary
- generate an execution report

Instead of restoring an entire video uniformly, the agent should restore only where necessary while preserving important content such as faces, text, and temporal consistency.

---

# Long-term Roadmap

## Version 1

Agentic Video Restoration

## Version 2

Agentic Video Restoration & Editing

- video editing
- object-aware editing
- timeline manipulation
- video understanding
- multi-step workflows

## Version 3

General-Purpose Video Agent

- video generation
- long-horizon planning
- multimodal reasoning
- multi-agent collaboration

Future versions will gradually expand toward a complete runtime for practical multimodal video agents.

---

# Why "Project Protoss"?

Project Protoss is the codename of this long-term research initiative.

The project aims to evolve over multiple generations, gradually expanding the capabilities of AI agents from video restoration to general-purpose multimodal video intelligence.

---

# Status

🚧 Early development (Version 1)

Contributions, discussions, and ideas are always welcome.

## ACG

See [Docker setup](DOCKER.md) for the ACG GPU environment.

## RunPod Pod에서 ACG 설치

RunPod의 일반 Pod는 그 자체가 컨테이너이므로 Pod 터미널 안에서
`run_docker.sh`를 실행할 필요가 없습니다. Docker daemon이 없는 일반 Pod에서는
해당 스크립트도 실행되지 않습니다. 아래 절차대로 Pod의 Python 환경에 ACG를
직접 설치합니다.

### 1. Pod 생성

RunPod에서 다음 조건으로 GPU Pod를 생성합니다.

- NVIDIA GPU 1개 이상
- PyTorch 2.7 계열과 CUDA 12.8을 제공하는 PyTorch 템플릿
- Container Disk: 최소 50 GB
- Volume Disk: 데이터셋과 체크포인트 크기에 맞게 설정하며 최소 100 GB 권장
- Volume Mount Path: `/workspace`

`/workspace`에 Volume 또는 Network Volume을 연결하면 Pod를 다시 만들더라도
소스, 데이터셋, 체크포인트를 유지할 수 있습니다. 학습 데이터가 크면 Container
Disk가 아니라 `/workspace` 아래에 저장합니다.

### 2. 저장소 받기

Pod의 Web Terminal 또는 SSH 터미널에서 실행합니다.

```bash
cd /workspace
git clone https://github.com/siri2100/Project_Protoss.git
cd Project_Protoss/ACG
```

이미 저장소를 업로드했다면 업로드한 `Project_Protoss/ACG` 디렉터리로 이동하면
됩니다.

### 3. 시스템 패키지 설치

```bash
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ffmpeg git libegl1 libgl1 libglib2.0-0 \
  libgles2 libglew2.2 libhdf5-dev libosmesa6 libsm6 libusb-1.0-0-dev \
  libxext6 libxrender1 ninja-build pkg-config python3-dev
rm -rf /var/lib/apt/lists/*
```

### 4. Python 패키지 설치

ACG는 Python 3.10을 기준으로 합니다. 먼저 버전을 확인합니다.

```bash
python --version
nvidia-smi
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

PyTorch 템플릿의 버전이 다르거나 설치되어 있지 않다면 ACG가 사용하는 버전으로
맞춥니다.

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

이어서 ACG와 포함된 라이브러리를 설치합니다.

```bash
python -m pip install -e libs/Isaac-GR00T-N1
python -m pip install --no-build-isolation --no-deps flash-attn==2.8.3

python -m pip install libs/robosuite/
python -m pip install -e libs/robocasa/ --no-deps
python -m pip install -e libs/robomimic/ --no-deps
python -m pip install -e libs/dexmimicgen/ --no-deps

python -m pip install \
  pygame pynput hidapi lxml tensorboard tensorboardX psutil imageio-ffmpeg
python -m pip install -r requirements.txt
python -m pip install \
  numpy==1.23.5 numba==0.56.4 mujoco==3.2.6 protobuf==4.25.0
```

FlashAttention은 설치 중 CUDA 확장을 컴파일하므로 시간이 오래 걸릴 수 있습니다.
컴파일 중 메모리가 부족하면 다음처럼 병렬 작업 수를 줄여 다시 실행합니다.

```bash
MAX_JOBS=2 python -m pip install \
  --no-build-isolation --no-deps flash-attn==2.8.3
```

### 5. 설치 및 GPU 확인

```bash
export MUJOCO_GL=egl
python - <<'PY'
import torch
import flash_attn
import gr00t
import robocasa
import robomimic
import robosuite
import dexmimicgen

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA GPU를 사용할 수 없습니다."
print("GPU:", torch.cuda.get_device_name(0))
print("ACG environment is ready.")
PY
```

매번 터미널을 열 때 MuJoCo의 headless 렌더링 설정이 적용되도록 저장합니다.

```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

### 6. RoboCasa 에셋 설치

RoboCasa를 사용할 경우 약 5 GB의 에셋을 한 번 내려받습니다.

```bash
cd /workspace/Project_Protoss/ACG
python libs/robocasa/robocasa/scripts/download_kitchen_assets.py
```

### 7. 데이터셋과 출력 경로

권장 디렉터리 구조는 다음과 같습니다.

```text
/workspace/
├── Project_Protoss/
│   └── ACG/
├── datasets/
│   ├── robocasa/
│   └── dexmimicgen/
├── checkpoints/
└── outputs/
```

ACG에 포함된 Robomimic 설정은 기본적으로 `~/datasets/robot/...` 경로를
참조합니다. 설정 파일을 수정하지 않으려면 다음 심볼릭 링크를 만듭니다.

```bash
mkdir -p ~/datasets
ln -sfn /workspace/datasets ~/datasets/robot
```

데이터셋의 실제 하위 경로가 설정과 일치하는지
`ACG/libs/Isaac-GR00T-N1/robomimic_configs/`의 JSON 파일에서 확인합니다.
학습과 rollout 명령은 `ACG/README.md` 및
`ACG/scripts/Training_Rollout_Guideline.md`를 따릅니다.

### Pod 재시작 후

`/workspace` Volume에 설치된 소스와 데이터는 유지됩니다. Python 패키지는
사용한 RunPod 템플릿과 저장 방식에 따라 사라질 수 있으므로 Pod를 종료하기
전 현재 환경을 저장해 두는 것이 좋습니다.

```bash
cd /workspace/Project_Protoss/ACG
python -m pip freeze > /workspace/acg-requirements.lock.txt
```

환경까지 반복해서 재사용하려면 루트의 `Dockerfile`을 외부 Docker 환경에서
빌드해 컨테이너 레지스트리에 push한 뒤, 그 이미지를 RunPod Custom Image로
지정합니다. 일반 RunPod Pod 내부에서 Docker 이미지를 다시 빌드하는 방식은
사용하지 않습니다.
