# Project Protoss

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

### 3. 가상환경 및 Python 패키지 설치

ACG는 Python 3.10을 기준으로 합니다. 먼저 버전을 확인합니다.
#### 3.1 Conda 설치
```bash
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
source /workspace/miniconda3/bin/activate
conda init bash

conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/r

conda create -n acg python=3.10 -y
conda activate acg
```
#### 3.2. ACG requirements.txt 설치
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
pip install -e libs/Isaac-GR00T-N1
pip install --no-build-isolation flash-attn==2.8.3

# Robosuite and RoboCasa
pip install libs/robosuite/
pip install -e libs/robocasa/

# Download RoboCasa assets (~5GB)
python libs/robocasa/robocasa/scripts/download_kitchen_assets.py

# Robomimic and DexMimicGen
pip install -e libs/robomimic/ --no-dependencies
pip install -e libs/dexmimicgen/ --no-dependencies

# Remaining requirements
pip install -r requirements.txt
```

#### 3.3. 추가 설치
```bash
python -m pip install --force-reinstall --no-deps \
  numpy==1.23.5 \
  tianshou==0.5.1 \
  tensorboardX
python -m pip install -e libs/robomimic/ --no-deps
```

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
