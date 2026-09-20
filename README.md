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
```

#### 3.2. Build environment for ACG (ICRA 2026)
```bash
conda create -n acg python=3.10 -y
conda activate acg
cd ACG
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

python -m pip install --force-reinstall --no-deps \
  numpy==1.23.5 \
  tianshou==0.5.1 \
  tensorboardX
python -m pip install -e libs/robomimic/ --no-deps
```

#### 3.3. Download Dataset (RoboCasa)
```bash
# bucket name : p5v8a9zcuo
# Endpoint URL : https://s3api-us-ca-2.runpod.io
# aws s3 ls --region us-ca-2 --endpoint-url https://s3api-us-ca-2.runpod.io s3://p5v8a9zcuo/

mkdir -p /workspace/datasets/robocasa
cat > libs/robocasa/robocasa/macros_private.py <<'PY'
DATASET_BASE_PATH = "/workspace/datasets/robocasa"
PY
python libs/robocasa/robocasa/scripts/download_kitchen_assets.py

printf 'y\n' | python -m robocasa.scripts.download_datasets --ds_types mg_im
```
