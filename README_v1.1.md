# Project Protoss v1.1 — GR00T N1.7 + π0.5-DROID

추가 학습 없이 GR00T-N1.7-DROID와 π0.5-DROID의 **absolute joint·gripper action**을 혼합한다.
실행 파일은 [Protoss/main_v1.1.py](Protoss/main_v1.1.py)이다.

```text
RoboLab → Protoss :5555 ─→ GR00T N1.7 :5557 (ZeroMQ)
                       └→ π0.5-DROID :8000 (WebSocket)
RoboLab ← alpha × N1.7 action + (1-alpha) × π0.5 action
```

| alpha | 최종 action |
|---|---|
| 0 | π0.5-DROID |
| 0.5 | 두 모델 동일 비율 |
| 1 | GR00T-N1.7-DROID |

양 끝값에서도 두 서버 모두 호출한다. v1.0의 N1.6 서버는 이 버전에서 사용하지 않는다.
N1.7은 기존 `decode_action()`으로 복원한 결과를 사용한다. π 서버는 반드시
**`pi05_droid_jointpos`** 설정으로 시작한다. 이 설정은 서버에서 상대 관절 출력을
absolute 관절 목표로 복원한다. Protoss는 해당 결과에 현재 관절값을 다시 더하지 않는다.
[RoboLab의 π 서버 안내](https://github.com/NVlabs/RoboLab/blob/main/policies/pi0_family/README.md),
[OpenPI jointpos 설정](https://github.com/xuningy/openpi/blob/main/src/openpi/training/config.py)을 기준으로 한다.

## 1. 환경과 다운로드 위치

Ubuntu 22.04/24.04, Linux x86_64, CUDA가 사용 가능한 NVIDIA GPU와 Bash를 기준으로 한다.
OS 설치 명령은 root 계정 기준이며 일반 계정은 `apt-get`, `ldconfig`에 `sudo`를 붙인다.
프로젝트는 `/workspace/Project_Protoss`에 있다고 가정한다. 실제 경로가 `/project_protoss`라면
**각 블록의 `PROTOSS_ROOT`를 `/project_protoss`로 바꾼다.**

```text
Project_Protoss/
├── Issac-GR00T-N17/                  # N1.7 소스와 전용 환경
├── openpi/                          # π0.5 소스와 전용 환경
├── openpi-assets-simeval/
│   └── pi05_droid_jointpos/          # π0.5 실제 weight와 normalization assets
├── big_vision/                      # PaliGemma tokenizer cache
├── pi05_droid_jointpos/              # 위 checkpoint를 가리키는 링크
├── Protoss/
│   ├── main_v1.0.py                  # v1.1에서도 GR00T 통신 함수를 재사용
│   ├── main_v1.1.py
│   ├── requirements_v1.1.txt
│   ├── .venv/
│   └── checkpoints/GR00T-N1.7-DROID/
└── README_v1.1.md
```

π 소스와 weight 모두 **프로젝트 루트 아래**에 설치한다. Git 추적에서는 제외한다.
RoboLab은 `/workspace/RoboLab`에 별도로 설치한다.

| 프로세스 | Python | 포트 |
|---|---|---|
| GR00T N1.7 | 3.12 | 5557 |
| OpenPI π0.5 | 3.11 | 8000 |
| Protoss proxy | 3.12 | 5555 |
| RoboLab / Isaac Sim 5.1 | 3.11 | client |

명령은 기본적으로 GPU 0을 사용한다. 충분한 VRAM이 있어야 함께 실행할 수 있다.
GPU가 여러 개면 아래의 `PI_GPU`, `SIM_GPU`를 각각 1, 2 등 실제 번호로 바꿔 분산한다.
`nvidia-smi -L`에 없는 번호를 지정하면 CUDA가 보이지 않아 FlashAttention 오류 등이 발생한다.
괄호 안의 `set -e`는 오류 발생 시 작업만 중단하며 부모 터미널을 종료하지 않는다.

```bash
(
set -e
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
nvidia-smi -L
apt-get update
apt-get install -y git git-lfs curl build-essential ffmpeg \
  vulkan-tools libvulkan1 libglvnd0 libgl1 libegl1 libglu1-mesa \
  libxt6 libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 \
  libxcursor1 libxi6 libxkbcommon0 libgomp1 libsm6 libice6
ldconfig
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git lfs install
)
```

### 프로젝트 받기

시스템 도구 설치 후 이 블록으로 프로젝트를 받는다.
이미 받은 프로젝트는 새로 clone하지 않고 실제 경로를 사용한다. 이후 모든 블록의
`PROTOSS_ROOT`를 같은 경로로 지정한다. 설치 블록은 **Bash** 터미널에서 순서대로 실행한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
mkdir -p "$(dirname "$PROTOSS_ROOT")"
if [ ! -d "$PROTOSS_ROOT" ]; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/siri2100/Project_Protoss.git "$PROTOSS_ROOT"
fi
test -f "$PROTOSS_ROOT/Protoss/main_v1.1.py"
test -f "$PROTOSS_ROOT/Protoss/main_v1.0.py"
)
```

## 2. GR00T-N1.7 설치와 weight 다운로드

이미 설치·다운로드했다면 이 단계는 생략한다. 새 설치는 기존 v1.0에서 사용한 N1.7 소스 커밋을 사용한다.
Hugging Face 로그인 계정은 [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)에 접근할 수 있어야 한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT"
if [ ! -d Issac-GR00T-N17 ]; then
  GIT_TERMINAL_PROMPT=0 git clone --no-checkout \
    https://github.com/NVIDIA/Isaac-GR00T.git Issac-GR00T-N17
  git -C Issac-GR00T-N17 checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
fi
test "$(git -C Issac-GR00T-N17 rev-parse HEAD)" = 51d4c89f72fda44cbf77285c6a8114b52676b8a1 || {
  echo "N1.7 소스 버전이 다릅니다. 기존 폴더를 백업하고 지정 커밋으로 설치하세요." >&2
  exit 1
}
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
test -f pyproject.toml
GIT_TERMINAL_PROMPT=0 git submodule update --init --recursive
git lfs pull --include="scripts/deployment/dgpu/wheels/**"
uv sync --locked --python 3.12
uv run --no-sync hf auth whoami || uv run --no-sync hf auth login
HF_HUB_ENABLE_HF_TRANSFER=0 uv run --no-sync hf download nvidia/Cosmos-Reason2-2B config.json \
  --local-dir "$PROTOSS_ROOT/Protoss/access-check/cosmos"
HF_HUB_ENABLE_HF_TRANSFER=0 uv run --no-sync hf download nvidia/GR00T-N1.7-DROID \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID"
)
```

기존 N1.7 폴더가 `Isaac-GR00T-N17`이라면 위 경로를 실제 이름으로 바꾼다.
이 문서의 기본 이름은 프로젝트의 실제 로컬 폴더명인 `Issac-GR00T-N17`이다.
FFmpeg는 N1.7 torchcodec이 지원하는 4~7 버전을 사용한다.

## 3. OpenPI 설치 — 프로젝트 루트의 openpi/

RoboLab에서 안내하는 **`xuningy/openpi` fork**를 사용한다. 이 fork에 `pi05_droid_jointpos`가 있다.
N1.7 또는 RoboLab 환경과 합쳐 설치하지 않는다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT"
if [ ! -d openpi ]; then
  GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0 git clone --recurse-submodules \
    https://github.com/xuningy/openpi.git openpi
  git -C openpi checkout aa6420561529593114160d05e5ad155792b272f3
fi
test "$(git -C openpi rev-parse HEAD)" = aa6420561529593114160d05e5ad155792b272f3 || {
  echo "OpenPI 소스 버전이 다릅니다. 기존 폴더를 백업하고 지정 커밋으로 설치하세요." >&2
  exit 1
}
GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0 git -C openpi submodule update --init --recursive
cd "$PROTOSS_ROOT/openpi"
test -f pyproject.toml
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e .
uv run --no-sync python - <<'PY'
from openpi.training import config

cfg = config.get_config("pi05_droid_jointpos")
print("OpenPI config:", cfg.name, "horizon:", cfg.model.action_horizon)
PY
)
```

## 4. π0.5-DROID weight 다운로드 — 프로젝트 루트 아래

RoboLab의 jointpos checkpoint는 Hugging Face가 아니라 GCS의
`gs://openpi-assets-simeval/pi05_droid_jointpos`에서 받는다.
공개 asset을 anonymous 모드로 요청하며 GitHub 비밀번호나 Hugging Face 토큰은 필요하지 않다.
403/404가 발생하면 접근이나 원격 checkpoint 상태를 확인해야 하며, 다른 checkpoint로 임의 대체하지 않는다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export OPENPI_DATA_HOME="$PROTOSS_ROOT"
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/openpi"
uv run --no-sync python - <<'PY'
import os
from pathlib import Path
from openpi.shared import download

root = Path(os.environ["PROTOSS_ROOT"]).resolve()
# checkpoint와 tokenizer의 다운로드·lock 파일 상위 경로를 준비한다.
cache = download.get_cache_dir()
for bucket in ("openpi-assets-simeval", "big_vision"):
    (cache / bucket).mkdir(parents=True, exist_ok=True)
checkpoint = download.maybe_download(
    "gs://openpi-assets-simeval/pi05_droid_jointpos", gs={"token": "anon"}
)
assert (checkpoint / "params").is_dir(), "Checkpoint params 폴더가 없습니다"
assert any((checkpoint / "assets").rglob("norm_stats.json")), "Normalization 통계가 없습니다"
alias = root / "pi05_droid_jointpos"
if alias.exists() or alias.is_symlink():
    assert alias.resolve() == checkpoint.resolve(), f"다른 경로가 이미 존재합니다: {alias}"
else:
    alias.symlink_to(checkpoint, target_is_directory=True)
tokenizer = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
)
assert tokenizer.is_file() and tokenizer.stat().st_size > 0, "Tokenizer 다운로드 실패"
print("Tokenizer:", tokenizer)
print("Downloaded checkpoint:", checkpoint)
print("Server checkpoint path:", alias)
PY
)
```

checkpoint와 PaliGemma tokenizer를 함께 받는다. `mkdir`는 첫 설치에서
다운로드·잠금 파일의 상위 경로를 준비한다.
[고정 버전 다운로드 구현](https://github.com/xuningy/openpi/blob/aa6420561529593114160d05e5ad155792b272f3/src/openpi/shared/download.py),
[tokenizer 구현](https://github.com/xuningy/openpi/blob/aa6420561529593114160d05e5ad155792b272f3/src/openpi/models/tokenizer.py)을 기준으로 한다.

실제 weight는 `$PROTOSS_ROOT/openpi-assets-simeval/pi05_droid_jointpos`에 저장되고,
서버에는 `$PROTOSS_ROOT/pi05_droid_jointpos`를 전달한다. 재실행하면 완성된 cache를 재사용한다.
서버 실행에도 동일한 `OPENPI_DATA_HOME`을 사용해 추가 모델 asset cache를 프로젝트 안에 유지한다.

## 5. Protoss v1.1 의존성 설치

OpenPI 모델 전체가 아니라 가벼운 `openpi-client`만 Protoss 환경에 설치한다.
이미 v1.0 가상환경이 있으면 그대로 사용한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT"
test -f openpi/packages/openpi-client/pyproject.toml
if [ ! -x Protoss/.venv/bin/python ]; then
  uv venv --python 3.12 Protoss/.venv
fi
uv pip install --python Protoss/.venv/bin/python \
  -r Protoss/requirements_v1.1.txt -e "$PROTOSS_ROOT/openpi/packages/openpi-client"
Protoss/.venv/bin/python Protoss/main_v1.1.py --help
Protoss/.venv/bin/python -m unittest discover -s Protoss -p 'test_main_v1_*.py' -v
)
```

`main_v1.1.py`와 통신 함수를 제공하는 `main_v1.0.py`를 같은 폴더에 둔다.

## 6. RoboLab 설치

이미 `.venv-51` 환경을 준비했다면 생략한다. RoboLab은 현재 upstream `main`을 사용하므로
N1.7/OpenPI처럼 커밋을 고정한 설치가 아니다. 설치 후 `git rev-parse HEAD`를 기록한다.
Protoss에는 GR00T 형식으로 연결하므로 기존 `policies/gr00t/client.py`를 그대로 사용한다.
RoboLab 환경에 OpenPI 모델이나 OpenPI client를 추가 설치할 필요는 없다.

```bash
(
set -e
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
cd /workspace
if [ ! -d RoboLab ]; then
  GIT_TERMINAL_PROMPT=0 git clone https://github.com/NVlabs/RoboLab.git RoboLab
fi
cd /workspace/RoboLab
git rev-parse HEAD
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --python 3.11 --extra isaac51
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --no-sync --extra isaac51 python -c \
  "import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab import OK')"
)
```

## 7. Inference — 터미널 4개

각 서버가 준비된 후 다음 터미널을 시작한다. 모든 포트와 주소는 같은 서버 기준이다.

### 터미널 1 — GR00T N1.7, 포트 5557

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA가 보이지 않습니다'; print(torch.cuda.get_device_name(0))"
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID" \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
)
```

### 터미널 2 — π0.5-DROID jointpos, 포트 8000

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export OPENPI_DATA_HOME="$PROTOSS_ROOT"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export PI_GPU=0
export CUDA_VISIBLE_DEVICES="$PI_GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/openpi"
uv run --no-sync python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'JAX에서 GPU가 보이지 않습니다'"
test -d "$PROTOSS_ROOT/pi05_droid_jointpos/params"
test -d "$PROTOSS_ROOT/pi05_droid_jointpos/assets"
uv run --no-sync scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir="$PROTOSS_ROOT/pi05_droid_jointpos"
)
```

GPU가 여러 개면 `PI_GPU=1` 등으로 바꾼다. 일반 `pi05_droid` 설정을 이 명령 대신 사용하면
관절 출력 의미가 달라질 수 있으므로 `jointpos` 설정을 유지한다.

### 터미널 3 — Protoss main_v1.1.py, 포트 5555

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv/bin/python Protoss/main_v1.1.py \
  --n17-endpoint tcp://127.0.0.1:5557 \
  --pi-uri ws://127.0.0.1:8000 \
  --alpha 0.5 --horizon 8 --pi-horizon 15 --timeout-ms 120000 \
  --host 127.0.0.1 --port 5555
)
```

`Protoss ready`가 나오면 두 서버 연결과 N1.7 modality config 확인이 완료된 것이다.
π handshake에는 action 의미가 명시되지 않을 수 있어 실제 jointpos 설정까지 자동 확인하지는 않는다.
비율을 바꾸려면 이 서버를 종료하고 `--alpha`를 바꿔 재시작한다.

### 터미널 4 — RoboLab 평가

```bash
(
set -e
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
export SIM_GPU=0
export CUDA_VISIBLE_DEVICES="$SIM_GPU"
cd /workspace/RoboLab
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --no-sync --extra isaac51 \
  python policies/gr00t/run.py \
  --headless --device cuda:0 \
  --remote-host 127.0.0.1 --remote-port 5555 \
  --task BananaOnPlateTask \
  --num-envs 1 --num-runs 1 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name protoss_v11_alpha05_smoke
)
```

GPU가 여러 개면 `SIM_GPU=2` 등으로 바꾼다. 처음에는 환경 1개로 확인한 뒤 `--num-envs 10`으로 늘린다.
각 alpha 실험은 output 이름을 바꾼다. RoboLab은 이전 완료 episode를 재사용할 수 있다.
RoboLab은 5557이나 8000이 아닌 **Protoss 5555**로 연결한다.

## 8. NPZ 관측으로 단일 inference

실제 N1.7 형식 관측을 `observation.npz`로 저장한 경우, 두 upstream 서버만 실행하고
아래 명령을 사용한다. Protoss proxy 서버는 필요하지 않다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv/bin/python Protoss/main_v1.1.py \
  --n17-endpoint tcp://127.0.0.1:5557 --pi-uri ws://127.0.0.1:8000 \
  --observation observation.npz --output blended_actions_v1.1.npz \
  --alpha 0.5 --horizon 8
)
```

입력은 flat N1.7 관측이며 공개 DROID checkpoint의 현재 영상 `T=1`을 기준으로 한다.

| 키 | Shape / dtype |
|---|---|
| `video.exterior_image_1_left` | `[B,1,H,W,3]`, RGB uint8 |
| `video.wrist_image_left` | `[B,1,H,W,3]`, RGB uint8 |
| `state.eef_9d` | `[B,1,9]`, float32 |
| `state.joint_position` | `[B,1,7]`, float32 |
| `state.gripper_position` | `[B,1,1]`, float32 |
| `annotation.language.language_instruction` | `[B]`, 문자열 배열 |

π 입력은 같은 현재 관절·gripper와 instruction을 사용하며 EEF state는 사용하지 않는다.
영상은 N1.7에 원본을 전달하고, π에는 OpenPI 방식의 224×224 letterbox를 적용한다.
π 서버는 batch 없이 호출하므로 B>1이면 각 샘플을 순서대로 요청한다.
출력은 `action.joint_position` `[B,8,7]`, `action.gripper_position` `[B,8,1]`, float32이다.
π의 15-step chunk 중 처음 8개를 N1.7의 처음 8개와 혼합한다. RoboLab은 gripper를 `>0.5`로 이진화한다.

## 9. 검증 범위와 오류 확인

- `uv sync --locked` 실패: 소스 커밋과 `uv.lock` 변경 여부를 확인한다.
- `.venv`가 없거나 `--no-sync` 실행 실패: 해당 설치 단계를 먼저 완료한다.
- tokenizer/asset 다운로드 오류: 4단계를 다시 실행하고 동일한 `OPENPI_DATA_HOME`을 사용한다.
- `ModuleNotFoundError: openpi_client`: 5단계에서 local client package까지 함께 설치한다.
- `Flash Attention 2 is not available on CPU`: `nvidia-smi -L`에 있는 GPU 번호와 Torch CUDA 접근을 확인한다.
- `JAX에서 GPU가 보이지 않습니다`: OpenPI 전용 환경과 CUDA/driver 접근을 확인한다.
- `hf_transfer` 오류: 다운로드·서버 블록의 `HF_HUB_ENABLE_HF_TRANSFER=0`을 유지한다.
- `[T,8]` action 오류: π 서버가 `pi05_droid_jointpos` config로 실행되는지 확인한다.
- timeout: upstream 서버의 모델 로딩 완료를 확인한다. 실패한 action을 자동 재시도하거나 대체하지 않는다.

두 모델이 같은 로봇 관절 순서, 단위, gripper 의미와 실행 시간 간격을 사용해야 한다.
v1.1은 같은 step index를 혼합하며 다른 제어 주파수에 대한 resampling은 하지 않는다.
RTC는 지원하지 않는다. α=0/1에도 두 서버가 호출되어 순수 단일 모델의 추론 지연과 다르다.
CPU 테스트는 action 변환, blending, batch 및 WebSocket 처리를 확인한다.
실제 weight 다운로드와 GPU 모델·RoboLab 평가는 RunPod에서 위 명령으로 별도 확인해야 한다.
