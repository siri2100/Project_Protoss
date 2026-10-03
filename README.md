# Project Protoss v1.0

GR00T-N1.7-DROID와 GR00T-N1.6-DROID의 pretrained weight를 추가 학습 없이 사용하고,
각 모델의 최종 action을 물리 단위로 복원한 뒤 joint·gripper action을 혼합한다.

```text
RoboLab → Protoss :5555 → N1.7 server :5557 → decoded actions17 ─┐
                       → N1.6 server :5556 → decoded actions16 ─┤
RoboLab ← blended action = alpha × actions17 + (1-alpha) × actions16
```

`alpha=1`은 N1.7, `alpha=0`은 N1.6, `alpha=0.5`는 동일 비율이다.
실제 혼합 코드는 [Protoss/main_v1.0.py](Protoss/main_v1.0.py)의 `blend_actions()`에 있다.
모델별 정규화와 채널 구성이 달라 내부 `action_pred` 전체를 그대로 평균하지 않는다.
각 서버의 `decode_action()` 결과 중 공통 joint·gripper만 혼합하고 EEF 채널은 반환하지 않는다.

## 1. 실행 환경과 경로

아래 명령은 **Ubuntu 22.04/24.04, Linux x86_64, NVIDIA CUDA GPU가 있는 서버 또는
RunPod 컨테이너의 Bash 터미널**을 기준으로 한다. macOS에서 모델·RoboLab GPU 추론을 실행하는 절차는 아니다.
GR00T 설치는 CUDA 12.8 환경을 기준으로 하며 `nvidia-smi`로 GPU 접근을 먼저 확인한다.
OS 패키지 설치 명령은 root 기준이다. 일반 계정이면 `apt-get`, `ldconfig`에 `sudo`를 붙인다.

| 구성 요소 | Python | 환경 | 기본 GPU | 포트 |
|---|---|---|---|---|
| GR00T N1.7 | 3.12 | `Issac-GR00T-N17/.venv` | 0 | 5557 |
| GR00T N1.6 | 3.10 | `Isaac-GR00T-N16/.venv` | 1 | 5556 |
| Protoss | 3.12 | `Protoss/.venv` | CPU | 5555 |
| RoboLab + Isaac Sim 5.1 | 3.11 | `/workspace/RoboLab/.venv-51` | 2 | client |

기본 예시는 GPU 3개를 사용한다. GPU가 적으면 아래 실행 명령의 `CUDA_VISIBLE_DEVICES`를
같은 GPU 번호로 바꿀 수 있지만 두 모델과 시뮬레이터가 동시에 들어갈 VRAM이 필요하다.
처음에는 `--num-envs 1`로 실행한다. 프로젝트 위치가 다르면 모든 터미널에서
`PROTOSS_ROOT`만 실제 경로로 바꾼다. 환경 간 패키지가 섞이지 않도록 `source activate` 대신
각 저장소의 `uv run`과 전용 Python 실행 파일을 사용한다.

```bash
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy

nvidia-smi
apt-get update
apt-get install -y git git-lfs curl build-essential ffmpeg \
  vulkan-tools libvulkan1 libglvnd0 libgl1 libegl1 libglu1-mesa \
  libxt6 libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 \
  libxcursor1 libxi6 libxkbcommon0
ldconfig

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git lfs install

if [ ! -d "$PROTOSS_ROOT" ]; then
  git clone https://github.com/siri2100/Project_Protoss.git "$PROTOSS_ROOT"
fi
cd "$PROTOSS_ROOT"
```

현재 Git 추적에서 모델 폴더와 `experiment/`를 제외했으므로, 새로 clone한 프로젝트에는
모델 소스를 별도로 설치해야 한다. 실제 폴더명은 N1.7의 **`Issac`**, N1.6의 **`Isaac`**에 유의한다.

## 2. GR00T-N1.7 설치

새 설치는 이 프로젝트에서 확인한 N1.7 소스 커밋을 사용한다.
기존 폴더가 있으면 clone/checkout을 생략하므로 로컬 수정은 유지된다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT"
if [ ! -d Issac-GR00T-N17 ]; then
  git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git Issac-GR00T-N17
  git -C Issac-GR00T-N17 checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
fi
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
test -f pyproject.toml || { echo "N1.7 소스 폴더에 pyproject.toml이 없습니다: $PWD"; exit 1; }
git submodule update --init --recursive
git lfs pull --include="scripts/deployment/dgpu/wheels/**"
uv sync --python 3.12

uv run python -c "import gr00t, torch; print(gr00t.__file__); print('CUDA:', torch.cuda.is_available())"
)
```

FFmpeg는 4~7 버전을 사용한다. N1.7의 torchcodec은 FFmpeg 8을 지원하지 않는다.
GPU 의존성 설치가 실패하면 해당 저장소의 [설치 안내](https://github.com/NVIDIA/Isaac-GR00T#installation)를 확인한다.

## 3. GR00T-N1.6 설치

N1.6은 `n1.6.1-release`와 별도 Python 환경을 사용한다.
아래 블록은 새 Bash 터미널에서도 독립적으로 실행할 수 있다. 프로젝트 위치가 다르면
`PROTOSS_ROOT`를 실제 경로로 바꾼다. 괄호 안의 `set -e`는 경로 이동이나 설치가 실패하면
나머지 명령을 중단하며, 터미널 자체는 종료하지 않는다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT"
if [ ! -d Isaac-GR00T-N16 ]; then
  git clone --recurse-submodules --branch n1.6.1-release \
    https://github.com/NVIDIA/Isaac-GR00T.git Isaac-GR00T-N16
fi
cd "$PROTOSS_ROOT/Isaac-GR00T-N16"
test -f pyproject.toml || { echo "N1.6 소스 폴더에 pyproject.toml이 없습니다: $PWD"; exit 1; }
git submodule update --init --recursive
uv sync --python 3.10
uv pip install --python .venv/bin/python -e .

uv run python -c "import gr00t, torch; print(gr00t.__file__); print('CUDA:', torch.cuda.is_available())"
)
```

두 버전의 `gr00t` 패키지를 하나의 가상환경에 함께 설치하지 않는다.

## 4. Hugging Face 로그인과 DROID weight 다운로드

먼저 Hugging Face 계정으로 [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)의
접근 승인을 받는다. 각 DROID 모델 페이지에도 접근 조건이 있으면 동의한다.
로그인은 CLI 프롬프트에 토큰을 입력하며 README나 소스 파일에 토큰을 저장하지 않는다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV

for model_dir in Issac-GR00T-N17 Isaac-GR00T-N16; do
  test -f "$PROTOSS_ROOT/$model_dir/pyproject.toml" || {
    echo "모델 소스 폴더를 확인하세요: $PROTOSS_ROOT/$model_dir (README 2·3단계 설치 필요)"
    exit 1
  }
done

cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run hf auth login
uv run hf auth whoami
HF_HUB_ENABLE_HF_TRANSFER=0 uv run hf download nvidia/Cosmos-Reason2-2B config.json \
  --local-dir "$PROTOSS_ROOT/Protoss/access-check/cosmos"
HF_HUB_ENABLE_HF_TRANSFER=0 uv run hf download nvidia/GR00T-N1.7-DROID \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID"

cd "$PROTOSS_ROOT/Isaac-GR00T-N16"
uv run hf auth whoami
HF_HUB_ENABLE_HF_TRANSFER=0 uv run hf download nvidia/GR00T-N1.6-DROID \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.6-DROID"
)
```

같은 `HF_HOME`을 사용하므로 로그인은 공유된다. N1.6에서 `whoami`가 실패하면
해당 환경에서 `uv run hf auth login`을 실행한다. 아래 실행 명령은 다운로드한 로컬 경로를
`--model-path`에 전달한다. 처음 시작할 때 backbone 관련 파일이 추가로 다운로드될 수 있다.

`hf_transfer`가 설치되지 않은 환경에서도 다운로드할 수 있도록 각 다운로드 명령에
`HF_HUB_ENABLE_HF_TRANSFER=0`을 직접 지정했다. `Fast download using 'hf_transfer' is enabled`
오류가 발생하면 로그인은 유지한 채 아래 블록으로 두 DROID 다운로드만 다시 실행한다.
`git credential helper` 관련 경고는 이 다운로드 오류의 원인이 아니다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV

cd "$PROTOSS_ROOT/Issac-GR00T-N17"
HF_HUB_ENABLE_HF_TRANSFER=0 uv run hf download nvidia/GR00T-N1.7-DROID \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID"

cd "$PROTOSS_ROOT/Isaac-GR00T-N16"
HF_HUB_ENABLE_HF_TRANSFER=0 uv run hf download nvidia/GR00T-N1.6-DROID \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.6-DROID"
)
```

입력 영상의 시점도 확인한다.

```bash
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PROTOSS_ROOT"])
path = root / "Protoss/checkpoints/GR00T-N1.7-DROID/processor_config.json"
config = json.loads(path.read_text())
modalities = config["processor_kwargs"]["modality_configs"]["oxe_droid_relative_eef_relative_joint"]
print("N1.7 video offsets:", modalities["video"]["delta_indices"])
print("N1.7 state keys:", modalities["state"]["modality_keys"])
assert modalities["video"]["delta_indices"] == [0], "이 RoboLab 실행 예시는 현재 영상 T=1을 전제로 합니다"
PY
```

공개 [N1.7-DROID processor 설정](https://huggingface.co/nvidia/GR00T-N1.7-DROID/blob/main/processor_config.json)의
영상 시점은 `[0]`이고, [RoboLab 클라이언트](https://github.com/NVlabs/RoboLab/blob/main/policies/gr00t/client.py)도
현재 영상을 `T=1`로 보낸다. 다른 checkpoint가 `[-15,0]`을 요구하면 실제 과거 영상을
보내는 클라이언트가 필요하며, Protoss는 영상을 복제해서 채우지 않는다.

## 5. RoboLab 설치

RoboLab은 프로젝트 폴더 바깥의 `/workspace/RoboLab`에 설치한다.
[공식 의존성 구성](https://github.com/NVlabs/RoboLab/blob/main/pyproject.toml)의 `isaac51` extra를 사용하고,
Isaac Sim 5.0 환경이 있다면 섞지 않고 `.venv-51`을 별도로 만든다.

```bash
cd /workspace
if [ ! -d RoboLab ]; then
  git clone https://github.com/NVlabs/RoboLab.git RoboLab
fi
cd /workspace/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --python 3.11 --extra isaac51

UV_PROJECT_ENVIRONMENT=.venv-51 uv run --extra isaac51 python -c \
  "import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab import OK')"
```

RoboLab의 `policies/gr00t/client.py`는 원래 N1.7의 `msgpack_numpy` 형식을 사용한다.
이전에 N1.6용 serializer로 수정했다면 원본 N1.7 클라이언트를 먼저 복원한다.
Protoss가 N1.6 서버 통신을 변환하므로 여기서는 N1.6용 패치를 적용하지 않는다.

## 6. Protoss 설치

```bash
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
uv venv --python 3.12 Protoss/.venv
uv pip install --python Protoss/.venv/bin/python -r Protoss/requirements.txt
Protoss/.venv/bin/python Protoss/main_v1.0.py --help
Protoss/.venv/bin/python -m unittest discover -s Protoss -p 'test_*.py' -v
```

Protoss 자체에는 PyTorch나 CUDA가 필요하지 않다. 각 upstream 서버가 모델 추론을 담당한다.

## 7. Inference: 터미널 4개 실행

다음 순서대로 실행하며 각 서버의 준비 완료 메시지를 확인한 뒤 다음 터미널을 시작한다.
각 터미널에서 경로 및 Hugging Face cache 변수를 다시 지정한다.

### 터미널 1 — N1.7-DROID 서버, 포트 5557

```bash
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID" \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
```

### 터미널 2 — N1.6-DROID 서버, 포트 5556

```bash
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"
cd "$PROTOSS_ROOT/Isaac-GR00T-N16"
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.6-DROID" \
  --embodiment-tag OXE_DROID \
  --device cuda:0 --host 127.0.0.1 --port 5556 --use-sim-policy-wrapper
```

`CUDA_VISIBLE_DEVICES=1`로 제한한 프로세스 안에서는 실제 GPU 1이 `cuda:0`으로 보인다.
GPU 1개 구성에서는 위 값을 `CUDA_VISIBLE_DEVICES=0`으로 바꾼다.

### 터미널 3 — main_v1.0.py blending 서버, 포트 5555

```bash
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv/bin/python Protoss/main_v1.0.py \
  --n17-endpoint tcp://127.0.0.1:5557 \
  --n16-endpoint tcp://127.0.0.1:5556 \
  --alpha 0.5 --horizon 8 --timeout-ms 120000 \
  --host 127.0.0.1 --port 5555
```

`Protoss ready`가 출력되면 두 서버의 modality config 조회와 기본 호환성 검사가 완료된 것이다.
비율을 바꾸려면 이 서버를 종료하고 `--alpha`를 바꿔 재시작한다.
v1.0은 두 모델을 순서대로 추론하며, alpha가 0 또는 1이어도 두 서버가 필요하다.

### 터미널 4 — RoboLab에서 blended action 실행

```bash
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
cd /workspace/RoboLab

CUDA_VISIBLE_DEVICES=0 UV_PROJECT_ENVIRONMENT=.venv-51 \
uv run --extra isaac51 python policies/gr00t/run.py \
  --headless --device cuda:0 \
  --remote-host 127.0.0.1 --remote-port 5555 \
  --task BananaOnPlateTask \
  --num-envs 1 --num-runs 1 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name protoss_v1_alpha05_smoke
```

RoboLab은 모델 서버의 5556/5557이 아닌 **Protoss의 5555**로 연결한다.
GPU가 1~2개이면 `CUDA_VISIBLE_DEVICES=2`를 사용 가능한 GPU 번호로 바꾼다.
`--open-loop-horizon`은 Protoss의 `--horizon` 이하로 지정한다.
정상 동작 후 환경 수와 task 수를 늘리고, 영상 저장은 `--video-mode sensor`로 변경한다.
평가 결과는 `/workspace/RoboLab/output/protoss_v1_alpha05_smoke/` 아래에 저장된다.
같은 output 이름을 재사용하면 기존 완료 episode를 건너뛸 수 있으므로 새 실험은 이름도 바꾼다.
실행 옵션은 [RoboLab runner](https://github.com/NVlabs/RoboLab/blob/main/robolab/eval/runner.py)를 기준으로 한다.

## 8. 저장한 observation으로 한 번 추론

RoboLab 대신 실제 관측을 NPZ로 저장해 추론할 수도 있다.
N1.7/N1.6 서버 2개만 실행한 뒤 아래 명령을 사용한다. Protoss 서버는 실행하지 않아도 된다.

```bash
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv/bin/python Protoss/main_v1.0.py \
  --n17-endpoint tcp://127.0.0.1:5557 \
  --n16-endpoint tcp://127.0.0.1:5556 \
  --observation observation.npz \
  --output blended_actions.npz --alpha 0.5 --horizon 8
```

`observation.npz`에 필요한 키와 shape는 다음과 같다. B는 batch 크기이며 모든 키에서 동일해야 한다.

| 키 | Shape / dtype |
|---|---|
| `video.exterior_image_1_left` | `[B,1,H,W,3]`, RGB uint8 |
| `video.wrist_image_left` | `[B,1,H,W,3]`, RGB uint8 |
| `state.eef_9d` | `[B,1,9]`, float32 |
| `state.joint_position` | `[B,1,7]`, float32 |
| `state.gripper_position` | `[B,1,1]`, float32 |
| `annotation.language.language_instruction` | `[B]`, 문자열 배열 |

관측 배열이 준비되면 `np.savez("observation.npz", **observation)`으로 저장한다.
언어는 `np.array(["put the banana on the plate"])`처럼 문자열 dtype을 사용한다.
EEF pose는 로봇 기준 좌표와 DROID 회전 표현이 맞아야 하므로 RoboLab의
`compute_eef_9d()` 및 `_extract_observation()`을 참고한다.

출력 확인:

```bash
Protoss/.venv/bin/python - <<'PY'
import numpy as np

with np.load("blended_actions.npz", allow_pickle=False) as actions:
    for key in actions.files:
        print(key, actions[key].shape, actions[key].dtype)
    print("First joint target:", actions["action.joint_position"][0, 0])
    print("First gripper target:", actions["action.gripper_position"][0, 0])
PY
```

기본 출력은 joint `[B,8,7]`, gripper `[B,8,1]`이다. 값은 decode된 absolute 목표이고,
RoboLab은 gripper 혼합 결과를 `>0.5` 기준으로 이진화한다.

## 9. 자주 확인할 항목

- **401/403 또는 GatedRepoError:** 같은 `HF_HOME`에서 `hf auth whoami`와 Cosmos 접근 승인을 확인한다.
- **모델 import 오류:** N1.7은 Python 3.12, N1.6은 Python 3.10 환경에서 각각 실행한다.
- **서버 timeout:** N1.7 5557, N1.6 5556의 준비 완료를 확인한다. 초기 모델 다운로드 중이면 기다린다.
- **영상 horizon 오류:** 실제 processor config와 입력 T를 맞춘다. 이 DROID/RoboLab 예시는 T=1이다.
- **CUDA OOM:** `--num-envs 1`로 줄이고 GPU 배치 및 모델 동시 메모리 사용을 확인한다.
- **Isaac Sim driver 로딩 오류:** toolkit의 stub `libcuda`가 먼저 로드되는지 `LD_LIBRARY_PATH`를 확인한다.
- **RTC 옵션 오류:** v1.0은 일반 action 생성만 지원하며 RTC를 지원하지 않는다.

두 모델의 관절 순서, 단위, 실제 관측·제어 주기는 같아야 한다.
Protoss는 action delta_indices의 대응을 검사하지만 실제 제어 주파수까지 검증하지 않는다.
원격 RoboLab 접속 시 Protoss를 `--host 0.0.0.0`으로 실행하고 RoboLab의 `--remote-host`를
Protoss 서버 IP로 지정한다. upstream 서버가 다른 호스트이면 각 endpoint도 변경한다.

CPU 검증은 혼합·시간 정렬·직렬화 테스트를 대상으로 한다.
이 문서의 전체 GPU 설치, 두 모델 추론, RoboLab 평가는 현재 작업 환경에서 실행 검증하지 않았다.
