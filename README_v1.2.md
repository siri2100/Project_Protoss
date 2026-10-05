# Project Protoss v1.2 — 설치부터 학습·평가까지

이 문서의 명령을 위에서부터 실행하면 **공개 DROID 성공 demonstration 다운로드 → NPZ export → 두 pretrained 모델의 blending 캐시 → 추가 네트워크 학습 → held-out test 평가**를 진행할 수 있다. 별도의 observation 파일을 직접 만들거나 v1.1 README를 읽을 필요가 없다. 마지막에는 RoboLab에서 v1.1 대비 closed-loop 성공률을 비교하는 절차도 있다.

학습 대상은 blending 이후의 `ResidualFlow` MLP다. GR00T-N1.7-DROID와 π0.5-DROID pretrained weight는 고정한다.

```text
DROID LeRobot v3 subset → episode 단위 train/val/test → GR00T 관측+demo action NPZ
                                  ↓ 두 pretrained 서버로 offline cache 생성
                 alpha × GR00T action + (1-alpha) × π action
                                  ↓ train/val cache
                           ResidualFlow MLP 학습
                                  ↓ test cache
                    v1.1 blend vs v1.2 refined action 오차
                                  ↓ 선택 사항
                          RoboLab task 성공률 평가
```

기본값: alpha=0.5, horizon=8, stride=8, hidden=256, batch=64, seed=42. 공개 데이터는 `lerobot/droid_1.0.1`의 첫 metadata shard에서 **성공 여부가 참이고 언어가 있는 episode 30개**를 선택한다. 그 안에서 episode를 섞어 train 24 / validation 3 / test 3으로 분리한다. 이 subset은 실행 가능한 첫 실험용이며 task별 균형이나 평가 환경과의 일치를 보장하지 않는다.

RunPod의 Ubuntu 22.04/24.04 + CUDA 환경을 기준으로 한다. 캐시 생성은 VRAM 48GB급 한 장과 RAM 32GB 이상을 시작 예산으로 잡는다(실측 보장값 아님). 추가 MLP만 학습하면 CPU 또는 작은 GPU로도 가능하다. RoboLab에는 RTX rendering을 지원하는 GPU와 Vulkan이 필요하므로 A100/H100에서 모델 학습이 된다는 사실만으로 simulator 실행이 가능한 것은 아니다.

이 문서의 명령은 원격 Linux 서버용이다. 로컬 macOS에서 모델/Isaac Sim 서버를 실행하는 절차가 아니다. 프로젝트는 아래 clone 대상의 **v1.2 파일이 포함된 revision**이어야 한다. 로컬에서 수정한 파일을 아직 push하지 않았다면 같은 폴더 구조로 서버에 복사한 뒤 설치를 진행한다.

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
│   ├── main_v1.0.py                  # v1.2에서도 GR00T 통신 함수를 재사용
│   ├── main_v1.2.py
│   ├── requirements_v1.2.txt
│   ├── .venv-v12/
│   └── checkpoints/GR00T-N1.7-DROID/
└── README_v1.2.md
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
모델 서버는 GPU가 여러 개면 아래 서버 명령의 `CUDA_VISIBLE_DEVICES`를 바꿔 분산한다.
Isaac Sim은 기본 단일 GPU 설정으로 시작하며, 여러 GPU 사용 시 CUDA/Vulkan 장치 매핑을 별도로 확인한다.
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
  git clone https://github.com/siri2100/Project_Protoss.git "$PROTOSS_ROOT"
fi
test -f "$PROTOSS_ROOT/Protoss/main_v1.2.py"
test -f "$PROTOSS_ROOT/Protoss/main_v1.0.py"
test -f "$PROTOSS_ROOT/Protoss/prepare_droid_v1_2.py"
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
# 이 블록 안에서만 시스템·사용자 Git 설정과 주입된 인증 설정을 격리한다.
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
unset GIT_CONFIG GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS GIT_ASKPASS SSH_ASKPASS GIT_TERMINAL_PROMPT
cd "$PROTOSS_ROOT"
if [ -e Issac-GR00T-N17 ] && [ ! -d Issac-GR00T-N17/.git ]; then
  backup=$(mktemp -d "$PROTOSS_ROOT/Issac-GR00T-N17.incomplete.XXXXXX")
  mv Issac-GR00T-N17 "$backup/source"
  echo "불완전한 기존 폴더 보관: $backup/source"
fi
if [ ! -d Issac-GR00T-N17/.git ]; then
  staging=$(mktemp -d "$PROTOSS_ROOT/.n17-clone.XXXXXX")
  GIT_LFS_SKIP_SMUDGE=1 git -c credential.helper= -c http.extraHeader= \
    clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "$staging/source"
  git -C "$staging/source" lfs install --local
  GIT_LFS_SKIP_SMUDGE=1 git -C "$staging/source" checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
  mv "$staging/source" Issac-GR00T-N17
  rmdir "$staging"
fi
test "$(git -C Issac-GR00T-N17 rev-parse HEAD)" = 51d4c89f72fda44cbf77285c6a8114b52676b8a1 || {
  echo "N1.7 소스 버전이 다릅니다. 기존 폴더를 백업하고 지정 커밋으로 설치하세요." >&2
  exit 1
}
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
test -f pyproject.toml
git -c credential.helper= -c http.extraHeader= submodule update --init --recursive
git -c credential.helper= -c http.extraHeader= lfs pull --include="scripts/deployment/dgpu/wheels/**"
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

### GitHub clone에서 Username 오류가 발생한 경우

Isaac-GR00T 소스는 공개 저장소다. `could not read Username`은 GitHub 모델 접근 승인과
별개인 Git HTTPS 인증 실패다. 사용자·시스템 Git의 URL 변환, 인증 헤더, credential 설정이나
프록시의 인증 응답 등이 원인일 수 있으며, 이 메시지만으로 원인을 확정할 수 없다.
위 설치 블록은 subshell에서 해당 Git 설정을 격리하고 임시 경로에서 clone·checkout을
완료한 뒤 설치 폴더로 옮긴다. `.git`이 없는 기존 폴더는 삭제하지 않고 백업한다.

설정 격리 후에도 GitHub Username을 요구하면 입력을 취소한다. 공개 저장소에 GitHub 비밀번호를
입력해서 해결하는 절차가 아니다. 서버의 프록시·네트워크 접근과 실제 오류 응답을 확인해야 한다.
실패한 임시 clone 폴더는 남을 수 있지만 다음 실행의 설치 경로와 충돌하지 않는다.
[Git 환경변수 문서](https://git-scm.com/docs/git),
[공개 upstream 저장소](https://github.com/NVIDIA/Isaac-GR00T).

### Cosmos 접근 확인에서 401 / GatedRepoError가 발생한 경우

`Cosmos-Reason2-2B/resolve/main/config.json`의 401은 Cosmos 접근 확인 실패다.
`hf auth whoami` 성공은 로그인 확인이며 gated 모델 접근 승인을 확인하지 않는다.

1. 브라우저에서 [Cosmos 모델 페이지](https://huggingface.co/nvidia/Cosmos-Reason2-2B)에 로그인하고 약관 동의·접근 신청을 완료한다. 승인 대기 상태라면 승인 후 진행한다.
2. **승인받은 동일 계정**의 [읽기 토큰](https://huggingface.co/settings/tokens)을 준비한다. Fine-grained 토큰은 해당 gated 모델을 읽을 권한도 허용해야 한다.
3. 아래 블록으로 서버에서 다시 로그인하고 실패한 단계부터 실행한다. 토큰은 `hf auth login`의 입력 프롬프트에 붙여넣는다.

`HF_TOKEN` 환경변수는 저장된 로그인 토큰보다 우선한다. 아래 복구 블록은 subshell 안에서만
기존 토큰 환경변수를 해제하며, `HF_HOME`을 설치·서버 실행과 동일하게 사용한다.
[Hugging Face gated 모델 안내](https://huggingface.co/docs/hub/models-gated),
[인증 환경변수 안내](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync hf auth login
uv run --no-sync hf auth whoami
uv run --no-sync hf download nvidia/Cosmos-Reason2-2B config.json \
  --local-dir "$PROTOSS_ROOT/Protoss/access-check/cosmos"
uv run --no-sync hf download nvidia/GR00T-N1.7-DROID \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID"
)
```

계속 실패하면 `whoami` 계정과 웹에서 승인받은 계정, 토큰의 읽기 권한을 다시 확인한다.
Cosmos 접근 확인을 생략해도 N1.7 모델 로딩에서 같은 권한 오류가 발생할 수 있다.
새 서버 터미널에도 오래된 `HF_TOKEN`이 설정되어 있다면 해제하거나 승인된 토큰으로 교체한다.

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
  GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
    https://github.com/xuningy/openpi.git openpi
  git -C openpi checkout aa6420561529593114160d05e5ad155792b272f3
fi
test "$(git -C openpi rev-parse HEAD)" = aa6420561529593114160d05e5ad155792b272f3 || {
  echo "OpenPI 소스 버전이 다릅니다. 기존 폴더를 백업하고 지정 커밋으로 설치하세요." >&2
  exit 1
}
GIT_LFS_SKIP_SMUDGE=1 git -C openpi submodule update --init --recursive
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

## 5. Protoss v1.2 의존성 설치

OpenPI 모델 전체가 아니라 가벼운 `openpi-client`만 Protoss 환경에 설치한다.
추가 MLP 학습을 위한 PyTorch와 DROID 준비를 위한 PyArrow/Pandas/SciPy도 설치한다.
Linux의 PyTorch는 `2.9.1+cu128`로 고정한다. 드라이버 570 환경에서 CUDA 13 wheel이
설치되는 것을 방지하기 위해 아래 명령의 CUDA 12.8 index를 반드시 함께 사용한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT"
test -f openpi/packages/openpi-client/pyproject.toml
if [ ! -x Protoss/.venv-v12/bin/python ]; then
  uv venv --python 3.12 Protoss/.venv-v12
fi
uv pip install --python Protoss/.venv-v12/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  -r Protoss/requirements_v1.2.txt -e "$PROTOSS_ROOT/openpi/packages/openpi-client"
Protoss/.venv-v12/bin/python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "wheel CUDA:", torch.version.cuda)
assert torch.version.cuda == "12.8", "CUDA 12.8 PyTorch wheel이 필요합니다"
print(torch.ones(1, device="cuda"), torch.cuda.get_device_name(0))
PY
Protoss/.venv-v12/bin/python Protoss/main_v1.2.py --help
Protoss/.venv-v12/bin/python -m unittest discover -s Protoss -p 'test_main_v1_*.py' -v
)
```

`main_v1.2.py`와 통신 함수를 제공하는 `main_v1.0.py`를 같은 폴더에 둔다.


## 6. DROID 데이터 다운로드 — 모델 서버 없이 실행

수동 원본 변환 대신 Hugging Face의 공개 **DROID LeRobot v3.0** 데이터를 직접 읽는다. 전체 데이터셋을 받지 않고 선택 episode가 속한 data shard와 exterior/wrist video shard만 받는다. 영상은 shard 단위 다운로드라 30개 episode보다 훨씬 많은 frame이 포함될 수 있다. 처음에는 디스크에 모델·환경 설치 공간과 별도로 **데이터용 10GB 이상**을 확보한다. 실제 다운로드 파일 수/크기는 dataset revision에 따라 달라진다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python Protoss/prepare_droid_v1_2.py download \
  --dataset-dir data/v1.2/droid --num-episodes 30
)
```

성공 출력: `Subset ready: 30 episodes ... revision=...`. `data/v1.2/droid/subset.json`에는 선택된 episode와 resolved Hugging Face commit이 저장된다. 재실행은 같은 revision을 사용해 다운로드 cache를 재사용한다. 정확히 재현하려면 첫 실험의 `subset.json`에 기록된 SHA를 `--revision SHA`로 전달한다. Episode 수/revision을 바꾸려면 새 dataset 디렉터리를 사용한다.

이 경로에서는 raw HDF5, 언어 annotation JSON, LeRobot 설치, 별도의 camera mapping 작업이 필요 없다. `action.joint_position`과 `action.gripper_position`을 **정답 absolute command**로 읽으며 관측 joint 값이나 velocity를 정답으로 대신 사용하지 않는다. 데이터 계약은 [공개 dataset](https://huggingface.co/datasets/lerobot/droid_1.0.1)과 고정 GR00T 소스의 `scripts/download_droid_sample.py`를 확인했다.

## 7. 두 pretrained 서버 시작 — 각각 별도 터미널

### 터미널 A: GR00T N1.7, 포트 5557

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
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID" \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
)
```

### 터미널 B: π0.5-DROID jointpos, 포트 8000

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export OPENPI_DATA_HOME="$PROTOSS_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/openpi"
uv run --no-sync python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
uv run --no-sync scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir="$PROTOSS_ROOT/pi05_droid_jointpos"
)
```

GPU가 두 개라면 터미널 B만 `CUDA_VISIBLE_DEVICES=1`로 변경할 수 있다. 터미널 A/B는 캐시 생성이 끝날 때까지 유지한다. 같은 GPU에서 Python 함수를 순서대로 호출해도 **두 서버의 weight는 계속 상주**한다. OOM이면 GPU/호스트를 분리하거나 더 큰 GPU를 사용한다. JAX preallocation을 끄는 설정은 메모리 사용량의 강제 상한이 아니다. [JAX 메모리 안내](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

## 8. Export와 캐시 생성 — 터미널 C

먼저 GR00T 서버에서 실제 modality config를 읽어 export한다. 현재 프레임만 필요한 설정과 과거 프레임이 필요한 설정 모두 처리하며, episode 시작에서 부족한 과거 이미지를 복제하지 않고 해당 시점을 건너뛴다. EEF 9D 변환은 **고정 GR00T 소스의 `droid_frame.py` 함수**를 직접 사용한다. 영상은 FFmpeg로 정확한 episode 시작 시점부터 RGB로 디코딩한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python Protoss/prepare_droid_v1_2.py export \
  --dataset-dir data/v1.2/droid --output-dir data/v1.2/export \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --n17-endpoint tcp://127.0.0.1:5557 \
  --horizon 8 --stride 8 --seed 42
for split in train val test; do
  Protoss/.venv-v12/bin/python Protoss/main_v1.2.py --mode cache \
    --input-dir "data/v1.2/export/$split" --cache-dir "data/v1.2/cache/$split" \
    --n17-endpoint tcp://127.0.0.1:5557 --pi-uri ws://127.0.0.1:8000 \
    --alpha 0.5 --horizon 8 --timeout-ms 300000 --resume-cache
done
)
```

성공 출력: `Export ready: .../manifest.json`, 이어서 각 split의 `Cached ... files`. `data/v1.2/export/manifest.json`에서 train/val/test episode 목록, 관측 config, frame/sample 수를 확인할 수 있다. `stride=8`은 관측 샘플을 8 frame마다 선택한다는 뜻이며 **target chunk 내부의 action은 계속 연속 15Hz**다. 첫 검증 뒤 더 촘촘한 데이터가 필요하면 새 export 디렉터리에 `--stride 1`로 생성한다.

Export 출력 디렉터리가 비어 있지 않으면 중단한다. 실패한 export는 원인을 해결한 뒤 `--output-dir data/v1.2/export_retry`처럼 새 경로로 실행하고 cache 입력도 그 경로로 바꾼다. 캐시 도중 중단되면 **for 블록만** 다시 실행한다. `--resume-cache`는 source 파일 hash와 alpha/horizon이 같은 완성 파일만 건너뛴다. Pretrained weight나 전처리를 바꾼 경우에는 새 cache 디렉터리를 사용해야 한다.

각 cache 파일에는 `[B,H,8]` blended action과 demo target, `[B,8]` 현재 state, `[B,H]` padding mask, episode ID가 저장된다. 이미지는 저장하지 않는다. 마지막 chunk는 episode를 넘지 않도록 padding하며 loss/평가에서 mask=0을 제외한다.

캐시 생성이 끝나면 터미널 A/B를 Ctrl+C로 종료할 수 있다. 아래 학습과 offline 평가는 모델 서버에 접속하지 않는다.

## 9. 추가 네트워크 학습

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python Protoss/main_v1.2.py --mode train \
  --train-cache data/v1.2/cache/train --val-cache data/v1.2/cache/val \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt \
  --alpha 0.5 --horizon 8 --hidden 256 \
  --epochs 50 --batch-size 64 --lr 1e-4 --device cuda --seed 42
)
```

GPU 없이 학습하려면 `--device cpu`만 바꾼다. 성공 출력: `Saved best checkpoint: ...`. 산출물:

- `Protoss/checkpoints/refiner_v1.2.pt`: validation flow MSE가 가장 낮은 모델, train-only normalization, alpha/horizon, train/val episode ID.
- `Protoss/checkpoints/refiner_v1.2.history.json`: epoch별 train/validation flow MSE.

학습 데이터와 validation episode가 겹치면 중단한다. 기본 MLP는 약 11.8만 파라미터이고 두 pretrained 모델은 학습하지 않는다. Train/val cache 전체를 CPU RAM에 로드하므로 매우 큰 데이터셋에서는 RAM 사용량을 확인한다. 같은 checkpoint 경로에서 학습을 다시 시작하면 기존 checkpoint와 history를 덮어쓴다. 다른 실험은 경로를 바꾼다. Optimizer 상태를 이용한 학습 재개는 지원하지 않는다.

Loss는 [OpenPI의 flow-matching 구현](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)을 residual action에 적용한다.

```text
a_blend = alpha * a_GR00T + (1-alpha) * a_pi
r = (a_demo - a_blend) / train_action_scale
noise ~ N(0,I), t ~ Beta(1.5,1) * 0.999 + 0.001
x_t = t * noise + (1-t) * r
loss = masked_mean((v_theta(x_t, normalized_blend, normalized_state, t) - (noise-r))²)
```

`ResidualFlow`는 전체 chunk의 noisy residual, blended action, 현재 state, time을 입력받는 MLP다. 이미지/언어는 두 pretrained VLA 출력을 통해 간접 반영된다. 추론은 t=1의 noise에서 Euler 10 step으로 t=0까지 진행한 뒤 residual을 blend에 더한다. 원본 π0.5 네트워크를 재현하거나 fine-tune한 구조는 아니다.

## 10. Held-out test 평가 — pretrained 서버 불필요

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python Protoss/main_v1.2.py --mode eval \
  --test-cache data/v1.2/cache/test \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt \
  --alpha 0.5 --horizon 8 --flow-steps 10 --batch-size 64 \
  --device cuda --eval-seeds 42 43 44 \
  --metrics-output results/v1.2/offline_test.json
)
```

성공 출력: `Saved evaluation: results/v1.2/offline_test.json`. JSON의 `v1.1_blend`와 `v1.2_refined`를 비교한다. 각 seed의 결과는 `per_seed`, 평균·표준편차는 `v1.2_refined`에 기록된다. 같은 seed와 batch size/device를 유지해야 비교를 재현하기 쉽다.

| Metric | 의미 |
|---|---|
| `joint_mae_rad`, `joint_rmse_rad` | 7개 joint의 demo command 오차, rad |
| `gripper_mae`, `gripper_rmse` | 연속 gripper command 오차, 0–1 |
| `normalized_action_mse` | train split의 scale로 정규화한 8차원 평균 제곱 오차 |
| `refinement_seconds_per_chunk` | 추가 네트워크 보정 시간; VLA 서버 latency는 제외 |

유효 timestep에 대해서만 계산하며 train/val과 test episode가 겹치면 중단한다. 이전 버전 checkpoint에 episode ID가 없다면 현재 trainer로 다시 학습해야 한다. **이 평가는 기록된 관측에서의 action imitation 오차다.** 이 값이 줄어든 것만으로 실제 로봇 task 성공률이 높아졌다고 판단하지 않는다. 이미 두 pretrained 모델이 학습한 DROID episode일 수 있으므로 VLA 전체의 unseen-data 평가로 해석하지 않는다.

## 11. 학습된 정책 추론

터미널 A/B를 7절 명령으로 다시 시작하고, 터미널 C에서 다음 proxy를 띄운다. RoboLab 평가를 위해 refiner는 CPU에 두어 simulator의 GPU 메모리를 아낀다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python Protoss/main_v1.2.py --mode infer \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt \
  --n17-endpoint tcp://127.0.0.1:5557 --pi-uri ws://127.0.0.1:8000 \
  --alpha 0.5 --horizon 8 --flow-steps 10 --device cpu --seed 42 \
  --host 127.0.0.1 --port 5555 --timeout-ms 300000
)
```

`Protoss ready`가 나오면 proxy 연결이 준비된 것이다. 학습 당시 alpha/horizon과 다르면 checkpoint 로딩을 거부한다. Alpha=0/1은 blending 단계의 모델 선택이며 추가 보정은 계속 적용된다. 최종 출력은 absolute joint 7개+gripper 1개다. RTC는 지원하지 않는다. Gripper threshold와 robot별 action 제한은 기존 RoboLab client/control 경로가 처리한다.

실제 test export 관측 한 개로 먼저 확인하려면 proxy 대신 아래를 실행한다. 추가 observation 파일을 작성할 필요가 없다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
SAMPLE=$(find data/v1.2/export/test -name '*.npz' -print -quit)
test -n "$SAMPLE"
Protoss/.venv-v12/bin/python Protoss/main_v1.2.py --mode infer \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt \
  --observation "$SAMPLE" --output results/v1.2/refined_sample.npz \
  --alpha 0.5 --horizon 8 --device cpu --seed 42 --timeout-ms 300000
)
```

## 12. RoboLab closed-loop 성공률 평가 — 선택 사항

Offline 평가는 10절에서 끝난다. 실제 simulated task 성공률이 필요하면 다음을 추가한다. Isaac Sim 5.1 EULA를 확인한 후 아래 설치 블록을 실행한다. RTX 지원 GPU와 NVIDIA graphics/Vulkan 라이브러리가 노출된 Linux 컨테이너가 필요하다. Isaac Sim은 headless에서도 정책용 camera rendering을 수행한다. [Isaac Sim 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

### RoboLab 설치

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
  git clone https://github.com/NVlabs/RoboLab.git RoboLab
fi
cd /workspace/RoboLab
git rev-parse HEAD | tee protoss_v12_robolab_revision.txt
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --python 3.11 --extra isaac51
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --no-sync --extra isaac51 \
  python -c "import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab import OK')"
)
```

RoboLab은 설치 당시 main을 사용하므로 기록된 revision을 두 비교 실험에서 동일하게 유지한다. 모델 두 개와 simulator까지 한 GPU에 올리는 경우 48GB로 충분하다고 보장하지 않는다. 먼저 `--num-envs 1`로 시작하고 모델과 simulator를 GPU/호스트별로 분리할 수 있다.

### v1.2 평가 — 터미널 D

7절의 두 upstream 서버와 11절의 proxy가 실행 중이어야 한다.

```bash
(
set -e
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV CUDA_VISIBLE_DEVICES
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
cd /workspace/RoboLab
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --no-sync --extra isaac51 \
  python policies/gr00t/run.py \
  --headless --device cuda:0 --remote-host 127.0.0.1 --remote-port 5555 \
  --task BananaOnPlateTask BananasInBinOneMoreTask BananasInCrateTask \
  --num-envs 1 --num-runs 10 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name protoss_v12_run01_alpha05
)
```

첫 확인은 `--num-runs 1`로 줄일 수 있다. 완료 후 `/workspace/RoboLab/output/protoss_v12_run01_alpha05/`에 episode 결과와 summary가 생성된다. [RoboLab runner](https://github.com/NVlabs/RoboLab/blob/main/robolab/eval/runner.py)는 같은 output 이름의 완료 episode를 재사용하므로 새로운 실험은 run 번호를 바꾼다. Simulator에서의 gripper action 후처리는 [GR00T client](https://github.com/NVlabs/RoboLab/blob/main/policies/gr00t/client.py)의 0.5 threshold를 따른다.

### 같은 설정으로 v1.1 baseline 평가

v1.2 평가가 완료되면 **터미널 C의 proxy만** Ctrl+C로 종료하고, 같은 자리에서 아래 baseline을 시작한다. 두 upstream 서버는 유지한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python Protoss/main_v1.1.py \
  --n17-endpoint tcp://127.0.0.1:5557 --pi-uri ws://127.0.0.1:8000 \
  --alpha 0.5 --horizon 8 --host 127.0.0.1 --port 5555 --timeout-ms 300000
)
```

터미널 D의 동일 평가 명령을 다시 실행하되 `--output-folder-name protoss_v11_run01_alpha05`로 바꾼다. Task 목록, num-envs, num-runs, horizon, instruction과 RoboLab revision은 유지한다. 두 output의 task별 success rate와 전체 success rate를 비교한다. Simulator randomization과 VLA sampling 때문에 단일 run 차이를 개선으로 단정하지 않고 여러 run으로 반복한다. DROID에서 학습한 residual이 RoboLab으로 전이된다는 보장은 없으며, 성공률이 떨어지면 baseline과 offline action metric을 함께 확인한다.

## 13. 실행 중 막히는 경우

### 학습 시작 시 NVIDIA driver is too old (found version 12080)

`12080`은 드라이버가 제공하는 CUDA API 버전 12.8이며 NVIDIA 드라이버의
`570.xxx` 문자열 자체가 아니다. 드라이버가 너무 오래되었다는 오류가 여기서 발생하면
v1.2 환경의 PyTorch wheel이 더 높은 CUDA 버전을 요구할 수 있다.
기존 설치 안내의 `torch>=2.2,<3`만으로는 CUDA wheel 버전이 고정되지 않았다.
현재 안내는 Linux에서 `torch==2.9.1+cu128`을 설치하도록 수정했다.
[PyTorch 공식 설치 명령](https://pytorch.org/get-started/previous-versions/),
[CUDA 13 드라이버 요구사항](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

캐시를 다시 만들 필요 없이 **v1.2 환경만** 아래 명령으로 복구한 뒤 9절을 다시 실행한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python -c 'import torch; print("before:", torch.__version__, torch.version.cuda)'
uv pip install --python Protoss/.venv-v12/bin/python \
  --reinstall-package torch --index https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128'
Protoss/.venv-v12/bin/python - <<'PY'
import torch
print("after:", torch.__version__, "wheel CUDA:", torch.version.cuda)
assert torch.version.cuda == "12.8"
print(torch.ones(1, device="cuda"), torch.cuda.get_device_name(0))
PY
)
```

CUDA tensor 생성까지 성공하면 9절의 `--device cuda` 학습 명령으로 진행한다.
GPU 사용 없이 먼저 학습을 시작하려면 9절에서 `--device cpu`로 변경해도 된다.

- **다운로드 401 / 403:** GR00T/Cosmos는 승인받은 Hugging Face 계정으로 2절 로그인. 공개 DROID asset이나 π checkpoint가 접근 불가하면 원격 오류를 확인하고 임의의 action checkpoint로 대체하지 않는다.
- **영상 디코딩 실패:** `ffmpeg -version` 확인. Export는 AV1을 FFmpeg로 디코딩하므로 OS의 FFmpeg에 AV1 decoder가 있어야 한다. 메타데이터 length/timestamp나 데이터 schema가 다르면 명시적 오류로 중단한다.
- **첫 캐시 요청이 느림:** JAX 첫 compile은 모델 로딩 이후 발생할 수 있다. 위 명령의 `--timeout-ms 300000`을 사용하고 서버의 OOM/compile 로그를 확인한다.
- **캐시 설정 mismatch:** alpha/horizon, 입력 hash가 바뀌면 새 cache 경로 사용. Export를 재실행할 때도 새 경로로 생성한다.
- **CUDA OOM:** `nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv -l 1`로 peak 확인. 캐시 완료 후 학습 시에는 두 pretrained 서버를 종료한다.
- **RoboLab 렌더러 crash:** `vulkaninfo --summary`로 NVIDIA GPU와 Vulkan 확인. 관리형 Pod의 host driver는 컨테이너 내부 apt로 교체할 수 없다. Graphics capability와 Isaac Sim 지원 GPU/driver를 확인한다.

## 14. 코드와 검증 범위

| 파일 | 역할 |
|---|---|
| `Protoss/prepare_droid_v1_2.py` | public subset 다운로드, server config에 맞춘 observation/target export, episode 분리 |
| `Protoss/main_v1.2.py` | pretrained blend, cache, ResidualFlow 학습, offline eval, proxy inference |
| `Protoss/requirements_v1.2.txt` | v1.2 환경 의존성 |
| `Protoss/test_main_v1_2.py` | 네트워크·cache·checkpoint CPU 테스트 |
| `Protoss/test_prepare_droid_v1_2.py` | 데이터/영상 정렬·split·pipeline CPU 테스트 |

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
cd "$PROTOSS_ROOT"
Protoss/.venv-v12/bin/python -m unittest discover -s Protoss -p 'test_*v1_2.py' -v
)
```

로컬 검증은 실제 공개 DROID metadata/data shard의 schema 확인과 synthetic 영상·데이터에 대한 export→cache→train→eval 경로를 포함한다. 실제 pretrained checkpoint를 GPU에 로딩한 캐시 생성, RunPod 환경 전체 설치, DROID 실제 학습 및 Isaac Sim closed-loop 성공률은 이 작업 환경에서 실행하지 않았다. 위 명령은 그 실행 경로를 제공하며 측정하지 않은 학습 성능/성공률을 주장하지 않는다.

2026-10-05 검증: 기존 버전 회귀 테스트를 포함한 32개 테스트 통과. 실제 FFmpeg 영상의 fractional timestamp seek, padding 제외, split 누수 검사, cache 재사용, CPU 학습/checkpoint 로딩, test 평가 및 서버 없이 실행하는 train/eval CLI를 확인했다. README의 19개 Bash 블록도 구문 검사를 통과했다.
