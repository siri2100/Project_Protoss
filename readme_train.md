# Project Protoss v1.7 — 학습

네 모델은 분할하지 않은 같은 원본 DROID 데이터 전체을 사용한다. baseline은 원본 pretrained 평가 전용이며, Model-FM/S/E는 같은 pretrained 모델에서 독립적으로 LoRA 학습한다.

| 모델 | 학습 | Loss | 구현 상태 |
|---|---|---|---|
| baseline | 없음 | 없음 | 원본 평가 가능 |
| Model-FM | LoRA | FM | 학습·평가 실행 경로 제공 |
| Model-S | LoRA | pretrained flow MSE + normalized joint velocity | 학습·평가 실행 가능 |
| Model-E | LoRA | FM (현재 Model-FM과 동일) | 학습·평가 실행 가능 |

Model-FM(v1.6.14)은 All QKV, LR=1e-6의 FM 기준 모델이다. Model-S(v1.7)는 같은 pretrained 모델에서 독립적으로 pretrained flow 보존 loss와 정규화 관절 속도 loss로 학습한다. Model-E는 현재 FM-only이며 efficiency loss는 아직 적용하지 않는다. 속도 억제 효과와 성공률 보존 여부는 동일 조건의 RoboLab rollout을 Model-FM과 비교해 확인한다.

이 문서는 환경 설치, 데이터 준비, LoRA 학습·재개와 Hugging Face 업로드를 다룬다. 추론·평가·모델 다운로드는 [readme_test.md](readme_test.md)를 따른다.

## 1. 환경과 프로젝트 설치

RunPod의 Ubuntu 22.04/24.04, Linux x86_64, NVIDIA CUDA GPU와 Bash를 기준으로 한다. 명령의 프로젝트 경로는 `/workspace/Project_Protoss`이며 실제 위치가 다르면 모든 블록의 `PROTOSS_ROOT`를 변경한다. OS 설치 명령은 root 기준이고 일반 계정은 `apt-get`, `ldconfig`에 `sudo`를 붙인다. 아래 설치·학습 명령은 원격 Linux용이다.

GR00T 소스는 `Issac-GR00T-N17/`, pretrained weight는 `../checkpoint/GR00T-N1.7-DROID/`, 데이터는 `Protoss/data/` 아래에 둔다. GPU 0을 기본으로 사용하며 모델별로 독립 실행한다. 괄호 안의 `set -e`는 오류 발생 시 해당 블록만 중단한다.
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

이미 프로젝트가 있으면 clone하지 않고 해당 경로를 사용한다. 서버에도 새 LoRA 학습·평가 코드가 포함된 revision을 준비한다.
```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
mkdir -p "$(dirname "$PROTOSS_ROOT")"
if [ ! -d "$PROTOSS_ROOT" ]; then
  git clone https://github.com/siri2100/Project_Protoss.git "$PROTOSS_ROOT"
fi
test -f "$PROTOSS_ROOT/Protoss/train.py"
test -f "$PROTOSS_ROOT/Protoss/prepare_droid.py"
)
```

프로젝트 저장소에서 원격 v1.6 브랜치를 기준으로 로컬 브랜치를 생성한다.
```bash
cd /workspace/Project_Protoss
git checkout -b v1.6 origin/v1.6
```

### 저장 폴더 준비

프로젝트 바로 상위 디렉터리에 `checkpoint`를 생성한다. 예를 들어 `/workspace/Project_Protoss`와 `/workspace/checkpoint`가 나란히 위치한다. 프로젝트 폴더 이름이 달라도 `CHECKPOINT_ROOT`는 상위 경로에서 계산한다. 새 Pod에서는 다음 명령으로 준비한다.
```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
mkdir -p "$PROTOSS_ROOT/Protoss/data/source" \
  "$PROTOSS_ROOT/Protoss/data/trainset" \
  "$CHECKPOINT_ROOT"
)
```

원본은 `Protoss/data/source`, 변환된 데이터는 `Protoss/data/trainset`, pretrained 및 학습 모델은 `../checkpoint`에 저장한다. 변환 명령의 output-dir과 학습·평가 명령의 dataset-root는 공통 부모 `Protoss/data`다. 보고서는 `styles.json`과 `inspection.json`으로 같은 폴더 안에 저장한다. 기존 다른 경로의 데이터/checkpoint는 자동 이동하지 않으며 필요하면 해당 CLI 경로를 지정해 재사용한다.

## 2. GR00T-N1.7 설치와 weight 다운로드

이미 설치·다운로드했다면 이 단계는 생략한다. 새 설치는 기존 v1.0에서 사용한 N1.7 소스 커밋을 사용한다.
Hugging Face 로그인 계정은 [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)에 접근할 수 있어야 한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
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
  --local-dir "$CHECKPOINT_ROOT/GR00T-N1.7-DROID"
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
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
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
  --local-dir "$CHECKPOINT_ROOT/GR00T-N1.7-DROID"
)
```

계속 실패하면 `whoami` 계정과 웹에서 승인받은 계정, 토큰의 읽기 권한을 다시 확인한다.
Cosmos 접근 확인을 생략해도 N1.7 모델 로딩에서 같은 권한 오류가 발생할 수 있다.
새 서버 터미널에도 오래된 `HF_TOKEN`이 설정되어 있다면 해제하거나 승인된 토큰으로 교체한다.

### 초기 설정: Hugging Face 업로드 권한 확인

2절의 `hf auth login`에서 모델 저장소 생성·쓰기 권한이 있는 토큰으로 로그인한다. Cosmos 접근도 같은 계정에서 승인되어 있어야 한다. 토큰을 코드나 Git에 기록하지 않는다. 학습·추론 터미널에서 동일한 `HF_HOME`을 사용한다. `HF_TOKEN` 환경변수가 있으면 저장된 로그인보다 우선한다.

## 3. 데이터 준비 의존성 확인
GR00T 전용 `.venv`에서 다운로드·변환·학습을 모두 실행한다. 데이터 준비용 의존성을 같은 환경에 설치한다. `--no-sync`로 실행해 추가 설치한 패키지를 유지한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv pip install --python .venv/bin/python \
  'huggingface-hub>=0.34,<2' 'pyarrow>=18,<24' 'pandas>=2.2,<3' 'scipy>=1.13,<2'
uv run --no-sync python - <<'CHECK'
import torch
import numpy, pandas, pyarrow, scipy, huggingface_hub
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print(torch.ones(1, device="cuda"), torch.cuda.get_device_name(0))
CHECK
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train.py" --help
)
```

Model-FM/S/E의 LoRA 대상은 action DiT 내부 `nn.Linear` weight다. LLM, vision encoder, projector, state/action encoder/decoder 및 pretrained weight를 고정하고 두 low-rank 행렬만 학습한다. CLI 기본 rank=64, alpha=128이며 아래 학습 예시는 사용자가 지정한 rank=32, alpha=64를 유지한다. Adapter dropout=0이다. 이는 이 프로젝트의 LoRA 적용 범위이며 NVIDIA 공식 LoRA 옵션이 아니다.

## 4. DROID 데이터 다운로드

1~3절 설치 후 진행한다. Fine-tuning에는 이미지·언어·정답 action이 필요하다. 아래 다운로드 명령은 GR00T 전용 Python 환경에서 실행한다.

`HF_HUB_ENABLE_HF_TRANSFER=0`으로 일반 다운로드를 사용한다. 서버에서 해당 변수가 `1`로 설정되어 있어도 아래 블록에서 덮어쓰므로 선택 패키지 `hf_transfer`를 설치할 필요가 없다. 실패하면 같은 명령을 다시 실행해 다운로드 cache를 재사용한다.

이미 받은 300개 `Protoss/data/source`를 그대로 사용한다. 다시 다운로드할 필요는 없다. 공통 데이터 준비는 동일 instruction 반복을 요구하지 않으며 최소 1개 원본 episode로 실행할 수 있다. 아래 다운로드 명령은 데이터가 아직 없을 때만 실행한다. 다운로드 helper는 첫 metadata shard의 성공·언어 있는 episode를 선택한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
export HF_HUB_ENABLE_HF_TRANSFER=0
uv run --no-sync python "$PROTOSS_ROOT/Protoss/prepare_droid.py" download \
  --dataset-dir "$PROTOSS_ROOT/Protoss/data/source" --num-episodes 300
)
```

## 5. 공통 원본 DROID 데이터 준비

수정된 코드를 서버에 반영한 뒤 아래 경로에 생성한다. 이미 변형한 이전 데이터는 원래 궤적으로 되돌릴 수 없으므로 원본 `Protoss/data/source`에서 다시 변환한다. 원본 다운로드는 다시 하지 않는다. 기존 다운로드 폴더가 `droid_styles_source`라면 아래 `--dataset-dir`만 그 경로로 변경한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/prepare_groot_dataset.py" \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-dir "$PROTOSS_ROOT/Protoss/data/source" \
  --output-dir "$PROTOSS_ROOT/Protoss/data" \
  --style-method original --seed 42
)
```

Train/test 분할 없이 원본 episode 전체를 변환한다. 300개면 300개 모두 공통 trainset에 포함한다. Model-FM/S/E는 같은 `trainset/` 폴더를 읽으므로 데이터와 통계가 동일하며 모델별 데이터를 중복 생성하지 않는다. 원래 joint/gripper command, 관측 state, frame 순서와 15Hz를 유지한다. 배속 변형, smoothing, 시간 보간은 적용하지 않는다.

GR00T의 DROID 변환과 동일한 `[eef_9d(9), gripper(1), joint(7)]` LeRobot v2 계약으로 저장한다. EEF 변환과 normalization/relative 통계는 설치된 GR00T 함수를 사용한다. 두 카메라는 원본 시작 시점에서 decode한 뒤 H.264로 재인코딩하므로 영상 압축은 바뀌지만 시간 축은 유지한다. 전체 GR00T pretraining 데이터셋을 재현하는 것이 아니라 다운로드한 공개 DROID subset의 형식을 GR00T 학습에 맞춘다.

결과: `Protoss/data/trainset/`, `Protoss/data/styles.json`. 성공 출력은 `Prepared train=300`이다. 보고서의 `style_method`는 `original`, `synthetic`은 `false`다. `testset/`은 생성하지 않는다. 학습과 open-loop 평가에 같은 데이터를 사용하므로 이 평가는 학습 데이터의 imitation 오차를 비교한다.

이미 생성된 trainset/testset 또는 styles.json이 있으면 중단한다. 기존 240/60 분할 데이터는 그대로 사용하지 않는다. 기존 `trainset/`, `testset/`, `styles.json`, `inspection.json`을 별도 백업 경로로 옮긴 뒤 5절 변환을 다시 실행한다. `source/`는 그대로 재사용하며 재다운로드하지 않는다. 학습·평가·데이터 업로드는 기존 분할 manifest를 거부한다. 빈 폴더와 .gitkeep는 허용한다. 진단만 필요하면 `--inspect-only`를 추가한다. 기존 변형 데이터와 checkpoint는 자동으로 삭제하지 않는다. 새로운 공통 데이터 실험은 새 checkpoint output 경로를 사용하고 기존 변형 데이터 학습을 `--resume`으로 이어붙이지 않는다.

## 6. Model-FM/S/E LoRA 학습

Model-FM/S는 `NAME`·`STEP`으로 run 폴더를 정하고, Model-E는 `mktemp -d`로 매번 고유한 빈 PEFT 실험 폴더를 생성한다. 생성 경로는 터미널과 `../checkpoint/latest_*_lora_v16.txt`에 기록한다. 데이터는 5절의 `Protoss/data`를 재사용한다.

`ValueError: Output is nonempty; choose a new path or --resume`는 지정한 output 폴더에 파일이 있어 덮어쓰기를 막았다는 뜻이다. 이전 실행이 학습 전에 중단되어 설정 JSON만 남아 있어도 발생한다. 같은 명령을 반복하면 같은 오류가 난다.

- 새 학습: 아래 명령은 기존 폴더를 보존하면서 새 폴더를 자동 생성한다. `latest_*_lora_v16.txt`는 가장 최근에 시작한 실험을 가리키며, 이전 실험은 실제 폴더 경로를 직접 지정한다.
- 학습 재개: 같은 PEFT 실험의 `checkpoint-*`에 모델·optimizer·scheduler·RNG 상태가 모두 있을 때만 기존 output 경로와 원래 학습 설정을 유지하고 `--resume`을 추가한다.
- 설정 JSON만 있거나 이전 자체 LoRA/full fine-tuning checkpoint만 있는 폴더는 `--resume`으로 해결되지 않는다. 새 output 경로에서 pretrained 모델부터 시작한다.

현재 학습은 **원본 pretrained processor, normalization 통계, 영상 시점을 그대로 유지**한다. `override_pretraining_statistics=False`로 학습 subset 통계 덮어쓰기를 막고, pretrained processor를 추가 override 없이 직접 로드한다. 원본 영상 시점이 `[-15, 0]`이면 학습에서도 그대로 사용한다. 저장되는 `processor/`의 JSON 파일은 원본 파일을 그대로 복사한다. Model config도 원본 checkpoint에서 읽고 기존 파라미터는 동결한 채 선택한 LoRA A/B만 학습한다.

이전 `[0]` 영상·subset normalization으로 학습한 checkpoint는 새 학습으로 `--resume`하지 않는다. 새 output에서 원본 pretrained 모델부터 다시 시작한다. 데이터는 기존 공통 `trainset/`을 재사용하며 재다운로드·변환은 필요 없다. `lora_settings.json`에 `processor_policy=pretrained-v1`, 원본 processor 파일 SHA256, 영상 시점, 통계 덮어쓰기 여부를 기록하고, 다르면 재개를 거부한다. `--state-dropout-prob` override 옵션은 제거했다.

학습 완료 후 run 전체를 Hugging Face private 모델 저장소에 **자동 업로드**한다. 기본 저장소는 `로그인계정/output폴더명`이다. 학습 시작 시 로그인을 확인하며, 저장소를 직접 지정하려면 학습 명령에 `--hf-repo-id ACCOUNT/REPO`를 추가한다. 오프라인 학습만 필요하면 `--no-hf-upload`를 지정한다. 모델·loss 저장과 분포 그래프 생성 후 업로드하며, 그래프 생성에 실패해도 저장된 모델의 업로드를 시도한다. 업로드 실패는 오류로 종료하며 로컬 모델은 유지된다. 이때 재학습하지 말고 8절의 수동 업로드 명령을 사용한다.

### 6.1 Model-FM 학습

```bash
(
set -e
NAME=v1614
STEP=500
LR=1e-6
FOLDER=Project_Protoss

export "PROTOSS_ROOT=/workspace/$FOLDER"
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
mkdir -p "$CHECKPOINT_ROOT"
OUTPUT_DIR="$CHECKPOINT_ROOT/${NAME}_model_fm_step${STEP}"
printf '%s\n' "$OUTPUT_DIR" > "$CHECKPOINT_ROOT/latest_FM_lora_v16.txt"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train.py" \
  --model FM --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-root "$PROTOSS_ROOT/Protoss/data" \
  --base-model "$CHECKPOINT_ROOT/GR00T-N1.7-DROID" \
  --output-dir "$OUTPUT_DIR" \
  --lora-rank 32 --lora-alpha 64 --max-steps "$STEP" --save-steps "$STEP" \
  --batch-size 2 --gradient-accumulation 16 --lr "$LR" --workers 2 --seed 42
)
```

### 6.2 Model-S 학습

실험 구현은 `Protoss/src/model_s_loss.py`와 `model_s_forward.py`에 모았다. `ModelSConfig`는 설정, `joint_affine()`는 관절 역정규화, `masked_velocity_loss()`는 속도 제곱 평균, `ModelSLoss`는 최종 조합을 담당한다.

- 목적 함수는 `L = fm_weight × loss_fm + lambda × normalized_loss_velocity`다. `loss_fm = masked_mean((v_LoRA - stopgrad(v_pretrained))²)`는 정답 action에 대한 FM이 아니라 pretrained flow velocity를 보존하는 distillation loss다. Model-FM/E의 기존 FM 학습은 유지한다.
- Teacher/student는 관측, state 특징, noisy action, 연속 timestep과 discrete timestep을 공유한다. Teacher는 같은 PEFT DiT의 adapter를 끈 상태에서 `no_grad`로 실행하고 student에는 adapter를 적용한다. 두 pass의 DiT dropout은 끄고 이후 training 상태를 복원한다. Backbone은 한 번, DiT는 두 번 실행하며 pretrained weight를 별도로 복제하지 않는다.
- DROID `[EEF(9), gripper(1), joints(7)]` 중 joint 7개를 pretrained 통계로 역정규화한다. `a_clean = x_t + (1-t)v_LoRA`로 clean endpoint를 추정한다. 관절 action은 chunk 공통 기준 위치에 대한 offset이므로 인접 차분에서 기준값이 상쇄된다. Gripper·EEF·padding은 속도 loss에서 제외한다.
- 관절 속도 `q_dot = (q[k+1]-q[k])/dt`로 `raw_velocity_loss = mean(q_dot²)`를 계산하고 `normalized_velocity_loss = raw_velocity_loss / v_ref²`로 무차원화한다. 두 점 모두 유효한 pair만 평균한다. 기본 `dt=1/15`초이며 raw 단위는 `rad²/s²`다.
- 학습 중 action 역정규화는 gradient 유지를 위해 clipping 없는 affine 변환을 사용한다. Clean endpoint는 실제 multi-step 추론 action의 추정값이므로, 실제 속도 감소와 성공률 유지는 RoboLab rollout으로 확인한다.
- 기본 첫 10개 target batch로 정답 관절 속도의 RMS를 추정해 `v_ref`로 고정한다. 유효 pair 수로 가중 평균하며 최저 기준은 `0.001 rad/s`다. 이 준비 단계는 모델 forward/backward, optimizer·scheduler 업데이트 없이 진행하고, 추정값은 이후 예측에 따라 변하지 않는다.
- `--s-fm-weight`로 pretrained flow 보존 항의 가중치를 조정한다(기본값 `1.0`). 아래 명령에서는 `L_FM=1000`을 예시로 사용하며 실험에 맞게 바꾼다. `--s-velocity-lambda`는 이와 독립적으로 속도 억제 강도를 조정한다.
- `--s-velocity-lambda 0.01`로 lambda를 조정한다. 0이면 teacher 보존 loss만 사용한다. `--s-velocity-reference 1.0`으로 기준 속도(rad/s)를 직접 지정하면 RMS 추정을 생략한다. `--s-calibration-batches 10`으로 측정 batch 수를 바꾼다. Jerk 관련 옵션은 사용하지 않는다.
- `--s-gradient-diagnostics`를 추가하면 첫 microbatch의 teacher 보존 loss와 정규화 속도 loss에 대한 LoRA gradient norm을 기록한다. 추가 autograd 비용이 있다.
- `training_loss.csv`와 checkpoint에 FM 보존 loss, raw/정규화/가중 속도 loss, lambda, 기준 속도를 기록한다. 그래프는 `fm_loss`, `normalized_velocity_loss`, `total_loss` 세 패널이다. 콘솔은 step마다 `fm_loss`, `weighted_fm_loss`, `velocity_loss`, `total_loss`를 한 줄에 출력한다. CSV에도 FM 가중치와 가중 FM loss를 기록한다. 기준 속도와 calibration 누적값은 `model_s_loss.json`에 저장하고 재개 시 복원한다.

아래 명령은 All QKV, rank/alpha=32/64, LR=1e-6으로 학습한다. Model-S 기본 lambda는 0.01이며 새 실험마다 NAME과 output을 구분한다.

이전 jerk-only, 일반 FM+velocity 또는 FM+jerk checkpoint는 새 flow distillation 목적 함수로 재개하지 않는다. Pretrained부터 새 run을 시작하고 새 모델 재개 시 원래 설정을 유지한다.

```bash
(
set -e
STEP=500
NAME=v176_model_s_step"${STEP}"
LR=1e-6
FOLDER=Project_Protoss
L_FM=1000
L_VELOCITY=1e-6

export PROTOSS_ROOT=/workspace/"$FOLDER"
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
mkdir -p "$CHECKPOINT_ROOT"
OUTPUT_DIR="$CHECKPOINT_ROOT/${NAME}"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train.py" \
  --model S --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-root "$PROTOSS_ROOT/Protoss/data" \
  --base-model "$CHECKPOINT_ROOT/GR00T-N1.7-DROID" \
  --output-dir "$OUTPUT_DIR" \
  --s-fm-weight "$L_FM" --s-velocity-lambda "$L_VELOCITY" --s-calibration-batches 10 \
  --lora-rank 32 --lora-alpha 64 --max-steps "$STEP" --save-steps 500 \
  --batch-size 2 --gradient-accumulation 16 --lr "$LR" --workers 1 --seed 42 \
)
```

### 6.3 Model-E 학습

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
mkdir -p "$CHECKPOINT_ROOT"
OUTPUT_DIR="$(mktemp -d "$CHECKPOINT_ROOT/model_E_lora_v16_peft_XXXXXXXX")"
printf '%s\n' "$OUTPUT_DIR" > "$CHECKPOINT_ROOT/latest_E_lora_v16.txt"
printf 'Training output: %s\n' "$OUTPUT_DIR"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train.py" \
  --model E --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-root "$PROTOSS_ROOT/Protoss/data" \
  --base-model "$CHECKPOINT_ROOT/GR00T-N1.7-DROID" \
  --output-dir "$OUTPUT_DIR" \
  --lora-rank 32 --lora-alpha 64 --max-steps 2000 --save-steps 2000 \
  --batch-size 2 --gradient-accumulation 16 --lr 1e-5 --workers 2 --seed 42
)
```

GR00T 기본 masked flow-matching MSE를 그대로 사용한다. FM의 velocity는 물리적 관절 속도가 아니다. Model-FM/S 비교는 LoRA rank/alpha, train 데이터, seed, batch, optimizer와 step 수를 맞추고 loss만 기존 FM과 pretrained flow 보존 + normalized velocity로 구분한다. Model-E 예시 설정은 별도 실험이다. 각 명령은 해당 모델 하나만 학습하고 종료한다. 원본 pretrained checkpoint에서 각각 시작하며 서로의 학습 weight를 이어받지 않는다. Model-S는 가중 pretrained flow 보존 loss와 정규화 관절 속도 loss를 사용한다.

### 6.4 중단된 PEFT 학습 재개

6.1의 PEFT 학습이 checkpoint 저장 이후 중단된 경우에만 사용한다. 아래 명령은 `latest_FM_lora_v16.txt`가 가리키는 실험을 재개하므로, 새 학습 명령을 다시 실행하기 전에 사용한다. 디버그 출력 후 종료되어 설정 파일만 남은 경우에는 6.1에서 새 run 경로를 사용한다. 다른 실험을 재개할 때는 해당 실험의 원래 output 경로와 모든 학습 옵션을 유지한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train.py" \
  --model FM --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-root "$PROTOSS_ROOT/Protoss/data" \
  --base-model "$CHECKPOINT_ROOT/GR00T-N1.7-DROID" \
  --output-dir "$(cat "$CHECKPOINT_ROOT/latest_FM_lora_v16.txt")" \
  --lora-rank 32 --lora-alpha 64 --max-steps 1000 --save-steps 100 \
  --batch-size 2 --gradient-accumulation 16 --lr 1e-5 --workers 2 --seed 42 \
  --resume
)
```

산출물:

- `checkpoint-*`: `adapter/`와 optimizer·scheduler·RNG·trainer 상태를 저장하는 재개용 checkpoint. pretrained 가중치는 중복 저장하지 않는다. 같은 학습 명령에 `--resume`을 추가해 재개한다. 기본 서버에 바로 전달하지 않는다.
- `adapter/`: PEFT 표준 `adapter_config.json`과 `adapter_model.safetensors`. 원본 DiT에 로드하는 adapter이며 standalone GR00T 모델은 아니다.
- `processor/`: 학습에서 사용한 processor와 normalization 통계.
- `inference/adapter_inference.json`: 원본 pretrained 모델, adapter, processor 경로를 연결하는 설정 파일. 학습 종료 시 병합 모델을 저장하지 않는다. 추론 모드는 `--lora-mode`로 선택한다. 기본 `unmerged`는 adapter를 forward마다 적용하며, `merged`는 FP32로 병합한 뒤 추론 dtype으로 변환한다.
- `lora_settings.json`, `lora_run.json`: LoRA backend(`peft`), 설정과 적용 layer 목록. 다른 backend/rank/alpha/base/data/seed로 resume하면 중단한다. 이전 자체 LoRA checkpoint는 PEFT 학습으로 resume할 수 없으며 새 output 경로를 사용한다. 기존 merged `inference/`는 계속 평가에 사용할 수 있다.

단일 CUDA GPU 경로를 제공한다. LoRA는 trainable parameter와 optimizer 메모리를 줄이지만 전체 모델 weight와 action-head activation은 필요하다. 해당 구성의 GPU peak와 학습 시간은 아직 측정하지 않았다.

### 6.5 학습 완료 후 자동 그래프 저장

`train.py`는 마지막 checkpoint와 adapter export, `training_loss.png` 저장을 마친 뒤 학습 모델·optimizer를 해제하고 pretrained와 최종 adapter를 순차 로드해 action 분포를 비교한다. 별도 명령 없이 `--output-dir`로 지정한 모델 저장 폴더에 **`distribution_plot.png`**가 생성된다.

학습 데이터의 trajectory 0에서 최대 32개 프레임을 균등 선택하고, 같은 관측과 샘플별 seed로 학습 전후 추론 action의 차원별 분포를 그린다. 원본 processor의 역정규화된 action 값이며, 전체 예측 chunk를 사용한다. 추론은 `unmerged`, denoising 4 steps로 실행한다. 이 단계에는 추가 추론 시간이 들며, 중단된 학습에서는 실행하지 않는다. 그래프 생성 중 오류가 나도 이미 저장된 checkpoint와 adapter는 유지된다.

분석 조건을 바꾸거나 저장된 모델로 그래프만 다시 생성하려면 [추론 문서 4절](readme_test.md#4-학습-전후-action-distribution-비교)의 명령을 사용한다.

## 7. 코드와 검증

| 파일 | 역할 |
|---|---|
| `Protoss/prepare_droid.py` | 공개 성공 DROID subset 다운로드 (`download` 명령) |
| `Protoss/prepare_groot_dataset.py` | 네 모델의 공통 원본 데이터 LeRobot v2 준비 |
| `Protoss/train.py` | Model-FM/S/E의 공통 FM LoRA 학습 |
| `Protoss/eval_groot_models.py` | baseline / Model-FM/S/E 공통 trainset·seed·horizon 평가 |
| `Protoss/src/groot_common.py` | 공통 데이터 변환과 학습 구현 |
| `Protoss/src/groot_lora.py` | action DiT Linear LoRA 주입과 merge |
| `Protoss/src/tests/test_groot_common.py` | 원본 command 보존, 전체 episode 보존와 학습 설정 검증 |

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python -m unittest discover \
  -s "$PROTOSS_ROOT/Protoss/src/tests" -t "$PROTOSS_ROOT/Protoss" -p 'test_*.py' -v
)
```

공통 데이터와 학습 설정 CPU 테스트 4개를 통과했다. 영상 encoder, GR00T 통계 함수와 GPU trainer는 테스트에서 mock을 사용했다. 실제 GPU 학습 및 네 모델의 성능 비교는 이 로컬 환경에서 검증하지 않았다.

검증: CPU mock 테스트로 dataset repo type/private 생성, 전체 trainset 전송 파일 목록, 불완전한 영상/episode 수 검사, dry-run을 확인했다. 실제 계정으로 데이터 업로드·다운로드는 이 작업에서 실행하지 않았다.

LoRA 검증 범위: CPU에서 초기 출력 일치, adapter만의 gradient, merge 전후 출력 일치와 native Linear state reload를 확인한다. 실제 GR00T LoRA GPU 학습/merged checkpoint 로딩은 이 작업 환경에서 실행하지 않았다. Model-S는 pretrained flow 보존 + normalized velocity를 사용하며 Model-E는 FM-only다.

이번 변경 검증: 데이터·학습 설정 4개, LoRA gradient/merge 1개, 네 모델의 공통 평가 조건 1개로 총 6개 CPU 테스트를 통과했다. 평가 조건 테스트는 subprocess를 mock하며 실제 VLA inference는 실행하지 않는다.

폴더 정리 후 현재 데이터·LoRA·평가·업로드 테스트 총 10개를 통과했다. 이전 blending/refiner, 스타일 데이터 변형과 A/B 전용 진입점은 삭제했다.

Model-S endpoint 복원, 3차 차분과 mask 처리, LoRA gradient 전달과 resume는 CPU 테스트로 검증한다. 실제 GPU 학습과 rollout 개선 여부는 별도로 확인한다.


## 8. Hugging Face에 checkpoint 보관

현재 PEFT 실험은 학습 완료 후 **run 폴더 전체**를 private 모델 저장소에 자동 업로드한다. 아래 수동 명령은 업로드 재시도나 별도 보관에 사용한다. 원본 pretrained 모델은 포함하지 않는다. `adapter/`, `processor/`, `inference/adapter_inference.json`과 실험 설정·loss 기록을 함께 보관한다. 업로드 중에는 해당 폴더를 수정하지 않는다.

### 8.1 인증

로그인은 초기 설정의 2절에서 완료한다. 자동 업로드는 저장된 토큰을 재사용한다.

### 8.2 모델별 adapter output 업로드

2절 초기 설정에서 로그인한 뒤 실행한다. 아래 Model-FM은 6.1절의 `v1614_model_fm_step500` 폴더를 사용한다. `FOLDER`, `NAME`, step은 실제 학습 설정과 동일하게 지정한다. S/E의 폴더와 step은 실제 학습 결과에 맞춘다. Organization은 `HF_ACCOUNT`를 해당 이름으로 지정한다.

**`--folder`에는 `/inference`를 붙이지 않는다.** PEFT output에는 전체 모델 `config.json`이 없으며, `adapter/adapter_config.json`과 adapter weight가 있다. `inference/`는 경로 설정뿐이므로 단독 업로드하지 않는다. 전체 run을 업로드하면 `processor/`, normalization 통계, `lora_settings.json`, `training_loss.png`·`training_loss.csv`, checkpoint의 loss 기록과 재개 상태가 함께 보관된다. 기존 loss log 복사 명령은 필요 없다.

```bash
(
set -e
FOLDER=Project_Protoss2
NAME=v1614
export PROTOSS_ROOT=/workspace/"$FOLDER"
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
HF_ACCOUNT=$(uv run --no-sync python -c \
  'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')
uv run --no-sync python "$PROTOSS_ROOT/Protoss/upload_checkpoint_hf.py" \
  --folder "$CHECKPOINT_ROOT/"$NAME"_model_fm_step500" \
  --repo-id "$HF_ACCOUNT/${NAME}_model_fm_step500" --workers 2
)
```

파일 검증만 하려면 업로드 명령에 `--dry-run`을 추가한다. 스크립트의 dry-run은 네트워크 요청을 하지 않지만, 위 계정명 조회는 네트워크를 사용한다. 실제 업로드는 private 저장소만 허용하며, 완료 후 원격 파일 목록을 확인한다. 중단 시 같은 폴더·저장소로 다시 실행하고 `.cache/huggingface`를 유지한다.

`committed: N/N`인데 원격 검증에서 파일이 누락되면, SDK의 이전 업로드 상태와 현재 저장소가 어긋났을 수 있다. 수정된 업로드 코드는 누락 파일만 `upload_file`로 정확한 원격 경로에 다시 전송하고 재검증한다. 같은 명령으로 재실행하면 되며, 전체 캐시 삭제나 기존 원격 파일 삭제는 필요 없다. 끝까지 누락되면 성공으로 처리하지 않고 저장소 URL과 누락 경로를 출력한다.

## 9. 공통 데이터를 Hugging Face에 보관

`Protoss/upload_dataset_hf.py`로 **`Protoss/data`의 변환된 학습·평가 데이터**를 private dataset 저장소에 업로드한다. trainset의 parquet, 영상, metadata/통계 및 `styles.json`이 모두 포함된다. 5절이 `Prepared train=300`으로 완료된 뒤 실행한다. 코드가 metadata의 episode 수와 필요한 파일의 존재/빈 파일 여부를 검사하며, 생성 중이거나 불완전한 폴더는 업로드 전에 중단한다. 영상 decode나 parquet 내용 검증은 하지 않는다.

원본 `Protoss/data/source`는 이 스크립트의 업로드 대상이 아니다. 보관한 생성 데이터만 있으면 5절을 다시 실행하지 않고 학습할 수 있지만, subset 설정을 바꿔 재생성하려면 원본도 별도로 보관해야 한다. 실행 파일과 `Protoss/src/`를 서버에 함께 반영한다.

### 9.1 데이터 업로드

8.1절처럼 dataset 저장소 생성·쓰기 권한이 있는 토큰으로 로그인한다. 아래 명령은 로그인된 실제 계정명을 자동으로 읽으므로 `YOUR_HF_ACCOUNT`를 입력할 필요가 없다. Organization에 올릴 때는 `HF_ACCOUNT`를 해당 organization 이름으로 직접 지정한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
HF_ACCOUNT=$(uv run --no-sync python -c \
  'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')
uv run --no-sync python "$PROTOSS_ROOT/Protoss/upload_dataset_hf.py" \
  --folder "$PROTOSS_ROOT/Protoss/data" \
  --repo-id "$HF_ACCOUNT/protoss-groot-v16-shared-data" --workers 2
)
```

네트워크 없이 파일 수와 용량을 확인하려면 업로드 명령에 `--dry-run`을 추가한다(계정명 자동 조회 명령은 별도로 네트워크를 사용한다). Private dataset 저장소를 자동 생성하며 기존 public 저장소에는 업로드하지 않는다. 대용량 업로드가 중단되면 같은 폴더/저장소로 재실행한다. `.cache/huggingface` 전송 상태는 유지한다. `.cache`, `.git`, `.DS_Store`는 보관 대상에서 제외한다. 완료 시 원격 파일 목록에 대상 파일이 있는지 검사한다. 다른 subset/seed의 데이터는 새 저장소 이름에 보관한다.

### 9.2 데이터 다운로드

보관한 데이터를 내려받아 재학습하려면 [추론·평가 문서 3절](readme_test.md#3-평가-데이터-다운로드)을 따른 뒤 이 문서 6절부터 진행한다.

## 10. 코드 폴더 구조

`Protoss/`에는 CLI 진입점, README, data/를 두고, 모델은 프로젝트 바로 상위의 `checkpoint/`에 둔다. 구현은 `Protoss/src/`, 테스트는 `Protoss/src/tests/`에 있다. Python package import를 사용하므로 실행 명령의 현재 작업 디렉터리에 의존하지 않는다. 서버에는 src 폴더도 함께 복사한다. 데이터는 `Protoss/data/trainset`, 모델은 `$CHECKPOINT_ROOT`를 사용한다.

```text
상위 디렉터리/
├── checkpoint/                 # 원본 pretrained 및 학습 run, HF 다운로드 모델
└── Project_Protoss/             # 프로젝트 루트

Project_Protoss/Protoss/
├── README.md
├── data/
│   ├── source/                  # 원본 다운로드
│   └── trainset/                # 전체 episode, 학습·open-loop 평가 공통
├── prepare_droid.py
├── prepare_groot_dataset.py
├── train.py
├── eval_groot_models.py
├── upload_checkpoint_hf.py
├── upload_dataset_hf.py
└── src/
    ├── __init__.py
    ├── groot_common.py
    ├── groot_lora.py
    ├── ... 각 CLI의 구현 모듈
    └── tests/
```

저장 경로 변경 검증: 전체 trainset 생성, 같은 폴더를 읽는 학습·평가, 데이터 업로드의 source 제외를 포함해 CPU 테스트 11개를 통과했다. 실제 다운로드·GPU 학습·원격 업로드는 실행하지 않았다.

v1.6: train/test 분할을 제거하고 원본 episode 전체를 trainset에 변환한다. Open-loop 평가는 같은 trainset을 사용한다. 기존 분할 manifest는 학습·평가·데이터 업로드에서 거부한다.
