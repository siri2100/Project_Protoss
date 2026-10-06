# Project Protoss v1.2 — GR00T A/B 학습·평가

이 문서의 명령을 순서대로 실행하면 **GR00T-N1.7-DROID 설치 → 성공 DROID 시연 다운로드 → 스타일별 데이터 분리 → 모델 A/B 독립 fine-tuning → 평가**를 진행할 수 있다.

- 모델 A: 느리고 부드러운 동작의 시연을 학습한다.
- 모델 B: 같은 task의 빠른 동작 시연을 학습한다.

두 모델은 같은 pretrained checkpoint에서 각각 시작한다. 스타일은 demonstration 선택으로 학습하며 loss는 GR00T 기본 flow-matching MSE를 사용한다.

## 1. 환경과 프로젝트 설치

RunPod의 Ubuntu 22.04/24.04, Linux x86_64, NVIDIA CUDA GPU와 Bash를 기준으로 한다. 명령의 프로젝트 경로는 `/workspace/Project_Protoss`이며 실제 위치가 다르면 모든 블록의 `PROTOSS_ROOT`를 변경한다. OS 설치 명령은 root 기준이고 일반 계정은 `apt-get`, `ldconfig`에 `sudo`를 붙인다. 아래 설치·학습 명령은 원격 Linux용이다.

GR00T 소스는 `Issac-GR00T-N17/`, pretrained weight는 `Protoss/checkpoints/GR00T-N1.7-DROID/`, 데이터는 `data/v1.2/` 아래에 둔다. GPU 0을 기본으로 사용하며 A/B를 순차 학습한다. 괄호 안의 `set -e`는 오류 발생 시 해당 블록만 중단한다.

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

이미 프로젝트가 있으면 clone하지 않고 해당 경로를 사용한다. 서버에도 A/B 학습 코드가 포함된 revision을 준비한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
mkdir -p "$(dirname "$PROTOSS_ROOT")"
if [ ! -d "$PROTOSS_ROOT" ]; then
  git clone https://github.com/siri2100/Project_Protoss.git "$PROTOSS_ROOT"
fi
test -f "$PROTOSS_ROOT/Protoss/train_groot_styles_v1_2.py"
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

## 3. 데이터 준비 의존성 확인

GR00T 전용 `.venv`에서 다운로드·변환·학습을 모두 실행한다. 데이터 준비용 의존성을 같은 환경에 설치한다. `--no-sync`로 실행해 추가 설치한 패키지를 유지한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
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
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train_groot_styles_v1_2.py" --help
)
```

`Protoss/train_groot_styles_v1_2.py`가 이번 학습의 실행 코드다. A/B 모두 **GR00T-N1.7-DROID**에서 시작해 multimodal projector와 diffusion action head를 학습한다. LLM/vision backbone은 고정한다.

| 모델 | 학습 demonstration | Loss |
|---|---|---|
| A (`a`) | 같은 task 내 관절 command 속도 RMS와 가속도 RMS가 낮은 episode | GR00T 기본 masked flow-matching MSE |
| B (`b`) | A와 겹치지 않는 빠른 episode | 같은 GR00T 기본 loss |

여기서는 **스타일별 시연을 imitation하는 방식**을 구현했다. A에 별도 smoothness penalty를 더하거나 B에 음의 smoothness penalty를 넣는 구현은 아니다. 가속도 RMS가 낮을수록 부드럽다는 기준을 사용하며 jerk도 보고서에 기록한다. B에 의도적으로 진동을 만들지는 않는다. 동일 task에서 A의 평균 속도와 평균 가속도가 모두 B보다 낮은 경우만 사용한다. 이는 데이터의 차이를 검증하며 학습된 모델의 실제 움직임 차이를 보장하지 않는다. 관절 속도가 높아도 task 완료 시간이 짧아지는지는 closed-loop에서 확인해야 한다.

## 4. DROID 데이터 다운로드

1~3절 설치 후 진행한다. Fine-tuning에는 이미지·언어·정답 action이 필요하다. 아래 다운로드 명령은 GR00T 전용 Python 환경에서 실행한다.

`HF_HUB_ENABLE_HF_TRANSFER=0`으로 일반 다운로드를 사용한다. 서버에서 해당 변수가 `1`로 설정되어 있어도 아래 블록에서 덮어쓰므로 선택 패키지 `hf_transfer`를 설치할 필요가 없다. 실패하면 같은 명령을 다시 실행해 다운로드 cache를 재사용한다.

이미 받은 `data/v1.2/droid`를 사용해도 된다. 스타일 준비에서 반복 task 부족 오류가 나면 아래처럼 **새 폴더**에 episode 수를 늘린다. 300은 첫 시도용 개수이며 task당 데이터 수를 보장하지 않는다. 다운로드 helper는 첫 metadata shard의 성공·언어 있는 episode만 선택한다. 해당 shard의 가용 수를 넘기면 오류가 나며 모든 DROID shard를 자동 탐색하지 않는다. 같은 task별로 최소 5개가 필요하고, 실험의 신뢰성을 위해 충분한 반복 시연을 모으는 것이 좋다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
export HF_HUB_ENABLE_HF_TRANSFER=0
uv run --no-sync python "$PROTOSS_ROOT/Protoss/prepare_droid_v1_2.py" download \
  --dataset-dir "$PROTOSS_ROOT/data/v1.2/droid_styles_source" --num-episodes 300
)
```

## 5. 스타일 분리와 GR00T LeRobot v2 변환

기존 데이터로 먼저 시도하려면 아래 `--dataset-dir`만 `data/v1.2/droid`로 변경한다. 기존 데이터를 덮어쓰지 않는다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train_groot_styles_v1_2.py" \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" prepare \
  --dataset-dir "$PROTOSS_ROOT/data/v1.2/droid_styles_source" \
  --output-dir "$PROTOSS_ROOT/data/v1.2/groot_styles" --seed 42
)
```

준비 절차:

1. 성공 subset의 episode별 absolute joint command로 속도, 가속도, jerk를 계산한다. 관측된 실제 joint motion과는 구분한다. `v = diff(q) × 15`, `acc = diff(v) × 15`, `jerk = diff(acc) × 15`이며 7개 관절·모든 timestep의 RMS다.
2. **instruction 문자열이 같은 task**끼리 모으고, 무작위 약 20%(최소 1개)를 공통 test로 먼저 분리한다. task 문자열이 같아도 물체 위치·경로 길이는 다를 수 있다.
3. 나머지에서 속도/가속도 순위 합이 낮은 절반을 A에, 남은 episode 중 빠른 순으로 같은 수를 B에 배정한다. 스타일 차이가 부족한 task는 제외한다. A/B의 task별 episode 수는 같으며 세 split 사이 episode 중복은 없다. Validation split은 이 초기 구현에서 별도로 만들지 않는다.
4. 선택된 episode를 영상과 함께 GR00T-flavored LeRobot v2로 변환한다. `[eef_9d(9), gripper(1), joint(7)]`의 17D state/action, task annotation, per-episode parquet/video 및 split별 normalization/relative statistics를 생성한다. EEF 변환과 통계는 설치된 GR00T 함수를 사용한다. 카메라는 exterior_1_left와 wrist_left다.

결과: `groot_styles/a`, `groot_styles/b`, `groot_styles/test`, `groot_styles/styles.json`. JSON에서 사용 episode, task, metric과 제외 이유를 확인한다. 영상은 FFmpeg로 정확한 시작 시점에서 decode 후 H.264로 저장하며 frame을 빠뜨리거나 action만 시간 축을 바꾸지 않는다. Episode 단위로 영상을 RAM에 읽으므로 긴 episode는 RAM 사용량이 커질 수 있다. 준비 중 실패하면 새 output 경로로 재실행한다. Task당 5개는 실행 최소치이며 fine-tuning 품질에 충분하다는 의미가 아니다.

## 6. A 학습 후 B 학습

GPU를 사용하는 모델 서버와 RoboLab을 종료하고 실행한다. NVIDIA의 [hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/hardware_recommendation.md)는 fine-tuning에 최소 40GB VRAM을 안내한다. 48GB급 GPU를 시작점으로 권장하며 여기의 batch 설정에 대한 GPU peak는 아직 측정하지 않았다. A/B를 순서대로 학습하면 한 번에 모델 하나만 GPU에 올라간다.

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
for STYLE in a b; do
  uv run --no-sync python "$PROTOSS_ROOT/Protoss/train_groot_styles_v1_2.py" \
    --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" train \
    --styles-dir "$PROTOSS_ROOT/data/v1.2/groot_styles" --style "$STYLE" \
    --base-model "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID" \
    --output-dir "$PROTOSS_ROOT/Protoss/checkpoints/groot_v1.2_$STYLE" \
    --max-steps 2000 --save-steps 500 --batch-size 2 \
    --gradient-accumulation 16 --lr 1e-5 --workers 2 --seed 42
done
)
```

각 모델은 같은 원본 checkpoint에서 독립적으로 시작한다. B는 A의 checkpoint를 이어받지 않는다. Single GPU만 지원하며 microbatch=2, gradient accumulation=16으로 optimizer step당 32 samples다. Loss는 GR00T `gr00t_n1d7.py`의 noisy trajectory→flow velocity masked MSE를 그대로 사용한다. Flow velocity는 로봇의 물리적인 joint velocity가 아니다. 학습률/step 수는 초기 실험용 설정이다.

각 output에 `checkpoint-500` … `checkpoint-2000`, 최종 모델, `style_run.json`(스타일/원본 weight/데이터 선택/학습 설정)이 저장된다. 가장 좋은 모델을 자동 선택하지 않으므로 공통 test 평가와 closed-loop 결과로 선택한다. 기존 output이 있으면 중단한다. 중단된 학습은 해당 STYLE만 실행하면서 `--resume`을 추가하면 최신 trainer checkpoint의 optimizer 상태부터 재개한다. 원본 weight/data/batch 등 설정은 같은 값으로 유지한다.

## 7. 같은 held-out episode에서 A/B 평가

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
for STYLE in a b; do
  uv run --no-sync python gr00t/eval/open_loop_eval.py \
    --dataset-path "$PROTOSS_ROOT/data/v1.2/groot_styles/test" \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --model-path "$PROTOSS_ROOT/Protoss/checkpoints/groot_v1.2_$STYLE/checkpoint-2000" \
    --traj-ids 0 --execution-horizon 8 --steps 400 \
    --modality-keys joint_position gripper_position \
    --save-plot-path "$PROTOSS_ROOT/results/v1.2/groot_$STYLE"
done
)
```

먼저 test episode 0으로 실행 경로를 확인한다. 이후 `--traj-ids 0 1 2 ...`에 실제 test episode index를 지정해 같은 목록으로 비교한다. 실행 step이 길면 `--steps`를 늘린다. 이 평가는 ground-truth와 예측의 MSE/MAE 및 plot이며 speed/smoothness나 실제 task 성공률 평가를 대체하지 않는다. 빠른 스타일의 모델이 느린 test demo와 차이나는 것 자체를 실패라고 판단하지 않는다.

실제 motion 비교는 아래 서버를 **STYLE=a, b로 각각** 실행하고 RoboLab을 서버 포트 5557에 직접 연결해 8절의 같은 task/control rate/horizon 조건으로 평가한다. 아래는 A 실행 예다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path "$PROTOSS_ROOT/Protoss/checkpoints/groot_v1.2_a/checkpoint-2000" \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
)
```

A 평가가 끝나면 서버를 종료하고 checkpoint 경로의 `a`를 `b`로 바꿔 같은 포트에서 실행한다. 8절의 RoboLab output 이름도 B용으로 변경한다.

검증 범위: 스타일 분리/command metric, 실제 parquet·metadata 변환, A/B의 원본 checkpoint 및 output 분리는 CPU 테스트 5개로 확인했다. 변환 테스트의 GR00T 통계 함수와 영상 encoder, 학습 테스트의 GPU/trainer는 mock을 사용했다. 실제 GR00T checkpoint GPU 로딩, backward, RunPod 학습, 학습 후 스타일 차이는 로컬 macOS에서 실행하지 않았다. 스타일별 성공 시연이 적거나 환경과 맞지 않으면 원하는 동작을 얻지 못할 수 있다.


## 8. RoboLab closed-loop 평가 — 선택 사항

실제 simulator task 성공률이 필요하면 진행한다. Isaac Sim 5.1 EULA를 확인한 뒤 설치한다. RTX rendering을 지원하는 GPU와 NVIDIA graphics/Vulkan 라이브러리가 노출된 Linux 컨테이너가 필요하다. A100/H100에서 학습이 가능해도 simulator 실행이 가능한 것은 아니다. [Isaac Sim 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

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

RoboLab은 설치 당시 main을 사용하므로 기록된 revision을 두 비교 실험에서 동일하게 유지한다. 모델 서버와 simulator를 함께 실행할 때의 VRAM은 학습과 별도로 확인한다. 먼저 `--num-envs 1`로 시작하고 모델과 simulator를 GPU/호스트별로 분리할 수 있다.

### A/B 평가

7절의 A 모델 서버가 포트 5557에서 실행 중인 상태로 다른 터미널에서 실행한다.

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
  --headless --device cuda:0 --remote-host 127.0.0.1 --remote-port 5557 \
  --task BananaOnPlateTask BananasInBinOneMoreTask BananasInCrateTask \
  --num-envs 1 --num-runs 10 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name groot_v12_a_run01
)
```

첫 확인은 `--num-runs 1`로 줄일 수 있다. A가 끝나면 모델 서버의 checkpoint를 B로 변경하고, 같은 평가 명령에서 `--output-folder-name groot_v12_b_run01`로 변경한다. 새 실험은 output 이름의 run 번호를 바꾼다. 결과는 `/workspace/RoboLab/output/` 아래에 저장된다. Task 목록, control rate, horizon, simulator revision을 동일하게 유지하고 여러 run으로 비교한다.

성공률, 성공 episode의 완료 시간, 실제 joint trajectory의 속도/가속도/jerk RMS를 함께 비교한다. 현재 코드의 motion metric은 학습 시연 command metric이며 RoboLab 실제 joint trajectory의 metric을 자동 수집하는 기능은 포함하지 않는다.

## 9. 코드와 검증

| 파일 | 역할 |
|---|---|
| `Protoss/prepare_droid_v1_2.py` | 공개 성공 DROID subset 다운로드 (`download` 명령) |
| `Protoss/train_groot_styles_v1_2.py` | 스타일 분리, LeRobot v2 변환, GR00T A/B 독립 학습 |
| `Protoss/test_train_groot_styles_v1_2.py` | command metric, split 누수, 변환 및 학습 설정 검증 |

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python -m unittest discover \
  -s "$PROTOSS_ROOT/Protoss" -p 'test_train_groot_styles_v1_2.py' -v
)
```

CPU 테스트 5개를 통과했다. 영상 encoder, GR00T 통계 함수와 GPU trainer는 테스트에서 mock을 사용했다. 실제 GPU 학습 및 학습 후 스타일 차이는 이 로컬 환경에서 검증하지 않았다.
