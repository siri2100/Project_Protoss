# Project Protoss v1.2 — GR00T A/B 학습·평가

이 문서의 명령을 순서대로 실행하면 **GR00T-N1.7-DROID 설치 → 성공 DROID 시연 다운로드 → 스타일별 데이터 분리 → 모델 A/B 독립 fine-tuning → 평가**를 진행할 수 있다.

- 모델 A: 느리고 부드러운 동작의 시연을 학습한다.
- 모델 B: 같은 task의 빠른 동작 시연을 학습한다.

두 모델은 같은 pretrained checkpoint에서 각각 시작한다. 스타일은 동일 시연의 시간 재샘플링과 A의 action smoothing으로 만들며 loss는 GR00T 기본 flow-matching MSE를 사용한다.

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
| A (`a`) | 원본 train 시연의 0.75배속 + 연속 action smoothing | GR00T 기본 masked flow-matching MSE |
| B (`b`) | 같은 원본 train 시연의 1.25배속 | 같은 GR00T 기본 loss |

기본 경로는 **합성 스타일 demonstration을 imitation하는 방식**이다. A/B 모두 GR00T 기본 loss를 사용하며 별도 smoothness penalty는 추가하지 않는다. 실제 동작의 속도·smoothness·task 성공률은 함께 평가해야 한다.

## 4. DROID 데이터 다운로드

1~3절 설치 후 진행한다. Fine-tuning에는 이미지·언어·정답 action이 필요하다. 아래 다운로드 명령은 GR00T 전용 Python 환경에서 실행한다.

`HF_HUB_ENABLE_HF_TRANSFER=0`으로 일반 다운로드를 사용한다. 서버에서 해당 변수가 `1`로 설정되어 있어도 아래 블록에서 덮어쓰므로 선택 패키지 `hf_transfer`를 설치할 필요가 없다. 실패하면 같은 명령을 다시 실행해 다운로드 cache를 재사용한다.

이미 받은 300개 `data/v1.2/droid_styles_source`를 그대로 사용한다. 다시 다운로드할 필요는 없다. 시간 재샘플링은 동일 instruction 반복을 요구하지 않으며 최소 3개 원본 episode로 실행할 수 있다. 아래 다운로드 명령은 데이터가 아직 없을 때만 실행한다. 다운로드 helper는 첫 metadata shard의 성공·언어 있는 episode를 선택한다.

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

## 5. 시간 재샘플링으로 A/B 데이터 생성

서버에 수정된 `Protoss/train_groot_styles_v1_2.py`를 반영한 뒤 실행한다. 실패했던 준비 시도와 구분해 새 폴더에 생성한다.

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
  --output-dir "$PROTOSS_ROOT/data/v1.2/groot_styles_retimed" \
  --style-method retime --slow-factor 0.75 --fast-factor 1.25 \
  --smooth-window 5 --seed 42
)
```

준비 절차:

1. 원본 episode를 먼저 train 80% / test 20%로 나눈다. 300개면 240/60개다. A/B는 같은 train 원본을 사용한다. Test 원본은 어느 학습 데이터에도 들어가지 않고 원래 속도로 보존한다. Episode 단위 split이며 unseen-task 평가나 pretrained DROID 전체의 데이터 누수 방지를 보장하지 않는다.
2. A는 0.75배속으로 시간 축을 확장한다. Joint/EEF 위치 action에 5프레임 중앙 이동 평균과 양 끝 위치를 보존하는 선형 보정을 적용한다. EEF 회전 action은 주변 rotation의 평균으로 smoothing하고 양 끝 회전을 보존한다. State에는 smoothing을 적용하지 않는다.
3. B는 같은 원본을 1.25배속으로 시간 압축하며 smoothing을 적용하지 않는다. 양쪽 모두 출력은 15Hz다. 시작/끝을 모두 포함하므로 짧은 episode의 실제 배속은 요청값과 약간 다를 수 있다.
4. 연속 joint/EEF 위치 state와 action은 선형 보간한다. 회전은 GR00T의 `XYZ` Euler convention으로 rotation을 만들고 Slerp로 보간한다. Gripper와 두 camera는 공통 시간 지도의 가장 가까운 원본 frame을 선택하며 gripper를 smoothing하지 않는다. 압축 시 아주 짧은 이벤트가 사라질 수 있으므로 큰 fast-factor 사용에 주의한다.
5. Timestamp, frame/global index, episode length를 새 15Hz 기준으로 작성한다. `[eef_9d(9), gripper(1), joint(7)]`의 GR00T LeRobot v2 데이터와 split별 normalization/relative statistics를 생성한다. EEF 변환과 통계는 GR00T 함수를 사용한다. 카메라는 exterior_1_left와 wrist_left다.

결과는 `groot_styles_retimed/a`, `b`, `test`, `styles.json`이다. 원본 300개 기준 성공 출력은 `Prepared A=240, B=240, test=60`이다. `styles.json`에는 `synthetic: true`, 원본 ID, 배속·smoothing 설정, 원본 및 변형 후 command 속도/가속도/jerk RMS를 기록한다. 계산은 `v = diff(q) × 15`, `acc = diff(v) × 15`, `jerk = diff(acc) × 15`이며 실제 로봇 motion과 구분한다. Smoothing 효과는 metric으로 확인한다.

영상은 원본 frame을 반복/선택하며 실제 새 장면을 생성하지 않는다. 보간된 state와 smoothed action 때문에 이미지·state·command의 물리적 일치가 근사적이고, joint/EEF command의 kinematic 일치도 보장하지 않는다. 접촉/파지 성공 여부와 빠른 궤적의 실행 가능성은 closed-loop에서 확인해야 한다. 초기 합성 데이터 실험으로 사용하고 이후 실제 스타일 시연으로 보완한다.

Episode 영상 전체를 RAM에 로드하므로 긴 episode는 메모리 사용량이 커질 수 있다. Output이 비어 있지 않으면 중단한다. 진단만 하려면 `--inspect-only`를 추가한다. `groot_styles_retimed.inspection.json`은 output 밖에 저장된다. 이전 방식은 `--style-method select`로 사용할 수 있으며 동일 instruction 최소 5개와 속도/가속도 차이 조건을 적용한다.

## 6. A/B 모델별 학습

GPU를 사용하는 모델 서버와 RoboLab을 종료하고 실행한다. NVIDIA의 [hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/hardware_recommendation.md)는 fine-tuning에 최소 40GB VRAM을 안내한다. 48GB급 GPU를 시작점으로 권장하며 여기의 batch 설정에 대한 GPU peak는 아직 측정하지 않았다. A/B를 순서대로 학습하면 한 번에 모델 하나만 GPU에 올라간다.

### 6.1 모델 A 학습

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
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train_groot_a_v1_2.py" \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --styles-dir "$PROTOSS_ROOT/data/v1.2/groot_styles_retimed" \
  --base-model "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID" \
  --output-dir "$PROTOSS_ROOT/Protoss/checkpoints/groot_v1.2_a" \
  --max-steps 2000 --save-steps 500 --batch-size 2 \
  --gradient-accumulation 16 --lr 1e-5 --workers 2 --seed 42
)
```

### 6.2 모델 B 학습

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
uv run --no-sync python "$PROTOSS_ROOT/Protoss/train_groot_b_v1_2.py" \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --styles-dir "$PROTOSS_ROOT/data/v1.2/groot_styles_retimed" \
  --base-model "$PROTOSS_ROOT/Protoss/checkpoints/GR00T-N1.7-DROID" \
  --output-dir "$PROTOSS_ROOT/Protoss/checkpoints/groot_v1.2_b" \
  --max-steps 2000 --save-steps 500 --batch-size 2 \
  --gradient-accumulation 16 --lr 1e-5 --workers 2 --seed 42
)
```

두 명령은 각각 해당 모델 하나만 학습하고 종료한다. A만 먼저 학습하고 B는 나중에 실행할 수 있다. 데이터 생성은 다시 하지 않는다.

각 모델은 같은 원본 checkpoint에서 독립적으로 시작한다. B는 A의 checkpoint를 이어받지 않는다. Single GPU만 지원하며 microbatch=2, gradient accumulation=16으로 optimizer step당 32 samples다. Loss는 GR00T `gr00t_n1d7.py`의 noisy trajectory→flow velocity masked MSE를 그대로 사용한다. Flow velocity는 로봇의 물리적인 joint velocity가 아니다. 학습률/step 수는 초기 실험용 설정이다.

각 output에 `checkpoint-500` … `checkpoint-2000`, 최종 모델, `style_run.json`(스타일/원본 weight/데이터 선택/학습 설정)이 저장된다. 가장 좋은 모델을 자동 선택하지 않으므로 공통 test 평가와 closed-loop 결과로 선택한다. 기존 output이 있으면 중단한다. 중단된 학습은 해당 모델의 명령에 `--resume`을 추가하면 최신 trainer checkpoint의 optimizer 상태부터 재개한다. 원본 weight/data/batch 등 설정은 같은 값으로 유지한다.

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
    --dataset-path "$PROTOSS_ROOT/data/v1.2/groot_styles_retimed/test" \
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

검증 범위: 동일 instruction 반복 없이 300개→240/60 split, 시간 보간의 배속·회전 wrap·gripper 정렬, smoothing, 실제 parquet·metadata·영상 frame 수 및 독립 학습 설정을 CPU 테스트 7개로 확인했다. 변환 테스트의 GR00T 통계 함수와 영상 encoder, 학습 테스트의 GPU/trainer는 mock을 사용했다. 실제 GR00T checkpoint GPU 로딩, backward, RunPod 학습, 학습 후 스타일 차이는 로컬 macOS에서 실행하지 않았다. 합성 스타일이 실제 task 수행에서 유효한지는 평가해야 한다.


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
| `Protoss/train_groot_styles_v1_2.py` | 시간 재샘플링/smoothing, LeRobot v2 변환, GR00T A/B 독립 학습 |
| `Protoss/train_groot_a_v1_2.py` | A 모델 전용 학습 진입점 |
| `Protoss/train_groot_b_v1_2.py` | B 모델 전용 학습 진입점 |
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

CPU 테스트 7개를 통과했다. 영상 encoder, GR00T 통계 함수와 GPU trainer는 테스트에서 mock을 사용했다. 실제 GPU 학습 및 학습 후 스타일 차이는 이 로컬 환경에서 검증하지 않았다.
