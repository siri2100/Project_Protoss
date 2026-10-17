# Project Protoss v1.6 — 추론·평가

이 문서는 pretrained 또는 학습된 PEFT adapter의 다운로드, open-loop 평가, RoboLab closed-loop 평가를 다룬다. 학습·재개·업로드는 [readme_train.md](readme_train.md)를 따른다.

## 1. 공통 환경 준비

새 서버에서는 [학습 문서 1~3절](readme_train.md#1-환경과-프로젝트-설치)에 따라 프로젝트·GR00T 환경, 원본 pretrained weight와 데이터 의존성을 준비한다. 이미 설치한 서버는 재설치하지 않는다. RoboLab만 평가한다면 학습 데이터를 다운로드하거나 학습할 필요가 없다.

`PROTOSS_ROOT=/workspace/Project_Protoss`를 기준으로 한다. 학습 output에는 원본 pretrained weight가 포함되지 않는다. 다운로드한 adapter와 원본 checkpoint를 함께 준비하고, 서버의 `--model-path`에는 run 루트를, `--base-model-path`에는 원본 checkpoint를 지정한다.

## 2. Hugging Face 모델 다운로드

### 2.1 로그인

[Hugging Face 토큰 설정](https://huggingface.co/settings/tokens)에서 모델 저장소를 생성하고 파일을 쓸 수 있는 토큰을 준비한다. 토큰은 로그인 프롬프트에 입력하며 코드나 Git에 저장하지 않는다. 아래 모든 명령은 같은 `HF_HOME`을 사용한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync hf auth login
uv run --no-sync hf auth whoami
)
```

기존 `HF_TOKEN` 환경변수가 있으면 저장된 로그인보다 우선하므로 올바른 write 권한의 토큰인지 확인한다.

### 2.2 새 Pod에서 모델별 다운로드

새 Pod에 환경과 원본 `GR00T-N1.7-DROID`를 준비하고 2.1절에서 로그인한다. Adapter 저장소에는 원본 pretrained 가중치가 포함되지 않는다. 다운로드 폴더는 업로드한 run 구조 전체를 유지한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
HF_ACCOUNT=$(uv run --no-sync python -c \
  'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')
uv run --no-sync hf download "$HF_ACCOUNT/protoss-v164-model-fm-step1000" \
  --local-dir "$PROTOSS_ROOT/Protoss/checkpoint/v164_model_fm_step1000"
)
```

서버 명령은 5절을 사용하되 `MODEL_PATH`를 다운로드한 run 루트(예: `/workspace/Project_Protoss/Protoss/checkpoint/v16_model_fm_step1000`)로 지정하고, `--base-model-path`에는 새 Pod의 원본 pretrained checkpoint 경로를 지정한다. 폴더 이름을 바꿔도 `adapter/`·`processor/` 상대 경로는 유지된다. 저장된 원래 서버의 pretrained 절대 경로는 `--base-model-path`로 대체한다.

재현하려면 `hf download`에 `--revision COMMIT_SHA`를 추가한다. 재개 학습에는 run의 `checkpoint-*`, optimizer/scheduler/RNG 상태, 원본 pretrained 모델과 동일 데이터가 필요하다. 이전 서버의 절대 경로가 기록된 학습 재개 설정은 추론의 경로 override와 별개다.

검증: 전체 모델 및 PEFT run 업로드의 private 저장소 조건, 필수 artifact 검사, 원격 파일 누락 검사, dry-run을 CPU mock 테스트로 확인했다. 실제 업로드는 사용자가 서버에서 실행한다.


## 3. RoboLab closed-loop 평가 — 선택 사항

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

### GR00T 클라이언트의 영상 이력 적용

v1.3 학습은 baseline과 동일하게 카메라마다 현재 프레임 `[0]` 한 장을 사용한다. 클라이언트는 서버의 modality config를 조회해 `[0]`이면 한 장, 기존 v1.2 모델의 `[-15, 0]`이면 과거·현재 두 장을 보낸다. 두 프레임 모델은 환경별 이력을 저장하고 reset 시 초기화한다. `--open-loop-horizon`은 action 실행 길이이며 영상 프레임 수와 별개다.

서버와 평가를 실행하기 전에 아래 명령을 한 번 실행한다. 기존 client.py를 백업한 뒤 프로젝트의 수정된 클라이언트를 적용한다. 이후 RoboLab 코드를 업데이트하면 다시 적용한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
CLIENT=/workspace/RoboLab/policies/gr00t/client.py
test -f "$CLIENT"
test -f "$PROTOSS_ROOT/Protoss/src/robolab_gr00t_client.py"
if [ ! -f "$CLIENT.before_protoss_history" ]; then
  cp "$CLIENT" "$CLIENT.before_protoss_history"
fi
cp "$PROTOSS_ROOT/Protoss/src/robolab_gr00t_client.py" "$CLIENT"
)
```

이 클라이언트는 영상 `[0]` 또는 `[-15, 0]`, state `[0]`인 DROID 설정을 지원한다. 다른 영상 시점은 오류로 중단한다.

### baseline / Model-FM/S/E 평가

다른 터미널에서 아래 Model-FM 서버를 먼저 실행한다. baseline은 `MODEL_PATH`를 원본 pretrained checkpoint 경로로 바꾸고 `--base-model-path`를 제거한다. Model-FM은 `latest_FM_lora_v16.txt`에 기록된 학습 output 경로를 사용한다. S/E도 각 학습 output 경로를 사용한다. 서버는 원본 pretrained 모델과 저장된 adapter를 로드한다. `unmerged`와 `merged` 모두 LoRA가 적용되며, 원본 비교에는 `pretrained` 모드를 사용한다. `Protoss/serve.py`가 별도 GR00T checkout에 `Protoss/patches/gr00t_adapter_inference.patch`를 자동 적용하므로 `Protoss/patches/`도 함께 복사한다. baseline 서버 실행 시에는 `--base-model-path`를 빼고 `--model-path`에 원본 checkpoint를 지정한다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
MODEL_PATH="$PROTOSS_ROOT/Protoss/checkpoint/v164_model_fm_step1000/inference"
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/serve.py" \
  --model-path "$MODEL_PATH" \
  --base-model-path "$PROTOSS_ROOT/Protoss/checkpoint/GR00T-N1.7-DROID" \
  --lora-mode unmerged \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
)
```

디버깅할 때 위 명령의 `--lora-mode`만 바꾸고 서버를 재시작한다. `model.action_head.model = dit`는 **LoRA가 켜진 상태**다. 병합 줄을 주석 처리해도 pretrained-only가 되지 않는다.

| `--lora-mode` | 가중치 | Processor |
|---|---|---|
| `pretrained` | 원본 pretrained, adapter 미로드 | 원본 pretrained processor |
| `disabled` | 원본 pretrained, adapter 미로드 | 학습 processor |
| `unmerged` (기본) | 원본 + LoRA, forward마다 적용 | 학습 processor |
| `merged` | 원본 + LoRA를 FP32로 병합 | 학습 processor |

먼저 같은 RoboLab 조건에서 `pretrained`가 원본 직접 실행과 같은지 확인한다. 이어서 `disabled`가 달라지면 processor 변경 영향을, `disabled`와 `unmerged`의 차이로 학습된 LoRA의 영향을 확인한다. `unmerged`와 `merged`는 같은 모델의 두 실행 방식이다. 실제 수치 비교에서는 동일 관측과 동일 난수 seed를 사용해야 한다. 서버 로그에 mode, pretrained 경로, processor 경로, LoRA 활성·병합 여부를 출력한다.

GR00T 기본 서버를 원본(baseline) 또는 adapter output(Model-FM/S/E)로 포트 5557에서 실행한 상태로 다른 터미널에서 실행한다.

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
  --task BananaOnPlateTask \
  --num-envs 2 --num-runs 5 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name v1.6.4_mdoel_fm_step1000
)
```

첫 확인은 `--num-runs 1`로 줄일 수 있다. 각 모델을 GR00T 기본 서버의 `--model-path`에 지정하고 포트 5557에 연결해 평가한다. baseline은 원본 checkpoint, Model-FM/S/E는 adapter와 processor가 있는 학습 output 경로를 사용한다. Model-S/E도 학습이 끝난 뒤 평가한다. 결과 폴더 이름을 `baseline_run01`, `modelFM_run01`, `modelS_run01`, `modelE_run01`처럼 각각 다르게 지정한다. Task 목록, control rate, horizon, simulator revision을 같게 유지하고 여러 run으로 비교한다.

성공률, 성공 episode의 완료 시간, 실제 joint trajectory의 속도/가속도/jerk RMS를 함께 비교한다. 현재 코드의 motion metric은 학습 시연 command metric이며 RoboLab 실제 joint trajectory의 metric을 자동 수집하는 기능은 포함하지 않는다.


## 4. 학습 전후 action distribution 비교

학습 완료 시 `train.py`가 모델 저장 폴더에 `distribution_plot.png`를 자동 생성한다. 아래 명령은 기존 모델에서 그래프만 다시 생성하거나 분석 조건을 바꿀 때 사용한다.

동일한 데이터 관측과 난수 seed로 원본 pretrained와 학습된 LoRA의 **추론 action 출력 분포**를 비교한다. 가중치 분포나 학습 loss가 아니라, processor가 역정규화한 action 값이다. 각 action 차원을 별도 패널로 표시하며 두 모델에 같은 histogram 구간을 사용한다. 선택한 관측의 전체 예측 action chunk를 모은 주변 분포이므로 이 그래프만으로 task 성공률을 판단하지 않는다.

```bash
(
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
RUN_PATH="$PROTOSS_ROOT/Protoss/checkpoint/v164_model_fm_step1000"
# 서버 실행 때와 동일하게 adapter manifest를 찾을 수 있는 경로를 지정한다.
MODEL_PATH="$RUN_PATH/inference"
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/plot_distribution.py" \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --base-model "$PROTOSS_ROOT/Protoss/checkpoint/GR00T-N1.7-DROID" \
  --model-path "$MODEL_PATH" \
  --dataset-root "$PROTOSS_ROOT/Protoss/data" \
  --output-dir "$RUN_PATH" \
  --split train --traj-ids 0 --frames-per-trajectory 32 \
  --lora-mode unmerged --denoising-steps 4 --seed 42 --device cuda:0
)
```

출력: `Protoss/checkpoint/v164_model_fm_step1000/distribution_plot.png`. 다시 실행하면 같은 이미지 파일을 덮어쓴다. `RUN_PATH`는 실제 모델 저장 폴더로 바꾼다. `MODEL_PATH`는 서버에서 사용한 경로를 지정하며, run 루트에 manifest가 있으면 `$RUN_PATH`를 사용한다. 두 모델은 GPU에 순차 로드한다.

데이터가 없는 추론 전용 Pod라면 먼저 학습 문서의 데이터를 준비한다. 기본 예시는 `trainset`의 trajectory 0에서 최대 32개 프레임을 균등 선택한다. 더 넓게 비교하려면 `--traj-ids 0 1 2`처럼 늘린다. 별도 testset이 있는 경우 `--split test`를 사용한다. 두 모델에는 동일한 영상 시점·state·instruction이 전달되고, 샘플별 seed도 동일하게 설정된다. 실행에는 GR00T 환경의 `matplotlib`이 필요하다.
