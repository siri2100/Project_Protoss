# Project Protoss v1.6 — 추론·평가

이 문서는 pretrained 또는 학습된 PEFT adapter의 다운로드, open-loop 평가, RoboLab closed-loop 평가를 다룬다. 학습·재개·업로드는 [readme_train.md](readme_train.md)를 따른다.

## 1. 공통 환경 준비

새 서버에서는 [학습 문서 1~3절](readme_train.md#1-환경과-프로젝트-설치)에 따라 프로젝트·GR00T 환경, 원본 pretrained weight와 데이터 의존성을 준비한다. 이미 설치한 서버는 재설치하지 않는다. RoboLab만 평가한다면 학습 데이터를 다운로드하거나 학습할 필요가 없다.

`PROTOSS_ROOT=/workspace/Project_Protoss`를 기준으로 한다. 학습 output에는 원본 pretrained weight가 포함되지 않는다. 다운로드한 adapter와 원본 checkpoint를 함께 준비하고, 서버의 `--model-path`에는 run 루트를, `--base-model-path`에는 원본 checkpoint를 지정한다.

## 2. RoboLab
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
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
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

### Model 평가

다른 터미널에서 아래 Model-FM 서버를 먼저 실행한다. baseline은 `MODEL_PATH`를 원본 pretrained checkpoint 경로로 바꾸고 `--base-model-path`를 제거한다. Model-FM은 2.2절에서 Hugging Face로부터 다운로드한 run 폴더를 사용한다. S/E도 업로드한 해당 저장소에서 전체 run을 다운로드한 뒤 그 경로를 사용한다. 서버는 원본 pretrained 모델과 저장된 adapter를 로드한다. `unmerged`와 `merged` 모두 LoRA가 적용되며, 원본 비교에는 `pretrained` 모드를 사용한다. `Protoss/serve.py`가 별도 GR00T checkout에 `Protoss/patches/gr00t_adapter_inference.patch`를 자동 적용하므로 `Protoss/patches/`도 함께 복사한다. baseline 서버 실행 시에는 `--base-model-path`를 빼고 `--model-path`에 원본 checkpoint를 지정한다.

```bash
(
set -e
NAME=v179_model_s_step500
CHECKPOINT_STEP=500
export PROTOSS_ROOT=/workspace/Project_Protoss
export CHECKPOINT_ROOT="$(dirname "$PROTOSS_ROOT")/checkpoint"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
MODEL_PATH="$CHECKPOINT_ROOT/${NAME}"
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
HF_ACCOUNT=$(uv run --no-sync python -c \
  'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')
uv run --no-sync python "$PROTOSS_ROOT/Protoss/serve.py" \
  --hf-repo-id "$HF_ACCOUNT/${NAME}" \
  --model-path "$MODEL_PATH" \
  --checkpoint-step "$CHECKPOINT_STEP" \
  --base-model-path "$CHECKPOINT_ROOT/GR00T-N1.7-DROID" \
  --lora-mode unmerged \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
)
```

서버는 `--hf-repo-id`의 run과 `nvidia/GR00T-N1.7-DROID`를 지정한 로컬 경로에 다운로드한 뒤 실행한다. 재실행 시 다운로드 캐시를 재사용한다. 다른 소유자가 업로드했다면 `HF_ACCOUNT`를 해당 계정·organization으로 지정한다. `--hf-revision COMMIT_SHA`와 `--hf-base-revision COMMIT_SHA`로 revision을 고정할 수 있다. 로컬 모델만 사용하려면 `--hf-repo-id`를 생략한다. baseline 실행 시에도 이 옵션을 생략하고 초기 설정에서 다운로드한 원본을 사용한다.

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
NAME=v1.7.9_model_s_step500
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV CUDA_VISIBLE_DEVICES
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
cd /workspace/RoboLab
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --no-sync --extra isaac51 \
  python policies/gr00t/run.py \
  --headless --device cuda:0 --remote-host 127.0.0.1 --remote-port 5557 \
  --task \
    BananaOnPlateTask \
  --num-envs 2 --num-runs 5 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name "$NAME"
)
```
# BananaInBowlTask \
# BananaOnPlateTask \
# BananasInBinOneMoreTask

첫 확인은 `--num-runs 1`로 줄일 수 있다. 각 모델을 GR00T 기본 서버의 `--model-path`에 지정하고 포트 5557에 연결해 평가한다. baseline은 원본 checkpoint, Model-FM/S/E는 adapter와 processor가 있는 학습 output 경로를 사용한다. Model-S/E도 학습이 끝난 뒤 평가한다. 결과 폴더 이름을 `baseline_run01`, `modelFM_run01`, `modelS_run01`, `modelE_run01`처럼 각각 다르게 지정한다. Task 목록, control rate, horizon, simulator revision을 같게 유지하고 여러 run으로 비교한다.

성공률, 성공 episode의 완료 시간, 실제 joint trajectory의 속도/가속도/jerk RMS를 함께 비교한다. 현재 코드의 motion metric은 학습 시연 command metric이며 RoboLab 실제 joint trajectory의 metric을 자동 수집하는 기능은 포함하지 않는다.


`--checkpoint-step 500`은 다운로드한 run의 `checkpoint-500/adapter`를 사용한다. 최종 adapter를 쓰려면 이 옵션을 생략한다. 지정 checkpoint가 없으면 오류로 종료한다. 저장 제한(현재 최대 3개)으로 이미 삭제된 step은 HF에도 없을 수 있으며 해당 adapter 파일이 별도로 남아 있어야 추론할 수 있다.
