# Project Protoss v1.6 — 추론·평가

이 문서는 pretrained 또는 학습된 PEFT adapter의 다운로드, open-loop 평가, RoboLab closed-loop 평가를 다룬다. 학습·재개·업로드는 [readme_train.md](readme_train.md)를 따른다.

`train.py`는 이제 학습 후 추론 서버 시작·RoboLab 평가·점수 계산·HF 업로드를 자동 실행한다. [학습 문서 6.6절](readme_train.md#66-학습--robolab-테스트--평가점수-자동-계산)의 `--eval-*` 옵션으로 조건을 지정하고, 저장된 모델은 `evaluate_robolab.py`로 평가만 다시 실행한다. 아래 Server/Client 명령은 수동으로 나누어 실행할 때 사용한다.

## 1. 공통 환경 준비

새 서버에서는 [학습 문서 1~3절](readme_train.md#1-환경과-프로젝트-설치)에 따라 프로젝트·GR00T 환경, 원본 pretrained weight와 데이터 의존성을 준비한다. 이미 설치한 서버는 재설치하지 않는다. RoboLab만 평가한다면 학습 데이터를 다운로드하거나 학습할 필요가 없다.

`PROTOSS_ROOT=/workspace/Project_Protoss`를 기준으로 한다. 학습 output에는 원본 pretrained weight가 포함되지 않는다. 다운로드한 adapter와 원본 checkpoint를 함께 준비하고, 서버의 `--model-path`에는 run 루트를, `--base-model-path`에는 원본 checkpoint를 지정한다.

## 2. RoboLab
실제 simulator task 성공률이 필요하면 진행한다. 사용할 Isaac Sim 버전의 EULA를 확인한 뒤 설치한다. RTX rendering을 지원하는 GPU와 NVIDIA graphics/Vulkan 라이브러리가 노출된 Linux 컨테이너가 필요하다. A100/H100에서 학습이 가능해도 simulator 실행이 가능한 것은 아니다. [Isaac Sim 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

### RoboLab 설치

**드라이버 595에서는 [595용 설치 문서](README_robolab_595.md)를 따른다.** 아래 두 명령으로 별도 Isaac Sim 6.0.1 / Python 3.12 환경을 설치하고, `SMOKE PASS`가 나오는지 먼저 확인한다. 실제 GPU 호환성은 이 검사로 확인해야 한다.

```bash
bash /workspace/Project_Protoss/Protoss/setup_robolab_595.sh setup
bash /workspace/Project_Protoss/Protoss/setup_robolab_595.sh smoke
```

아래 설치 블록은 기존 Isaac Sim 5.1 환경을 유지할 때만 사용한다. 595용 설치를 한 뒤 이 블록으로 평가 환경을 바꾸지 않는다.

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

595용 setup은 새 checkout에 클라이언트를 자동 적용한다. 기존 5.1 환경은 서버와 평가를 실행하기 전에 아래 명령을 한 번 실행한다. 기존 client.py를 백업한 뒤 프로젝트의 수정된 클라이언트를 적용한다. 이후 RoboLab 코드를 업데이트하면 다시 적용한다.

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
NAME=v179_model_s_step500
MODEL_NAME=v179_model_s_step500
EXP_NUM=05
ROBOLAB_STACK=595
export PROTOSS_ROOT=/workspace/Project_Protoss
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV CUDA_VISIBLE_DEVICES
export OMNI_KIT_ACCEPT_EULA=Y
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
GR00T_PYTHON="$PROTOSS_ROOT/Issac-GR00T-N17/.venv/bin/python"
HF_ACCOUNT=$("$GR00T_PYTHON" -c \
  'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')
HF_REPO_ID="${HF_REPO_ID:-$HF_ACCOUNT/$MODEL_NAME}"
case "$ROBOLAB_STACK" in
  595)
    ROBOLAB_ROOT="${ROBOLAB_595_ROOT:-/workspace/robolab-595}/RoboLab"
    EVAL_COMMAND=(bash "$PROTOSS_ROOT/Protoss/setup_robolab_595.sh" eval)
    ;;
  51)
    ROBOLAB_ROOT=/workspace/RoboLab
    EVAL_COMMAND=("$ROBOLAB_ROOT/.venv-51/bin/python" policies/gr00t/run.py)
    ;;
  *) echo "ROBOLAB_STACK must be 595 or 51" >&2; exit 1 ;;
esac
cd "$ROBOLAB_ROOT"
"${EVAL_COMMAND[@]}" \
  --headless --device cuda:0 --remote-host 127.0.0.1 --remote-port 5557 \
  --task \
    BananaOnPlateTask \
  --num-envs 2 --num-runs 5 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name "$NAME"

# 평가가 성공한 경우에만 모든 task의 metric을 계산하고 모델 저장소에 업로드
"$GR00T_PYTHON" "$PROTOSS_ROOT/Protoss/process_robolab_results.py" \
  --results-dir "$ROBOLAB_ROOT/output/$NAME" \
  --start-number "$EXP_NUM" \
  --hf-repo-id "$HF_REPO_ID"
)
```
# BananaInBowlTask \
# BananaOnPlateTask \
# BananasInBinOneMoreTask

기본값 `ROBOLAB_STACK=595`는 새 6.0.1 환경을 사용한다. 기존 5.1 환경에서 실행할 때만 `51`로 바꾼다. 선택한 환경에 따라 metric 입력 경로도 함께 바뀐다. 첫 확인은 `--num-runs 1`로 줄일 수 있다. 각 모델을 GR00T 기본 서버의 `--model-path`에 지정하고 포트 5557에 연결해 평가한다. baseline은 원본 checkpoint, Model-FM/S/E는 adapter와 processor가 있는 학습 output 경로를 사용한다. Model-S/E도 학습이 끝난 뒤 평가한다. 결과 폴더 이름을 `baseline_run01`, `modelFM_run01`, `modelS_run01`, `modelE_run01`처럼 각각 다르게 지정한다. Task 목록, control rate, horizon, simulator revision을 같게 유지하고 여러 run으로 비교한다.

평가가 정상 종료되면 `experiment/README_exp.md`의 task 분리·metric 평균 계산을 서버에서 자동 수행한다. 첫 등장 task부터 `RoboLab05.jsonl`, `RoboLab06.jsonl`, …으로 번호를 부여하고 **모든 task**를 계산한다. 평균은 기존 스크립트와 동일하게 성공·실패를 포함한 전체 episode의 유한한 숫자만 사용하며, metric별 유효 표본 수와 task별·전체 성공률도 기록한다. RoboLab이 기록한 `metrics`를 집계하며 HDF5에서 새 trajectory metric을 계산하지는 않는다.

로컬 결과는 `$ROBOLAB_ROOT/output/$NAME/metrics/`, 원격 결과는 모델 저장소의 `evaluations/$NAME/`에 저장된다. 원본 `episode_results.jsonl`, task별 JSONL, `metrics_summary.json`, `metrics_summary.txt`를 업로드한다. 모델 가중치·HDF5·영상은 업로드 대상이 아니다. `MODEL_NAME`은 모델 서버의 Hugging Face 저장소 이름과 맞춘다. `NAME`은 평가 결과 폴더 이름이므로 서로 달라도 된다. 다른 소유자의 모델은 블록 실행 전에 `HF_REPO_ID=소유자/모델저장소`를 export한다. 설치할 때 로그인한 쓰기 권한 토큰을 사용하며, 존재하지 않는 저장소에는 업로드하지 않는다.

평가 오류 시 계산·업로드를 실행하지 않으며, 계산·업로드 오류도 종료 코드로 전달한다. 업로드만 실패했다면 평가를 다시 하지 않고 위 변수들을 설정한 뒤 마지막 `process_robolab_results.py` 명령을 재실행할 수 있다. 595 환경의 결과는 `/workspace/robolab-595/RoboLab/output/$NAME`, 기존 5.1 환경의 결과는 `/workspace/RoboLab/output/$NAME`에 있다. 비교 실험에서는 성공률과 RoboLab이 기록한 SPARC·ISJ·경로 길이·속도 등을 같은 simulator 조건에서 비교한다.


`--checkpoint-step 500`은 다운로드한 run의 `checkpoint-500/adapter`를 사용한다. 최종 adapter를 쓰려면 이 옵션을 생략한다. 지정 checkpoint가 없으면 오류로 종료한다. 저장 제한(현재 최대 3개)으로 이미 삭제된 step은 HF에도 없을 수 있으며 해당 adapter 파일이 별도로 남아 있어야 추론할 수 있다.
