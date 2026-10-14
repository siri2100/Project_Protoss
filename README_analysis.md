# Model-FM 성능 저하 진단

현재 DROID held-out 결과는 baseline 대비 Model-FM MAE 약 20.2%, MSE 약 33.6% 증가다. 다음 세 검증으로 구현·processor·학습 추세를 구분한다. 진단 구현은 `Protoss/src/analyze_groot.py`, CLI는 `Protoss/analyze_groot.py`다. 기존 학습 파일을 덮어쓰지 않고 별도 analysis 경로에 export한다.

## 1. 실행 환경 준비

RunPod에서 프로젝트의 최신 파일과 기존 GR00T 환경을 준비한다. 아래 설정과 이후 명령은 **같은 Bash 터미널**에서 순서대로 실행한다. 다운로드는 기본 README 10.3, testset은 11.2를 따른다. 이 분석에는 DROID testset이 필요하다.

```bash
set -e
export PROTOSS_ROOT=/workspace/Project_Protoss_v1.5
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
cd "$PROTOSS_ROOT/Issac-GR00T-N17"
export BASE_MODEL="$PROTOSS_ROOT/Protoss/checkpoint/GR00T-N1.7-DROID"
export TRAIN_RUN="$PROTOSS_ROOT/Protoss/checkpoint/model_FM_lora"
export FM_MODEL="$TRAIN_RUN/inference"
export DATA_ROOT="$PROTOSS_ROOT/Protoss/data"
export ANALYSIS_ROOT="$PROTOSS_ROOT/experiment/DROID/analysis_run01"
mkdir -p "$ANALYSIS_ROOT"
test -f "$BASE_MODEL/config.json"
test -f "$FM_MODEL/config.json"
test -f "$DATA_ROOT/testset/meta/info.json"
uv run --no-sync python "$PROTOSS_ROOT/Protoss/analyze_groot.py" --help
```

예전 `model_1_lora` 폴더를 사용한다면 `TRAIN_RUN`을 그 경로로 바꾼다. Hugging Face에서 inference 모델만 받은 Pod에는 학습 중 checkpoint가 없다. 4절에는 원래 학습 output의 `checkpoint-*`, `lora_settings.json`, `processor/`가 필요하다. 기존 output을 복원해야 하며 inference만으로 과거 step을 재구성할 수 없다.

## 2. 학습 전 LoRA·merge·디스크 재로딩 출력 비교

하나의 baseline 모델을 순서대로 사용한다. baseline 반복 추론, B=0 초기 LoRA, 메모리에서 merge, merge 모델 저장 후 native 재로딩을 동일한 입력·seed로 비교한다. 모든 action 출력에 대해 shape, 최대 절대 차이와 allclose를 기록한다. 실제 GR00T inference 경로에서 비교하며 processor는 baseline 것을 유지한다.

```bash
uv run --no-sync python "$PROTOSS_ROOT/Protoss/analyze_groot.py" \
  --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" equivalence \
  --base-model "$BASE_MODEL" --dataset-root "$DATA_ROOT" \
  --output-dir "$ANALYSIS_ROOT/equivalence" \
  --rank 8 --alpha 16 --device cuda:0 --seed 42 \
  --traj-ids 0 1 --frame-ids 0 16 --denoising-steps 4 \
  --atol 1e-5 --rtol 1e-5
```

산출물은 `equivalence/equivalence.json`과 `equivalence/zero_lora_inference/`다. export는 전체 모델이므로 충분한 디스크 공간이 필요하다. 모델 두 개를 동시에 GPU에 유지하지 않는다. 재실행할 때는 `equivalence_run02`처럼 새 output 경로를 사용한다.

- 모든 stage가 통과: 선택한 관측에 대해 초기 LoRA와 merge/export 경로가 baseline 출력을 보존한다. 학습된 adapter의 정확성까지 증명하지는 않는다.
- `baseline_repeat`부터 실패: seed 고정 후에도 발생하는 비결정성부터 확인한다. tolerance를 임의로 크게 올려 통과시키지 않는다.
- baseline 반복은 통과하지만 `zero_lora` 실패: adapter 적용과 eval mode를 점검한다.
- `merged` 또는 `reloaded`만 실패: merge/export/loading 경로를 점검한다.

기본 tolerance는 BF16 비결정성에 대해 엄격할 수 있다. 보고서의 차이 크기와 baseline 반복 차이를 함께 해석한다. 음수 frame ID는 사용하지 않는다.

## 3. processor 설정·통계 비교

이 단계는 GPU 모델을 로딩하지 않는다. 모델 루트 또는 `processor/`의 processor/statistics/normalization/modality JSON을 비교한다. 파일별로 차이의 JSON 경로와 두 값을 모두 보관한다.

```bash
uv run --no-sync python "$PROTOSS_ROOT/Protoss/analyze_groot.py" processor \
  --base-model "$BASE_MODEL" --model-path "$FM_MODEL" \
  --output-dir "$ANALYSIS_ROOT/processor"
```

`processor/processor_diff.json`에서 먼저 `statistics.json`의 DROID embodiment에 해당하는 state/action의 min/max, mean/std, quantile 또는 relative 통계를 확인한다. `processor_config.json`의 modality keys, delta_indices와 normalization 설정도 확인한다. 다른 embodiment의 통계 변화는 DROID 추론에 직접 사용되지 않을 수 있다.

통계가 다르다는 사실 자체는 오류가 아니다. 학습에서 사용한 normalization과 추론 processor가 일치해야 한다. baseline 통계를 학습된 모델에 무조건 교체하면 오히려 잘못된 decoding이 될 수 있다. 이 비교는 JSON artifact 비교이며 같은 raw 입력에 대한 normalized tensor 비교나 processor 교체 실험은 포함하지 않는다.

## 4. 중간 checkpoint export와 동일 held-out 평가

### 4.1 남아 있는 checkpoint 확인

```bash
find "$TRAIN_RUN" -maxdepth 2 -name trainer_state.json -print
ls "$TRAIN_RUN"/checkpoint-*/model*.safetensors*
cat "$TRAIN_RUN/lora_settings.json"
```

현재 학습은 checkpoint를 최대 3개 유지하므로 초기 500-step checkpoint 등이 이미 지워졌을 수 있다. 없는 step은 분석할 수 없다. 지원 입력은 프로젝트 커스텀 LoRA parametrization이 포함된 전체 safetensors trainer checkpoint다. 일반 full fine-tuning weight, PEFT adapter-only, optimizer 파일은 이 export의 입력이 아니다.

### 4.2 baseline 평가

기존 결과를 복사해 재사용하려면 manifest, seed, trajectory 목록과 horizon이 같아야 한다. 아래는 비교 폴더 안에서 새로 평가하는 명령이다.

```bash
uv run --no-sync python "$PROTOSS_ROOT/Protoss/eval_groot_models.py" \
  --model baseline --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-root "$DATA_ROOT" --model-path "$BASE_MODEL" \
  --output-dir "$ANALYSIS_ROOT/evaluations/baseline" \
  --seeds 42 43 44 --execution-horizon 8 --denoising-steps 4 --steps 100000
```

### 4.3 남아 있는 checkpoint를 각각 merge/export하고 평가

전체 testset × 세 seed라서 시간이 걸린다. 중간 export는 CPU RAM에서 baseline과 한 개의 checkpoint shard를 로딩한다. GPU는 다음 평가에 사용한다. 각 step마다 전체 merged 모델을 저장하므로 디스크 사용량도 증가한다.

```bash
for CHECKPOINT in "$TRAIN_RUN"/checkpoint-*; do
  [ -d "$CHECKPOINT" ] || continue
  STEP_NAME=$(basename "$CHECKPOINT")
  uv run --no-sync python "$PROTOSS_ROOT/Protoss/analyze_groot.py" \
    --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" export \
    --base-model "$BASE_MODEL" --checkpoint "$CHECKPOINT" \
    --output-dir "$ANALYSIS_ROOT/exports/$STEP_NAME/inference"

  uv run --no-sync python "$PROTOSS_ROOT/Protoss/eval_groot_models.py" \
    --model FM --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
    --dataset-root "$DATA_ROOT" \
    --model-path "$ANALYSIS_ROOT/exports/$STEP_NAME/inference" \
    --output-dir "$ANALYSIS_ROOT/evaluations/$STEP_NAME" \
    --seeds 42 43 44 --execution-horizon 8 --denoising-steps 4 --steps 100000
 done
```

rank/alpha는 원래 run의 `lora_settings.json`에서 읽는다. 모든 weight key·shape를 검사하고 누락 시 중단한다. processor는 checkpoint의 것을 우선 사용하고, 없으면 같은 run의 processor를 사용한다. `analysis_export.json`에 출처를 기록한다. 원래 trainer/optimizer state는 변경하지 않는다. 다른 run의 processor나 baseline processor로 조용히 대체하지 않는다.

빠른 확인은 **baseline과 모든 checkpoint 명령에 동일하게** `--traj-ids 0 1 2 --seeds 42 --steps 400`을 적용한다. 빠른 확인 결과와 전체 평가를 한 results 폴더에 섞지 않는다.

### 4.4 최종 inference 모델도 평가

중간 checkpoint와 최종 export 사이에 차이가 있는지 함께 확인한다.

```bash
uv run --no-sync python "$PROTOSS_ROOT/Protoss/eval_groot_models.py" \
  --model FM --groot-root "$PROTOSS_ROOT/Issac-GR00T-N17" \
  --dataset-root "$DATA_ROOT" --model-path "$FM_MODEL" \
  --output-dir "$ANALYSIS_ROOT/evaluations/final_inference" \
  --seeds 42 43 44 --execution-horizon 8 --denoising-steps 4 --steps 100000
```

### 4.5 오차 추세와 기존 train log 수집

```bash
uv run --no-sync python "$PROTOSS_ROOT/Protoss/analyze_groot.py" trend \
  --results-root "$ANALYSIS_ROOT/evaluations" --training-root "$TRAIN_RUN" \
  --output-dir "$ANALYSIS_ROOT/trend"
```

`trend.csv`는 run/step/seed별 평균 MSE·MAE, `trend.json`은 비교 설정도 포함한다. step이 없는 baseline/final inference의 step 열은 비워 둔다. 각 trainer_state.json의 log_history도 별도 JSON으로 추출한다. 평가 조건 불일치, 미완료 로그와 누락 episode는 오류로 처리한다.

- step 증가에 따라 held-out 오차가 줄어듦: 추가 학습을 검토할 근거다. RoboLab 성공률 개선은 별도 확인한다.
- train loss는 줄고 held-out 오차가 커짐: 과적합 또는 데이터 적응 과정의 성능 저하를 의심한다.
- 최초 checkpoint부터 나쁨: processor, 학습 설정, adapter gradient부터 확인한다.
- 최종 step의 checkpoint 재export와 final inference가 크게 다름: 저장/merge/processor 경로와 두 weight가 같은 step인지 확인한다.

MSE/MAE는 joint와 gripper를 합친 원본 action 오차이며 success rate나 smoothness가 아니다. 현재 upstream plot은 episode별로 덮어쓰므로 마지막 episode PNG를 전체 결과로 해석하지 않는다.

## 5. 로컬 검증 범위

CLI, JSON 비교, 평가 로그 완결성·조건 검증과 CSV 집계는 CPU 테스트로 검증한다. 실제 GR00T GPU inference, safetensors checkpoint export와 결과 일치 여부는 RunPod에서 위 명령으로 확인해야 한다. 진단을 실행하기 전에는 LoRA 설정·학습 step을 변경하지 않는다.

## v1.3 비교 실험

v1.3은 영상 `[0]`, state dropout `0.0`으로 새로 학습한다. 기존 v1.2 분석을 재현할 때는 위 `TRAIN_RUN`을 이전 학습 폴더로 유지한다. v1.3 분석은 `model_FM_lora_v13`과 새 `analysis_v13_run01`을 사용한다. 새 모델은 영상 입력 조건이 달라지므로 v1.2 결과와 단순히 LoRA 효과만 비교하는 실험으로 해석하지 않는다.
