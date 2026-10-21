# Project Protoss v1.7 — 학습

Model-FM/S/E는 같은 원본 DROID 전체 데이터와 pretrained 모델에서 독립적으로 QKV + FFN LoRA를 학습한다. Baseline은 원본 pretrained 평가 전용이다. Model-FM(v1.6.14)은 All QKV, LR=1e-6으로 학습했던 기준 모델이며, 새 FM 실험은 현재 QKV + FFN 범위를 사용한다.

| 모델 | Loss |
|---|---|
| Model-FM | 정답 action의 FM |
| Model-S | weighted FMD + normalized joint jerk |
| Model-E | weighted FMD + joint path length |

모든 실행 블록은 **한 줄**이다. 프로젝트 루트에서 `python3 Protoss/run.py 명령`을 실행한다. launcher는 Python 표준 라이브러리만 사용하고, 학습·추론은 자동으로 GR00T `.venv/bin/python`을 사용한다. 환경 export, uv 활성화, cd를 매번 입력할 필요가 없다. Python 파일은 `bash`가 아닌 `python3`로 실행한다.

자주 바꾸는 값은 [Protoss/run_config.py](Protoss/run_config.py)에 모았다.

| 설정 | 의미 |
|---|---|
| `TRAIN_COMMON` | rank/alpha, batch, gradient accumulation, workers, seed |
| `TRAIN["FM"/"S"/"E"]` | version, 학습/save step, LR, 각 loss weight |
| `EVALUATION` | task, 환경 수, 반복 수, horizon, seed, device |
| `MODEL_NAME`, `CHECKPOINT_STEP` | 별도 평가·서버·업로드에서 선택할 저장 모델 |
| `HF_ACCOUNT` | 기본은 로그인 계정; organization이면 직접 지정 |
| `ROBOLAB_STACK_ROOT` | 동작 확인된 595 simulator stack 경로 |

프로젝트 경로는 실행 파일 위치에서 계산하고 checkpoint는 **프로젝트 바로 상위의 `checkpoint/`**에 저장한다. RunPod의 `Project_Protoss2`에서도 그 checkout의 실행 파일을 사용하면 된다. 로컬에 Project_Protoss2 폴더를 따로 만들지 않는다. `HF_HOME=/workspace/.cache/huggingface`, GPU 0을 기본으로 사용하며 config에서 바꾼다.

명령만 확인하려면 `--dry-run`을 붙인다. 로그인·다운로드·설치·파일 쓰기를 실행하지 않는다.

```bash
python3 Protoss/run.py train-s --dry-run
```

## 1. 환경과 프로젝트 설치

RunPod Ubuntu 22.04/24.04, Linux x86_64, CUDA/RTX GPU 기준이다. 아래 시스템 설치는 root에서 실행한다. 프로젝트가 없다면 먼저 받는다. 기존 checkout이 있다면 clone은 생략한다.

```bash
git clone --branch v1.7 https://github.com/siri2100/Project_Protoss.git /workspace/Project_Protoss
```

프로젝트 루트로 이동한 뒤 시스템 의존성·uv·Git LFS를 설치한다. Git 자체가 없는 최소 이미지에서는 먼저 OS 패키지 관리자로 Git을 설치해야 한다.

```bash
python3 Protoss/run.py setup-system
```

### 프로젝트 받기

기존 checkout의 `v1.7`을 fetch하고 fast-forward한다. 로컬 변경이나 분기가 있으면 강제로 덮어쓰지 않고 중단한다.

```bash
python3 Protoss/run.py update-project
```

### 저장 폴더 준비

```bash
python3 Protoss/run.py folders
```

원본은 `Protoss/data/source`, 변환 데이터는 `Protoss/data/trainset`, pretrained 및 학습 모델은 `../checkpoint`에 둔다. 기존 다른 경로의 파일을 자동 이동하지 않는다.

## 2. GR00T-N1.7 설치와 weight 다운로드

이미 설치했다면 생략한다. GR00T는 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`로 고정한다. Git 설정 격리, HTTP/1.1·protocol v0, submodule/LFS, Python 3.12 환경 설치, HF 로그인 확인, Cosmos 접근 확인, pretrained 다운로드를 순서대로 실행한다.

```bash
python3 Protoss/run.py setup-groot
```

`.git` 없는 불완전한 소스는 삭제하지 않고 백업한다. 다른 커밋의 기존 GR00T checkout은 덮어쓰지 않는다. 디렉터리 이름은 config의 `GROOT_DIRECTORY`로 지정하며 기본은 실제 프로젝트 이름인 `Issac-GR00T-N17`이다.

### GitHub clone에서 Username 오류가 발생한 경우

공개 저장소의 HTTPS 인증 실패이며 모델 접근 승인과 별개다. launcher는 사용자·시스템 Git 인증 설정을 해당 설치 프로세스에서만 격리하고 본체·submodule·LFS에 같은 HTTP 옵션을 사용한다. 본체 clone만 성공했다면 `setup-groot`를 다시 실행해 이어간다. 계속 Username을 요구하면 입력을 취소하고 프록시·네트워크 응답을 확인한다. GitHub 비밀번호 입력으로 해결하는 절차가 아니다.

### Cosmos 접근 확인에서 401 / GatedRepoError가 발생한 경우

[Cosmos 모델 페이지](https://huggingface.co/nvidia/Cosmos-Reason2-2B)에서 접근 승인을 받은 같은 계정으로 로그인한다. 토큰은 모델 읽기 및 저장소 생성·쓰기 권한이 있어야 한다. `whoami` 성공만으로 gated 접근 승인이 확인되지는 않는다.

```bash
python3 Protoss/run.py login
```

이 복구 명령은 기존 토큰 환경변수를 자식 프로세스에서 해제하고 로그인·접근 확인·pretrained 다운로드를 다시 실행한다. 다른 터미널의 오래된 `HF_TOKEN`도 해제하거나 교체해야 한다. 토큰은 코드나 Git에 기록하지 않는다. 학습·평가는 저장된 로그인을 재사용한다.

## 3. 데이터 준비 의존성 확인

GR00T 환경에 HF·PyArrow·Pandas·SciPy를 설치하고 CUDA 텐서 생성 및 학습 CLI를 확인한다.

```bash
python3 Protoss/run.py setup-data
```

LoRA 대상은 action DiT의 `attn1.to_q/to_k/to_v`와 `ff.*` 내부 Linear(`ff.net.0.proj`, `ff.net.2`)다. Attention output projection은 제외한다. LLM, vision encoder, projector, state/action encoder/decoder와 pretrained weight는 고정한다. 실행 config는 rank=32, alpha=64이며 adapter dropout=0이다.

## 4. DROID 데이터 다운로드

이미 받은 source가 있다면 생략한다. 성공·언어 있는 episode 300개를 기본으로 다운로드하며 개수는 config의 `DOWNLOAD_EPISODES`에서 바꾼다. 일반 HF 다운로드를 사용하고 같은 명령을 재실행하면 캐시를 재사용한다.

```bash
python3 Protoss/run.py download-droid
```

프로젝트 파일이 없다는 오류는 uv 재설치로 해결되지 않는다. 실제 프로젝트 checkout과 `Protoss/prepare_droid.py`, `Protoss/src/prepare_droid.py`를 확인한다.

## 5. 공통 원본 DROID 데이터 준비

```bash
python3 Protoss/run.py prepare-data
```

Train/test 분할 없이 원본 episode 전체를 공통 trainset으로 변환한다. Joint/gripper command, state, frame 순서와 15Hz를 유지하며 smoothing·배속·시간 보간은 적용하지 않는다. DROID `[EEF(9), gripper(1), joints(7)]` 계약과 GR00T normalization/relative 통계를 사용한다. 영상은 H.264 재인코딩으로 압축이 바뀌지만 시간 축은 유지한다.

결과는 `Protoss/data/trainset/`, `styles.json`, `inspection.json`이다. 300개이면 `Prepared train=300`을 확인한다. 기존 trainset/testset 또는 manifest가 있으면 중단한다. 이전 분할·변형 데이터는 백업한 뒤 source에서 다시 변환하며 자동 삭제하지 않는다. 진단만 하려면 다음 명령을 사용한다.

```bash
python3 Protoss/run.py prepare-data -- --inspect-only
```

## 6. Model-FM/S/E LoRA 학습

실험 이름은 `version_model_<모델>_step<max_steps>`로 생성한다. 같은 이름의 output이 비어 있지 않으면 덮어쓰지 않는다. 새 실험은 config의 version을 바꾸거나 `--name 새이름`을 지정한다. 실행 인자는 `../checkpoint/launches/<이름>.json`, 최근 경로는 `latest_<모델>_lora_v16.txt`에 기록한다.

원본 pretrained processor·normalization·영상 시점을 유지하고 subset 통계로 덮어쓰지 않는다. `lora_settings.json`의 processor 계약이나 loss/LoRA 설정이 다르면 resume를 거부한다. 이전 QKV-only·velocity·jerk-only·다른 목적 함수 checkpoint는 새 학습으로 이어붙이지 않는다.

기본적으로 **학습 → 그래프 저장 → HF 모델 업로드 → 학습 프로세스 종료 → HF 다운로드·추론 서버 → RoboLab 평가 → 점수 계산·HF 업로드**까지 실행한다. 먼저 6.6절의 simulator setup을 완료한다. 모델 저장소 기본값은 `로그인계정/output폴더명`이며 config의 `HF_ACCOUNT` 또는 `--repo-id ACCOUNT/REPO`로 바꾼다.

### 6.1 Model-FM 학습

```bash
python3 Protoss/run.py train-fm
```

현재 설정은 `v1614_model_fm_step500`, LR=1e-6, 500 step이다. 이름은 설정에서 변경할 수 있다.

### 6.2 Model-S 학습

실험 구현은 `Protoss/src/model_s_loss.py`와 `model_s_forward.py`에 모았다. `ModelSConfig`는 설정, `joint_affine()`는 관절 역정규화, `masked_jerk_loss()`는 3차 차분 제곱 평균, `ModelSLoss`는 최종 조합을 담당한다.

- 목적 함수는 `L = fm_weight × FMD + jerk_lambda × normalized_jerk_loss`다. `FMD = masked_mean((v_LoRA - stopgrad(v_pretrained))²)`이며 기존 pretrained flow velocity 보존 항을 유지한다. Model-FM의 정답 action FM 학습은 그대로이며 Model-E도 같은 FMD teacher/student 경로를 사용한다.
- Teacher/student는 관측, state 특징, noisy action, 연속 timestep과 discrete timestep을 공유한다. Teacher는 같은 PEFT DiT의 adapter를 끈 상태에서 `no_grad`로 실행하고 student에는 adapter를 적용한다. 두 pass의 DiT dropout은 끄고 이후 training 상태를 복원한다. Backbone은 한 번, DiT는 두 번 실행하며 pretrained weight를 별도로 복제하지 않는다.
- DROID `[EEF(9), gripper(1), joints(7)]` 중 joint 7개를 pretrained 통계로 역정규화한다. Student의 `a_clean = x_t + (1-t)v_LoRA`를 사용한다. Chunk 공통 기준 위치는 3차 차분에서 상쇄된다. Gripper·EEF·padding은 jerk 계산에서 제외한다.
- `jerk = (q[k+3]-3q[k+2]+3q[k+1]-q[k])/dt³`로 `raw_jerk_loss = mean(jerk²)`를 계산한다. 네 점이 모두 유효한 window만 사용한다. `normalized_jerk_loss = raw_jerk_loss / jerk_ref²`이며 raw 단위는 `rad²/s⁶`다. 기본 `dt=1/15`초다.
- 학습 중 action 역정규화는 gradient 유지를 위해 clipping 없는 affine 변환을 사용한다. Clean endpoint는 실제 multi-step 추론 action의 추정값이므로 jerk 감소와 성공률 보존은 RoboLab rollout으로 확인한다.
- 기본 첫 10개 target batch에서 정답 관절 jerk RMS를 구해 기준값으로 고정한다. 유효 window 수로 가중 평균하고, 정지 시연에서 기준값은 최소 `1 rad/s³`로 둔다. 이 준비 단계는 모델 forward/backward와 optimizer·scheduler 업데이트를 하지 않는다.
- `--s-fmd-weight` 또는 기존 이름 `--s-fm-weight`로 FMD 가중치를 조정한다(기본값 `1.0`). `run_config.py`의 `s_fmd_weight`, `s_jerk_lambda`로 두 항을 독립 조정한다.
- `--s-jerk-lambda 0.01`이 jerk 항의 가중치이며 0이면 FMD만 사용한다. `--s-jerk-reference 300`으로 기준 RMS(rad/s³)를 직접 지정하면 측정을 생략한다. `--s-calibration-batches 10`은 기준값 측정에 사용하는 batch 수다. 기존 velocity 옵션은 사용하지 않는다.
- `--s-gradient-diagnostics`를 추가하면 첫 microbatch의 FMD·정규화 jerk LoRA gradient norm을 기록한다. 추가 autograd 비용이 있다.
- CSV와 checkpoint에 `fmd_loss`, `weighted_fmd_loss`, raw/정규화/가중 jerk, 각 가중치와 기준 RMS를 기록한다. 그래프는 `fmd_loss`, `normalized_jerk_loss`, `total_loss` 세 패널이다. 콘솔은 step마다 FMD·가중 FMD·jerk·total을 한 줄에 출력한다. 기준값과 누적 통계는 재개 시 복원한다.

실행 설정은 `TRAIN["S"]`에 있다. 현재 2000 step, save=500, LR=1e-6, workers=1, FMD weight=1, jerk lambda=1e-3이다. 새 실험은 version 또는 `--name`을 바꾼다.

이전 FMD+velocity, jerk-only, FM+jerk checkpoint는 새 FMD+정규화 jerk 학습으로 재개하지 않는다. Pretrained부터 새 run으로 시작한다. 동일 목적 함수 재개 시 원래 설정을 유지한다.

```bash
python3 Protoss/run.py train-s
```

### 6.3 Model-E 학습

구현은 `Protoss/src/model_e_loss.py`에 있다. `ModelEConfig`에서 두 가중치를, `masked_path_length_loss()`에서 경로 길이 계산을 수정할 수 있다.

- `L = fm_weight × FMD + path_length_weight × path_length_loss`다. FMD는 Model-S와 동일한 pretrained/student flow velocity의 masked MSE다. 정답 action에 대한 기존 FM loss는 Model-E에서 사용하지 않는다.
- Student의 `a_clean = x_t + (1-t)v_LoRA`에서 joint 7개를 원본 pretrained 통계로 역정규화한다. 각 chunk의 `sum_t ||q[t+1]-q[t]||₂`를 구한 뒤 batch 평균한다. Gripper와 EEF는 경로 길이에 포함하지 않는다. Chunk 공통 기준 관절 위치는 차분에서 상쇄된다.
- 두 시점의 7개 관절이 모두 유효한 segment만 사용하고 padding·결측 구간을 연결하지 않는다. 유효 segment가 없는 chunk는 평균에서 제외하며, batch 전체에 segment가 없으면 오류로 중단한다. 현재 관측 state에서 첫 예측 action까지의 거리는 포함하지 않는다.
- Path length는 제곱·dt 나누기·기준값 정규화가 없는 rad 단위의 관절 공간 거리다. FMD와 단위가 다르므로 각 가중치로 상대적인 크기를 조정한다. `--e-fmd-weight`(또는 `--e-fm-weight`) 기본값은 1, `--e-path-length-weight` 기본값은 0.01이다. Path weight가 0이면 FMD만 사용한다.
- 콘솔·CSV·checkpoint 로그에 `fmd_loss`, `weighted_fmd_loss`, `path_length_loss`, `weighted_path_length_loss`, 두 가중치와 total을 기록한다. 그래프는 FMD/path length/total 세 패널이다. Loss 설정은 `model_e_loss.json`으로 checkpoint와 최종 run에 저장한다.
- 기존 FM-only Model-E checkpoint에서는 새 loss로 resume하지 않는다. 새 output에서 pretrained부터 시작하고, 같은 목적 함수 재개 시 원래 가중치를 유지한다. 역정규화는 Model-S처럼 clipping 없는 affine이며, 실제 경로 단축·성공률은 rollout에서 확인한다.

```bash
python3 Protoss/run.py train-e
```

현재 설정은 실제 기존 학습값인 500 step, LR=1e-5다. 이전 README에서 NAME의 step=2000과 max_steps=500이 달랐던 부분은 `max_steps`에서 이름을 생성해 일치시켰다.

### 6.4 중단된 PEFT 학습 재개

```bash
python3 Protoss/run.py resume-s
```

FM/E는 `resume-fm`, `resume-e`를 사용한다. 최근 run의 저장된 실행 인자를 사용하므로 현재 config의 변경값이 원래 loss/LR을 바꾸지 않는다. 기존 README로 만든 run은 `--name 기존이름`을 지정하면 `style_run.json`, `lora_settings.json`에서 설정을 복원한다. Optimizer·scheduler·RNG·trainer 상태가 포함된 checkpoint가 필요하며, 설정 JSON만 남은 output은 resume할 수 없다.

주요 산출물은 `checkpoint-*/`, `adapter/`, `processor/`, `inference/adapter_inference.json`, `lora_settings.json`, 학습 loss CSV/PNG다. Adapter는 원본 pretrained와 함께 로드하며 standalone GR00T 모델이 아니다. 학습 시 merged 모델을 별도 저장하지 않는다.

### 6.5 학습 완료 후 자동 그래프 저장

학습 종료 시 loss와 `distribution_plot.png`를 모델 폴더에 저장한다. 원본 trajectory 0의 최대 32개 프레임에서 같은 관측·seed로 pretrained와 최종 adapter의 역정규화 action 분포를 비교한다. 추론은 unmerged, denoising 4 steps다. 그래프만 다시 만들려면 다음을 실행한다.

```bash
python3 Protoss/run.py plot-distribution
```

### 6.6 학습 → RoboLab 테스트 → 평가점수 자동 계산

동작 확인된 595/Isaac Sim 6.0.1 stack에 프로젝트 업데이트 후 setup을 한 번 적용한다.

```bash
python3 Protoss/run.py setup-robolab
```

새 stack의 물리·카메라 확인은 다음 명령이다. 기본 2개 환경·600 step을 검사한다.

```bash
python3 Protoss/run.py smoke-robolab
```

이후 학습 명령 하나로 자동 평가한다. 기본은 BananaOnPlateTask, 환경 2개 × 5회, horizon 8, simulator seed 0, 영상 저장 없음이다. `EVALUATION`에서 변경한다. 학습 프로세스는 종료된 뒤 서버를 시작하지만, 평가 중 추론 서버와 simulator는 GPU 메모리를 함께 사용한다.

학습만 실행하려면 다음과 같이 기존 옵션을 `--` 뒤에 넘긴다.

```bash
python3 Protoss/run.py train-s -- --no-eval
```

HF 없이 로컬 학습·평가는 `--no-hf-upload`, 오프라인 학습만 하려면 두 옵션을 함께 넘긴다. 모든 원래 train.py 옵션은 `--` 뒤에 추가할 수 있지만 모델·기본 경로는 `--name`/`--project-root`/config에서 지정한다. 단발성 loss 변경 예시는 다음과 같다.

```bash
python3 Protoss/run.py train-s --name my_model_s -- --s-fmd-weight 1000 --s-jerk-lambda 0.001
```

저장된 모델의 서버 시작·평가·점수 업로드만 실행하려면 다음을 사용한다. `MODEL_NAME` 또는 `--name`으로 선택하며 재학습하지 않는다.

```bash
python3 Protoss/run.py evaluate
```

자동 평가 결과는 모델 폴더의 `evaluation_status.json`, `evaluation_logs/`, `evaluations/<평가이름>/`에 남는다. 원본 HDF5는 stack의 `RoboLab/output/`에 보관하고 HF에는 JSONL·점수 JSON/TXT·평가 설정만 업로드한다. RoboLab이 기록한 metric을 집계하며 HDF5에서 새 metric을 계산하지 않는다.

학습·그래프·모델 업로드가 실패하면 평가를 시작하지 않는다. 서버 준비와 평가 완료 표시·결과 파일을 확인한 뒤에만 metric을 업로드한다. 실패 시 모델을 보존하고 이 명령이 시작한 서버·Client를 정리한다. 점수 업로드만 실패했다면 평가를 반복하지 않고 재전송한다.

```bash
python3 Protoss/run.py metrics
```

## 7. 코드와 검증

```bash
python3 Protoss/run.py check
```

| 파일 | 역할 |
|---|---|
| `Protoss/run.py`, `run_config.py` | 한 줄 실행 진입점과 실험 설정 |
| `Protoss/src/recipe_runtime.py` | 환경·경로·전용 Python·명령 실행 |
| `Protoss/src/setup_recipes.py` | 설치·인증·데이터 준비 |
| `Protoss/src/experiment_recipes.py` | 학습·재개·Server/Client·업로드 |
| `Protoss/train.py`, `src/training_evaluation.py` | 학습과 자동 평가 흐름 |
| `Protoss/src/model_s_loss.py`, `model_e_loss.py` | 모델별 loss |

자동 학습·평가 통합은 CPU/프로세스 테스트 41개로 검증했다. 실제 자식 프로세스·로컬 ZeroMQ를 사용하며 모델/simulator와 HF 전송은 테스트용 실행 파일·mock으로 검사한다. 전체 GPU 학습부터 실제 rollout·HF 전송까지는 로컬에서 실행하지 않았다.

한 줄 실행 도구와 관련 회귀 테스트 20개를 통과했다. 전용 Python·환경·경로, 실제 자식 프로세스 인자, 저장 설정으로 재개, 실패 시 업로드 중단, dry-run의 무실행을 확인했다. README의 실행 블록 40개는 모두 한 줄이며 FM/S/E 생성 인자가 실제 학습 parser에서 허용되는지도 검사했다.

## 8. Hugging Face에 checkpoint 보관

### 8.1 인증

로그인은 2절 설치 때 완료한다. 토큰은 저장된 캐시에서 재사용하며 코드에 넣지 않는다.

### 8.2 모델별 adapter output 업로드

자동 업로드 재시도 또는 별도 보관에 사용한다. `MODEL_NAME` 또는 `--name`으로 선택한다.

```bash
python3 Protoss/run.py upload-model
```

전체 run을 private 모델 저장소에 업로드한다. `inference/`만 따로 올리지 않는다. Adapter·processor·통계·설정·loss·checkpoint의 재개 상태를 함께 보관한다. 전송 중 output을 수정하지 않는다. 완료 후 원격 파일 목록을 확인하고 누락 파일은 직접 재전송한다. 같은 명령으로 재시도하며 캐시·원격 파일을 삭제할 필요가 없다.

실제 전송 없이 파일을 검증하려면 아래처럼 downstream dry-run을 사용한다. 계정 조회까지 네트워크 없이 실행하려면 `--repo-id ACCOUNT/MODEL`도 지정한다. launcher의 `--dry-run`은 파일 검증도 실행하지 않고 명령만 출력한다.

```bash
python3 Protoss/run.py upload-model --repo-id ACCOUNT/MODEL -- --dry-run
```

## 9. 공통 데이터를 Hugging Face에 보관

### 9.1 데이터 업로드

```bash
python3 Protoss/run.py upload-data
```

Private dataset 저장소에 변환된 `Protoss/data` 전체를 보관한다. 원본 `source/`, `.cache`, `.git`은 제외한다. Manifest·episode 수·필수 파일을 검사하고 불완전한 데이터는 전송 전에 중단한다. 저장소 이름은 config의 `DATASET_REPO`로 지정한다. 다른 subset/seed는 새 저장소를 사용한다.

### 9.2 데이터 다운로드

```bash
python3 Protoss/run.py download-data
```

다른 소유자의 데이터는 `--repo-id OWNER/DATASET`을 지정한다. 보관한 변환 데이터가 있으면 재변환 없이 학습할 수 있다. Source subset을 바꿔 다시 생성하려면 원본도 별도로 보관해야 한다.

## 10. 경로와 실행

GR00T는 `<프로젝트>/Issac-GR00T-N17`, 데이터는 `<프로젝트>/Protoss/data`, 모델은 `<프로젝트 상위>/checkpoint`다. 모든 명령은 실행 파일 위치에서 경로를 계산하므로 다른 위치에서도 run.py의 절대경로로 실행할 수 있다. 원래 개별 Python CLI도 그대로 사용할 수 있다. 전체 명령 목록은 다음과 같다.

```bash
python3 Protoss/run.py --help
```
