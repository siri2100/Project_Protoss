# Project Protoss v1.7.13 — 학습

RunPod Linux x86_64, CUDA/RTX GPU 기준이다. 아래 순서로 설치한 뒤 모델별 학습 명령을 실행한다.

| 모델 | Loss |
|---|---|
| Model-S | weighted FMD + joint SPARC loss |
| Model-E | weighted FMD + FK EE path length |

## 1. 환경과 프로젝트 설치

프로젝트가 없다면 clone한다. 기존 checkout은 이 명령을 생략한다.

```bash
git clone --branch v1.7 https://github.com/siri2100/Project_Protoss.git /workspace/Project_Protoss
```

프로젝트 루트에서 **Protoss 폴더로 이동**한다. 이후 모든 명령은 이 위치에서 실행한다. 다른 checkout을 사용하면 첫 번째 경로를 바꾼다.

```bash
cd /workspace/Project_Protoss
cd Protoss
```

시스템 의존성·uv·Git LFS를 설치한다(root 기준).

```bash
python3 run.py setup-system
```

```bash
python3 run.py folders
```

기존 checkout을 갱신할 때는 `python3 run.py update-project`를 사용한다.

## 2. GR00T 설치와 Hugging Face 로그인

```bash
python3 run.py setup-groot
```

고정 버전의 GR00T와 Python 환경을 설치하고, **HF 로그인 → Cosmos 접근 확인 → pretrained 다운로드**를 진행한다. [Cosmos 모델](https://huggingface.co/nvidia/Cosmos-Reason2-2B) 접근 승인을 받은 계정과 모델 읽기·쓰기 권한이 있는 토큰을 사용한다. 이후 학습·평가는 저장된 로그인을 재사용한다.

## 3. RoboLab 설치

학습 후 자동 평가에 사용할 RoboLab·IsaacLab·Isaac Sim 6.0.1과 전용 Python 환경을 설치한다. 기존 설치가 있다면 같은 명령으로 최신 호환 패치를 적용한다.

```bash
python3 run.py setup-robolab
```

설치 후 물리·카메라 동작을 검사하고 `SMOKE PASS`를 확인한다.

```bash
python3 run.py smoke-robolab
```

기본 설치 경로는 `/workspace/robolab-595`다. 다른 곳에 설치했다면 [run_config.py](Protoss/run_config.py)의 `ROBOLAB_STACK_ROOT`를 맞춘다. 학습 전 `Run setup first.`가 나오면 이 경로의 `.venv/bin/python`을 실행할 수 없는 상태이므로 설치·경로를 확인한다. 설치 상세 내용은 [RoboLab 문서](README_robolab_595.md)에 있다.

## 4. 데이터 다운로드

```bash
python3 run.py setup-data
```

```bash
python3 run.py download-droid
```

기본은 성공·언어 정보가 있는 DROID episode 300개다. 개수는 `run_config.py`의 `DOWNLOAD_EPISODES`에서 바꾼다. 이미 받은 데이터는 재사용한다.

## 5. 학습 데이터 준비

```bash
python3 run.py prepare-data
```

원본 episode 전체를 train/test 분할 없이 공통 trainset으로 변환한다. 시간 축과 15Hz를 유지하며 smoothing·배속을 적용하지 않는다. 300개라면 `Prepared train=300`을 확인한다. 기존 변환 데이터가 있으면 재변환은 생략한다.

데이터는 `<프로젝트>/Protoss/data`, pretrained와 학습 모델은 **프로젝트 바로 상위의 `checkpoint/`**에 저장한다. 예: `/workspace/Project_Protoss` → `/workspace/checkpoint`. `cd Protoss` 여부는 저장 경로에 영향을 주지 않는다.

## 6. Model-S/E LoRA 학습

v1.7.13은 두 모델 모두 같은 원본 데이터와 pretrained 모델에서 **DiT 짝수 인덱스 block(0, 2, 4, …)**의 QKV + FFN에만 LoRA를 학습한다. GR00T에서는 cross-attention block에 해당한다. 원본 weight와 홀수 인덱스 block은 동결한다. 명령 인자는 [run_config.py](Protoss/run_config.py) 기본값보다 우선한다.

| 설정 | 위치 |
|---|---|
| 모델 이름·학습/save step·LR | `--name`, `--steps`, `--save-steps`, `--lr` |
| rank·alpha·batch·gradient accumulation·workers | `TRAIN_COMMON` |
| 모델별 기본값·loss weight | `TRAIN["S"/"E"]` |
| 평가 task·환경 수·반복 수·horizon·seed | `EVALUATION` |
| HF 업로드 계정 | `HF_ACCOUNT` (기본: 로그인 계정) |

### 6.1 Model-S

```bash
python3 run.py train-s --name v1.7.13_model_s_step500 --steps 500 --save-steps 500 --lr 1e-5 --s-fmd-weight 1 --s-sparc-weight 0.001
```

`loss = fm_weight × FMD + sparc_weight × (-mean joint SPARC)`다. FMD는 같은 관측·noise·timestep에서 pretrained와 LoRA 모델의 flow velocity 차이를 보존한다. 관절 7개의 속도 프로파일에서 SPARC를 계산하며, 부드러울수록 SPARC가 덜 음수이므로 loss에는 부호를 반전한다.

`--s-fmd-weight`, `--s-sparc-weight`로 두 항을 조절한다. SPARC는 속도 크기 자체를 낮추는 항은 아니다. 구현은 [model_s_loss.py](Protoss/src/model_s_loss.py), [sparc_loss.py](Protoss/src/sparc_loss.py)에 있다.

### 6.2 Model-E

```bash
python3 run.py train-e --name v1.7.13_model_e_step500 --steps 500 --save-steps 500 --lr 1e-5 --e-fmd-weight 1 --e-path-length-weight 0.01
```

`loss = fm_weight × FMD + path_length_weight × mean(sum_t ||FK(q[t+1])-FK(q[t])||₂)`다. Joint 출력을 절대 관절각으로 복원한 뒤 RoboLab DROID의 FK로 EE 위치를 계산한다. EE는 Robotiq gripper `base_link`이며 경로 길이 단위는 **미터(m)**다.

`--e-fmd-weight`, `--e-path-length-weight`로 두 항을 조절한다. 구현은 [model_e_loss.py](Protoss/src/model_e_loss.py), [droid_kinematics.py](Protoss/src/droid_kinematics.py)에 있다.

### 6.3 학습 재개

```bash
python3 run.py resume-s --name v1.7.13_model_s_step500 --steps 500 --s-fmd-weight 1 --s-sparc-weight 0.001
```

Model-E는 `resume-e`를 사용한다. 같은 run 이름과 loss 설정을 유지한다. Loss·LR 등 학습 옵션은 저장된 실행 인자를 복원하고, 평가 task·반복 수는 현재 `EVALUATION` 설정을 사용한다. 새 실험에는 새 `--name`을 사용한다. 이전 전체 layer LoRA checkpoint는 짝수 layer 학습으로 재개할 수 없다. 기존 jerk·관절 path length 목적 함수의 checkpoint는 새 SPARC·EE loss로 재개할 수 없다.

### 6.4 학습 로그와 저장 파일

콘솔에 FMD·보조 loss·total을 출력하고 `training_loss.csv`, `training_loss.png`에 기록한다. 그래프는 각 loss와 total의 세 패널이다. 모델 폴더에는 `checkpoint-*/`, `adapter/`, `processor/`, loss 설정과 학습 그래프를 저장한다.

### 6.5 학습 → RoboLab 테스트 → 평가점수 자동 계산

기본 학습 명령은 **학습 → 그래프 저장 → HF 모델 업로드 → HF 다운로드·추론 서버 → RoboLab 평가 → 점수 계산·HF 업로드**까지 실행한다. 평가 testset은 [robolab_testset.py](Protoss/src/robolab_testset.py)의 지정된 5개 task이며, **각 task를 환경 1개 × 10회 = 총 50 episode** 실행한다. Task별·전체 metric과 성공률을 같은 모델 저장소에 업로드한다. HF 모델 저장소는 기본 `로그인계정/모델이름`이다. 3절의 RoboLab 설치를 먼저 완료해야 하며, 학습 전 사전 검사에서 실패하면 학습을 시작하지 않는다.

| 추가 옵션 | 동작 |
|---|---|
| `--no-eval` | RoboLab 사전 검사·자동 평가를 생략하고 학습·모델 업로드만 실행 |
| `--no-hf-upload` | HF 전송을 생략하고 로컬 모델·평가 결과를 저장 |
| `--dry-run` | 실제 실행 없이 명령 확인 |
| `--repo-id ACCOUNT/MODEL` | HF 저장소 직접 지정 |

평가 결과는 모델 폴더의 `evaluation_status.json`, `evaluation_logs/`, `evaluations/`에 남는다. 원본 HDF5는 RoboLab의 `output/`에 보관한다. 수동 추론·평가와 점수 업로드 재시도는 [readme_test.md](readme_test.md)를 따른다.
