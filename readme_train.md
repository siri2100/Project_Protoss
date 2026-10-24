# Project Protoss — Train


## Step 01. Set environment
```bash
git clone --branch v1.7 https://github.com/siri2100/Project_Protoss.git /workspace/Project_Protoss
cd /workspace/Project_Protoss
cd Protoss
```

시스템 의존성·uv·Git LFS를 설치한다(root 기준).
```bash
python3 run.py setup-system
python3 run.py folders
```

## Step 02. Download GR00T and login HuggingFace
```bash
python3 run.py setup-groot
python3 run.py login
```

## Step 03. Install RoboLab
```bash
python3 run.py setup-robolab
python3 run.py smoke-robolab
```

## Step 04. Download dataset

```bash
python3 run.py setup-data
python3 run.py download-droid
```

원본 episode 전체를 train/test 분할 없이 공통 trainset으로 변환한다. 시간 축과 15Hz를 유지하며 smoothing·배속을 적용하지 않는다. 300개라면 `Prepared train=300`을 확인한다. 기존 변환 데이터가 있으면 재변환은 생략한다.
```bash
python3 run.py prepare-data
```

## Step 05. Train model
| 설정 | 위치 |
|---|---|
| 모델 이름·학습/save step·LR | `--name`, `--steps`, `--save-steps`, `--lr` |
| rank·alpha·batch·gradient accumulation·workers | `TRAIN_COMMON` |
| 모델별 기본값·loss weight | `TRAIN["S"/"E"]` |
| 평가 task·환경 수·반복 수·horizon·seed | `EVALUATION` |
| HF 업로드 계정 | `HF_ACCOUNT` (기본: 로그인 계정) |

### Model-S
```bash
python3 run.py train-s --name v1.7.13_model_s_step500 --steps 500 --save-steps 500 --lr 1e-5 --s-fmd-weight 1 --s-sparc-weight 0.001
```
### Model-E
```bash
python3 run.py train-e --name v1.7.13_model_e_step500 --steps 500 --save-steps 500 --lr 1e-5 --e-fmd-weight 1 --e-path-length-weight 0.01
```
