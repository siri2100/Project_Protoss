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
python3 run.py train-e --name v1.7.13_model_e_fast_step500 --steps 500 --save-steps 500 --lr 1e-5 --e-fmd-weight 1 --e-fast-weight 1 --e-limit-weight 0.01 --e-speedup 1.2 --e-dt 0.06666666666666667 --e-velocity-limit 2 --e-acceleration-limit 10
```

Model-E: `loss = e_fmd_weight × FMD + e_fast_weight × fast_loss + e_limit_weight × limit_loss`.

- `fast_loss`: pretrained가 예측한 관절·그리퍼 궤적의 시간을 `e_speedup`배 압축해 추종. 기본 1.2이며 마지막 위치를 유지한다. 원본 데이터는 배속하지 않는다.
- `limit_loss`: 관절 속도·가속도의 상한 초과에 대한 정규화된 penalty. 상한 단위는 rad/s·rad/s²이며 모든 관절에 공통 적용하는 실험용 soft bound다.
- 콘솔·CSV·그래프에 `fmd_loss`, `fast_loss`, `limit_loss`, `total_loss`를 기록한다. 설정은 `run_config.py`의 `TRAIN["E"]`에서도 변경할 수 있다.

기존 EE path length loss의 checkpoint는 새 목적함수로 재개할 수 없다. 새로운 `--name`으로 학습하고, 성공률·성공 episode 완료 시간·SPARC를 함께 비교한다. 실제 속도 향상은 RoboLab 평가로 확인한다.
