# Project Protoss v1.7 — 추론·평가

프로젝트 루트에서 `Protoss` 폴더로 이동한 뒤 아래 명령을 실행한다. **학습 명령은 이미 서버 시작·RoboLab 평가·점수 업로드를 자동 실행**하므로 수동 Server/Client는 별도 디버깅할 때 사용한다. 학습·설치는 [readme_train.md](readme_train.md), 595 호환 상세 내용은 [README_robolab_595.md](README_robolab_595.md)를 따른다.

환경 export·uv 활성화·HF 계정 조회는 `run.py`가 처리한다. `.py` 파일은 `python3`로 실행한다. 경로는 실행 파일의 checkout에서 계산하므로 RunPod의 Project_Protoss2에서도 같은 명령을 쓴다.

```bash
cd Protoss
```

이후 명령은 모두 `Protoss` 폴더 기준이다. 새 터미널에서도 해당 프로젝트의 `Protoss` 폴더로 이동한다.

## 1. 공통 환경 준비

새 서버는 [학습 문서 1~3절](readme_train.md#1-환경과-프로젝트-설치)을 먼저 진행한다. RoboLab 평가만 한다면 학습 데이터를 받을 필요는 없다. 원본 pretrained는 `<프로젝트 상위>/checkpoint/GR00T-N1.7-DROID`, adapter run은 `<프로젝트 상위>/checkpoint/<모델이름>`을 사용한다. HF 로그인은 설치 때 완료하며 평가 시작 전에 모델을 다운로드한다.

자주 바꾸는 값은 [Protoss/run_config.py](Protoss/run_config.py)에 모았다.

| 설정 | 의미 |
|---|---|
| `MODEL_NAME` | HF 저장소와 로컬 모델 폴더 이름 |
| `CHECKPOINT_STEP` | 수동 서버가 사용할 checkpoint; `None`이면 최종 adapter |
| `LORA_MODE` | unmerged / merged / disabled / pretrained |
| `SERVER_HOST`, `SERVER_PORT` | Server/Client 연결 주소 |
| `EVALUATION` | task, num_envs, num_runs, horizon, seed, device |
| `ROBOLAB_STACK_ROOT`, `ROBOLAB_STACK` | 595 stack 경로 및 수동 Client의 595/51 선택 |
| `RESULT_NAME` | 기본 `None`: 실행마다 시각을 붙인 새 결과 이름 |
| `HF_ACCOUNT` | 기본 로그인 계정 또는 organization |

한 번만 다르게 실행하려면 `--name 모델이름`(또는 `--model-name`), `--checkpoint-step 500`, `--port 5558`, `--repo-id OWNER/MODEL`을 붙인다. 원래 CLI 옵션도 구분자 없이 바로 지정한다. 명령 인자가 config 기본값보다 우선하며 기존 `--` 구분자도 지원한다. `--dry-run`은 로그인·다운로드·쓰기 없이 실행 명령만 출력한다.

아래 예시의 `v1713_model_s_step500`을 실제 학습 모델의 폴더·HF 저장소 이름으로 바꾼다. `--name`을 생략하면 `run_config.py`의 `MODEL_NAME`을 사용한다.

## 2. RoboLab

### RoboLab 설치

드라이버 595는 동작 확인된 별도 Isaac Sim 6.0.1 / Python 3.12 stack을 사용한다. 업데이트 후 setup으로 최신 호환·평가 완료 패치를 반영한다.

```bash
python3 run.py setup-robolab
```

새 stack의 물리·카메라 검사다. 기본 2개 환경·600 step이며 `SMOKE PASS`를 확인한다.

```bash
python3 run.py smoke-robolab
```

기존 5.1 환경을 유지할 때만 아래 명령을 사용한다. 595 환경과 섞지 않는다.

```bash
python3 run.py setup-robolab51
```

595 setup은 수정 Client를 자동 적용한다. 기존 5.1 checkout을 업데이트한 뒤 Client만 다시 적용하려면 다음 명령을 쓴다. 원본은 한 번 백업한다.

```bash
python3 run.py patch-client51
```

Client는 서버의 영상 `[0]` 또는 `[-15, 0]` 설정에 맞춰 환경별 이력을 구성한다. State는 `[0]`을 지원한다. `horizon`은 실행할 action 수이며 영상 이력 길이와 별개다.

### Model 평가

HF 모델 다운로드부터 서버 시작·점수 계산·HF 업로드까지 한 번에 실행한다. 재학습하지 않으며 **최종 adapter**를 평가한다. 현재 `MODEL_NAME` 또는 `--name`으로 선택한다.

```bash
python3 run.py evaluate --name v1713_model_s_step500
```

모델 다운로드도 포함된다. `로그인계정/모델이름`의 학습 run(adapter·processor·설정)과 `nvidia/GR00T-N1.7-DROID`를 자동 다운로드한 뒤 추론 서버를 시작한다. 이미 받은 파일은 캐시를 재사용한다. `--no-hf-upload`를 지정하면 로컬 모델만 사용하므로 두 모델이 미리 준비되어 있어야 한다.

`EVALUATION` 기본값은 아래 10개 task를 **각각 환경 1개 × 10회, 총 100 episode** 평가하며 horizon 8, seed 0이다. 학습 완료 후 자동 평가와 별도 `evaluate`·`client`에 동일하게 적용된다. 목록은 [robolab_testset.py](Protoss/src/robolab_testset.py)에 모았다. 평가마다 새 이름과 빈 포트를 선택하고 종료·실패 시 자신이 띄운 서버를 정리한다. HF 없이 로컬 모델로 평가하려면 `--no-hf-upload`를 추가한다.

평가 task는 다음과 같다.

```text
AnimalsInBinTask
AppleAndYogurtInBowlTask
BagelsOnPlateTask
BananaInBowlTask
BananaOnPlateTask
BananasInBinOneMoreTask
BananasInBinThreeTotalTask
BananasInCrateTask
BananasOutOfBinTask
BananaThenRubiksCubeTask
```

반복 수는 `run_config.py`의 `EVALUATION`에서 바꾸거나, 학습·`evaluate`에는 `--eval-num-runs`, 수동 `client`에는 `--num-runs`를 지정한다. RoboLab의 task별 episode 수는 `num_runs × num_envs`다.

### Server

특정 checkpoint나 LoRA 모드를 디버깅할 때는 서버를 별도 터미널에서 시작한다. `CHECKPOINT_STEP=500`이면 해당 `checkpoint-500/adapter`를 사용하고 `None`이면 최종 adapter를 사용한다. 존재하지 않거나 이미 회전 삭제된 checkpoint는 오류로 종료한다.

```bash
python3 run.py server --name v1713_model_s_step500 --checkpoint-step 500
```

서버는 `--name`으로 지정한 모델(`로그인계정/모델이름`)과 원본 pretrained를 HF에서 다운로드한 뒤 로드한다. 다른 소유자는 `--repo-id OWNER/MODEL`로 지정한다. `CHECKPOINT_STEP`, `LORA_MODE`는 config에서 바꾼다. `--hf-revision COMMIT_SHA --hf-base-revision COMMIT_SHA`로 다운로드 revision을 고정할 수 있다.

| LORA_MODE | 가중치 | Processor |
|---|---|---|
| pretrained | 원본 pretrained, adapter 미로드 | 원본 pretrained |
| disabled | 원본 pretrained, adapter 미로드 | 학습 processor |
| unmerged | 원본 + LoRA, forward마다 적용 | 학습 processor |
| merged | 원본 + FP32 병합 LoRA | 학습 processor |

`disabled`와 `unmerged`의 차이로 adapter 영향을, `unmerged`와 `merged`로 실행 방식 차이를 비교한다. 모드를 바꾸면 서버를 재시작하고 같은 관측·seed로 비교한다. 원본 pretrained만 직접 실행하려면 다음을 사용한다.

```bash
python3 run.py server-baseline
```

### Client

서버가 준비되면 다른 터미널에서 실행한다. Server와 같은 `--name`, 포트를 사용한다. 실제 추론 모델은 Server의 `--name`으로 선택하며, Client의 `--name`은 결과 이름과 HF 점수 업로드 대상을 지정한다. 평가 완료 후 task별·전체 점수를 계산하고 같은 모델 저장소에 업로드한다. 결과 이름은 자동 생성된다.

```bash
python3 run.py client --name v1713_model_s_step500
```

디버깅할 때 한 task만 1회 실행하는 예시다.

```bash
python3 run.py client --name v1713_model_s_step500 --task BananaOnPlateTask --num-envs 1 --num-runs 1
```

`--stack 51`은 기존 5.1 checkout을 사용한다. `--result-name 이름`으로 결과 폴더를 직접 지정할 수 있다. 같은 이름의 폴더가 있으면 기존 JSONL·HDF5·metric 전체를 `output/_previous/<이름>/<실행ID>/`에 보관하고 원래 경로를 비운 뒤 **처음부터 새로 평가**한다. 이전 완료 episode를 건너뛰거나 기존 HDF5를 이어 쓰지 않는다. 같은 이름의 평가가 실행 중이면 중단하고 다른 이름을 요구한다. Baseline 점수를 보관할 때도 `--repo-id OWNER/EXISTING_MODEL`로 기존 업로드 저장소를 명시한다. 추론 서버와 simulator는 동시에 VRAM을 사용하며 기본 환경 수는 1개다.

이 수정은 RoboLab 실행 진입점에도 적용된다. 업데이트 후 `python3 run.py setup-robolab`을 한 번 실행한다. 기존 5.1 stack은 `python3 run.py patch-client51`을 실행한다. 이후 같은 명령을 반복해도 매번 새 평가를 수행한다.

```bash
python3 run.py client --name v1713_model_s_step500 --result-name repeated_test --num-runs 1
```

로그의 `Archived previous evaluation:`은 이전 결과 보관 위치, `Fresh evaluation output:`은 이번 평가 위치다. 각 폴더의 `evaluation_run.json`에 실행 ID와 이전 경로를 기록한다. `metrics` 명령은 지정한 결과의 점수 재계산·재업로드 전용이며 simulator를 다시 실행하지 않는다.

평가가 정상 완료됐다는 표시와 결과 파일을 확인한 뒤에만 점수를 업로드한다. 성공·실패를 포함한 전체 episode의 유한한 metric 평균, metric별 표본 수, task별·전체 성공률과 평균 score를 기록한다. `experiment/README_exp.md`의 분리·평균 계산과 같은 작업이며 HDF5에서 새 metric을 계산하지 않는다.

Task별 JSONL은 `METRIC_START_NUMBER`(기본 5)부터 RoboLab05, RoboLab06 순서로 저장한다. 로컬 결과는 `<RoboLab>/output/<평가이름>/metrics`, 원격 결과는 모델 저장소의 `evaluations/<평가이름>/`다. HDF5·영상·모델 weight는 점수 업로드에서 제외한다.

점수 업로드만 실패했다면 RoboLab을 다시 실행하지 않고 재전송한다. 자동 평가 모델은 `evaluation_status.json`에서 실제 경로를 읽는다. 수동 Client 결과는 로그에 나온 이름을 지정한다.

```bash
python3 run.py metrics --name v1713_model_s_step500 --result-name 평가결과폴더이름
```

## 3. 평가 데이터 다운로드

동일한 trainset으로 open-loop 분석을 다시 할 때 사용한다. RoboLab closed-loop 평가만 한다면 필요 없다. 다른 소유자는 `--repo-id OWNER/DATASET`을 지정한다.

```bash
python3 run.py download-data
```

## 4. 학습 전후 action distribution 비교

학습 때 자동 생성한 그래프만 다시 만들려면 `--name`으로 모델을 선택하고 실행한다. 같은 관측·seed에서 pretrained와 최종 adapter의 역정규화 action 분포를 비교한다.

```bash
python3 run.py plot-distribution --name v1713_model_s_step500
```
