# Project Protoss v1.2

v1.1의 `Protoss/main_v1.1.py`를 복사하여 `Protoss/main_v1.2.py`에 학습 가능한 post-blending 네트워크를 추가했다. 기존 파일과 두 pretrained checkpoint는 수정하지 않는다.

```text
관측 → GR00T-N1.7-DROID ─┐
                       ├→ absolute action blending → residual flow MLP → 최종 action
관측 → pi0.5-DROID ─────┘                              ↑ 현재 joint/gripper state
```

두 upstream 서버의 설치·checkpoint 다운로드·실행은 [README_v1.1.md](README_v1.1.md)의 1–5절을 따른다. 특히 π 서버는 `pi05_droid_jointpos` 설정과 그 checkpoint를 유지해야 한다. Joint velocity/EEF action checkpoint로 바꾸면 학습 타깃의 의미가 달라진다. 양쪽 서버는 inference만 수행하며 gradient는 추가 MLP에만 흐른다.

## 네트워크와 loss

`a_blend = alpha * a_GR00T + (1-alpha) * a_pi`, 기본 alpha=0.5, H=8, action dimension=8(absolute joint 7 + absolute gripper 1).

MLP는 전체 action chunk를 동시에 입력받는다. 입력은 normalized blended chunk, 현재 normalized state, noisy residual chunk, flow time이며 출력은 residual velocity `[B,H,8]`이다. 이미지와 언어는 upstream VLA를 통해 간접 반영된다. 별도의 visual/language encoder는 추가하지 않았다.

Train target은 실제 demonstration action이다. Train split의 유효 action만으로 차원별 평균/표준편차를 계산하고 checkpoint에 저장한다. Gripper도 연속 position으로 회귀한다.

```text
r = (a_demo - a_blend) / scale
noise ~ N(0,I), t ~ Beta(1.5,1) * 0.999 + 0.001
x_t = t * noise + (1-t) * r
L = masked_mean((v_theta(x_t, a_blend, state, t) - (noise-r))²)
```

이는 [OpenPI π 모델 구현](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)의 conditional flow-matching MSE와 시간 convention(t=1 noise, t=0 data)을 residual에 적용한 것이다. π0.5 원본 네트워크를 그대로 재현한 구조는 아니다. 추론은 Gaussian residual noise에서 Euler 10 step으로 시작하고 `a_out = a_blend + scale * r_hat`을 반환한다. 확률적 추론이므로 같은 관측도 결과가 달라질 수 있다. 마지막 출력에 로봇별 joint/gripper clipping은 적용하지 않는다. 실기 제어기의 기존 제한을 적용해야 한다.

## Trainset 선택

첫 실험은 **DROID에서 수집된 성공 demonstration 30개 규모의 task subset**을 권장한다. Franka joint/gripper, camera 및 언어 조건을 두 pretrained 모델과 맞추기 쉽고, OpenPI 공식 fine-tuning 튜토리얼에도 이 크기의 예시가 있다. 이후 실제 평가 환경의 custom DROID demonstration으로 확대한다. LIBERO는 action 표현과 embodiment가 달라 바로 섞지 않는다.

[OpenPI DROID 학습 가이드](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)는 full DROID에는 RLDS, 소규모 custom 데이터에는 LeRobot을 사용한다. 원본 다운로드 예시는 다음과 같다. 다운로드는 이 작업에서 실행하지 않았다.

```bash
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 data/droid_raw
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json data/
```

**현재 loader는 RLDS/LeRobot 원본을 직접 읽지 않는다.** 아래 명세의 observation+target NPZ로 export한 데이터가 필요하다. 이 export 과정은 데이터 소스별 camera mapping, control frequency, action 의미를 확인해야 하므로 자동 추측하지 않는다. 데이터 다운로드와 원본 exporter는 이번 구현 범위에 포함하지 않았다.

## NPZ 입력 계약

먼저 episode 단위로 train/validation을 분리한다(예: 80/20). 서로 겹치면 trainer가 오류를 낸다. Episode ID는 전체 데이터셋에서 고유해야 한다. 같은 episode의 겹치는 chunk를 서로 다른 split에 넣지 않는다. 기본 H=8이며 episode 끝을 넘는 구간은 finite 값으로 padding하고 mask=0으로 제외한다. 관측 시점 t의 target 첫 action은 action[t]이다. 현재 state[t]를 demonstration action으로 대신 사용하면 안 된다. 원본에 velocity/delta가 있으면 소스의 제어 규약대로 absolute joint command로 변환해야 한다.

각 파일은 observation batch와 이에 대응하는 target을 저장한다. 아래 형식은 B=1도 지원한다. v1.1과 같은 GR00T flat observation 키/시간 인덱스를 사용한다.

| 키 | Shape / 의미 |
|---|---|
| `video.exterior_image_1_left`, `video.wrist_image_left` | RGB uint8 `[B,T_video,H_img,W_img,3]` |
| `state.joint_position` | float32 `[B,T_state,7]` |
| `state.gripper_position` | float32 `[B,T_state,1]` |
| `state.eef_9d` | float32 `[B,T_state,9]`, upstream config가 요구하면 포함 |
| `instruction` | 문자열 `[B]`, 실제 language modality key와 일치 |
| `target.joint_position` | finite float32 `[B,8,7]`, absolute command |
| `target.gripper_position` | finite float32 `[B,8,1]`, 모델 출력과 동일한 unit/convention |
| `mask` | `[B,8]`, 유효 timestep=1, padding=0 |
| `episode_id` | 문자열 또는 정수 `[B]` |

현재 공개 DROID 설정은 보통 현재 영상/상태만 사용한다. 실제 서버의 `get_modality_config`가 기준이다. Target horizon과 15 Hz 시점 정렬을 확인한다. Object array는 사용하지 않는다(`allow_pickle=False`).

## 설치 및 실행

v1.1 proxy 환경에 PyTorch만 추가한다. OpenPI client는 v1.1과 동일한 matching fork에서 설치한다. 아래 명령은 프로젝트 루트 기준이다.

```bash
python -m pip install -r Protoss/requirements_v1.2.txt
```

두 pretrained 서버가 떠 있는 상태에서 train/validation 관측을 각각 캐시한다. 두 모델을 각 관측에 한 번씩 호출하며 학습에서는 다시 호출하지 않는다.

```bash
python Protoss/main_v1.2.py --mode cache \
  --input-dir data/export/train --cache-dir data/cache/train
python Protoss/main_v1.2.py --mode cache \
  --input-dir data/export/val --cache-dir data/cache/val
python Protoss/main_v1.2.py --mode train \
  --train-cache data/cache/train --val-cache data/cache/val \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt \
  --epochs 50 --batch-size 64 --lr 1e-4 --device cpu
```

GPU 환경에서는 `--device cuda`를 사용한다. Train split에서만 통계를 계산하고 validation flow MSE가 가장 낮은 checkpoint를 저장한다. Validation noise/time은 epoch마다 동일하게 유지한다. 학습 재개/optimizer state 저장은 아직 지원하지 않는다. 캐시는 alpha/horizon을 기록하고 trainer가 일치를 확인한다. 모델 weight 또는 preprocessing을 바꾸면 캐시를 새로 생성해야 한다.

추론 서버 및 단일 관측 추론:

```bash
python Protoss/main_v1.2.py --mode infer \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt --port 5555
python Protoss/main_v1.2.py --mode infer \
  --checkpoint Protoss/checkpoints/refiner_v1.2.pt \
  --observation observation.npz --output refined_actions_v1.2.npz
```

학습된 checkpoint가 반드시 필요하다. Alpha와 horizon은 학습 당시와 일치해야 한다. 두 모델 endpoint, timeout, reset, modality config 및 출력 key는 v1.1과 같다. Alpha=0/1은 **blending 단계**에서의 모델 선택 의미이며 이후 residual 보정은 계속 적용된다. RTC는 지원하지 않는다.

## 평가 및 작업 기록

- v1.1을 복사하여 inference transport와 action blending을 유지했다.
- Residual conditional flow MLP, masked loss, offline cache, episode 분리 검사, train-only normalization, best validation checkpoint 및 추론 연결을 추가했다.
- 검증은 CPU 단위 테스트로 수행한다. 실제 pretrained GPU 서버 호출, DROID 원본 export, 실제 데이터 학습 및 로봇 성공률 측정은 별도 환경이 필요하다.
- Flow MSE가 낮아졌다는 사실만으로 robot success 향상을 주장할 수 없다. 동일 평가 episode에서 v1.1 대비 task success, action error, latency 및 stochastic variation을 비교해야 한다.

2026-10-04 검증 결과: 임시 CPU 환경(PyTorch 2.8.0, 기존 matching OpenPI client source)에서 v1.0/v1.1/v1.2 총 26개 테스트 통과. v1.2의 5개 테스트는 cache roundtrip, mask/gradient, 짧은 synthetic training과 checkpoint 추론, 잘못된 mask, blending 이후 적용을 확인한다. Synthetic 학습은 실행 경로 검증이며 실제 성능 측정이 아니다.

```bash
python -m unittest discover -s Protoss -p 'test_main_v1_*.py'
```
