# Protoss v1.0

추가 학습 없이 GR00T-N1.7-DROID와 GR00T-N1.6-DROID의 최종 action을 혼합한다.

```text
observation → N1.7 VLM/DiT → actions → decode_action → joint/gripper ─┐
                                                                  ├→ alpha 혼합
observation → N1.6 VLM/DiT → actions → decode_action → joint/gripper ─┘
```

`main_v1.0.py`의 `blend_actions()`가 핵심 연산이다.
`alpha * actions17 + (1 - alpha) * actions16`을 공통 채널에 적용한다.
alpha=1은 N1.7, alpha=0은 N1.6이다. 두 모델 모두 추론하며 가중치는 변경하지 않는다.

모델 내부의 정규화된 `actions`/`action_pred` 전체를 그대로 평균하지 않는다.
두 모델의 채널 순서, padding, 정규화 통계, horizon이 다르기 때문이다.
각 서버의 기존 `Gr00tPolicy`가 `model.get_action()`의 최종 `action_pred`를
`decode_action()`으로 물리 단위와 absolute 목표로 복원한 결과를 사용한다.
출력은 `action.joint_position` `[B,H,7]`과 `action.gripper_position` `[B,H,1]`, float32이다.
N1.7의 EEF 채널은 반환하지 않는다. Gripper는 연속 목표값으로 혼합한다.

## 실행

프로젝트 루트 기준으로 아래 각 명령을 별도 터미널에서 실행한다.
두 모델 환경은 해당 저장소의 설치 안내대로 준비해야 한다. CUDA GPU에서 모델 추론을 실행한다.
GPU가 하나이면 두 모델 서버에 `cuda:0`을 지정할 수 있으나 두 모델이 들어갈 메모리가 필요하다.

N1.7 서버:

```bash
cd Issac-GR00T-N1.7
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-DROID \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 --host 127.0.0.1 --port 5557 --use-sim-policy-wrapper
```

N1.6 서버:

```bash
cd Isaac-GR00T-N16
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-DROID \
  --embodiment-tag OXE_DROID \
  --device cuda:1 --host 127.0.0.1 --port 5556 --use-sim-policy-wrapper
```

Protoss 서버（프로젝트 루트에서 실행）:

```bash
python3 -m venv Protoss/.venv
Protoss/.venv/bin/python -m pip install -r Protoss/requirements.txt
Protoss/.venv/bin/python Protoss/main_v1.0.py --alpha 0.5 --horizon 8 --port 5555
```

RoboLab 등 기존 N1.7 형식 클라이언트는 Protoss의 5555 포트에 연결한다.
서버의 `get_modality_config` 응답을 기준으로 관측을 구성하고 실행 horizon은 8 이하로 지정한다.
원격 접속은 `--host 0.0.0.0`으로 바인딩하고, upstream 주소는
`--n17-endpoint tcp://HOST:5557 --n16-endpoint tcp://HOST:5556`으로 지정한다.

## 관측 및 시간 정렬

입력은 N1.7 sim wrapper의 flat observation 형식이다.

| 키 | Shape |
|---|---|
| `video.exterior_image_1_left` | `[B,Tv,H,W,3]`, uint8 |
| `video.wrist_image_left` | `[B,Tv,H,W,3]`, uint8 |
| `state.eef_9d` | `[B,Ts,9]`, float32 |
| `state.joint_position` | `[B,Ts,7]`, float32 |
| `state.gripper_position` | `[B,Ts,1]`, float32 |
| `annotation.language.language_instruction` | batch의 문자열 리스트 |

Tv/Ts와 샘플 시점은 실제 서버의 modality config에서 읽는다. 예를 들어 N1.7의
video delta_indices가 `[-15,0]`이면 실제 과거/현재 영상 두 장이 필요하고,
N1.6이 `[0]`을 요구하면 그중 현재 영상만 전달한다. 누락된 과거 영상을 복제하지 않는다.
로드된 N1.7 checkpoint가 `[0]`만 요구하면 입력 Tv=1을 사용한다.
각 모델에 동일한 현재 state와 instruction을 제공한다.

두 action config의 처음 H개 delta_indices가 같아야 한다. 긴 chunk는 처음 H개만 사용한다.
실제 관측/실행 주기와 로봇 관절 순서 및 단위도 두 서버에서 같아야 한다.
delta_indices 비교만으로 실제 제어 주파수까지 확인할 수는 없다.

## 한 번 추론하고 파일로 저장

위 flat 키들을 가진 NPZ 파일을 `np.savez`로 저장한다. 언어는 object 배열 대신
`np.array(["put the banana on the plate"])` 같은 문자열 배열로 저장한다.
두 upstream 서버를 실행한 상태에서:

```bash
Protoss/.venv/bin/python Protoss/main_v1.0.py \
  --observation observation.npz --output blended_actions.npz --alpha 0.5 --horizon 8
```

v1.0은 일반 action 생성만 지원하며 RTC 옵션은 거부한다. 두 서버를 순서대로 호출하므로
추론 지연은 두 모델의 추론 시간과 통신 시간을 포함한다. timeout이나 잘못된 action은
오류로 반환하며 임의로 다른 모델 결과로 대체하지 않는다.

검증:

```bash
Protoss/.venv/bin/python -m unittest discover -s Protoss -p 'test_*.py' -v
```

단위 테스트는 혼합 수식, 시간 정렬, 두 통신 형식과 오류 처리를 확인한다.
실제 pretrained 모델의 GPU 추론 및 RoboLab 성공률은 별도로 검증해야 한다.
