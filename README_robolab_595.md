# RoboLab: 드라이버 595를 유지하는 실험용 환경

Linux 드라이버 595.91.07에서 기존 Isaac Sim 5.1 대신 Isaac Sim 6.0을 사용하는 경로다. 공식 RoboLab 지원 조합이 아니며 **실제 GPU 실행은 아직 검증하지 않았다.** 아래 smoke test 통과 후 평가한다. 드라이버 버전 검사 우회나 호스트 드라이버 변경은 하지 않는다.

고정 구성:

- RoboLab `b6cfcf6ccd2113c4cd1216d35b33024a7d5691da`
- Isaac Lab 3.0 beta2.patch1 `ffff603eafc6b74264a5261cc0183d6a65390d78`
- Isaac Sim `6.0.1.0`, Python 3.12, PyTorch 2.11 / CUDA 12.8
- PhysX backend와 RoboLab의 기존 solver 설정을 명시적으로 적용

[Isaac Lab 호환 버전](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [Isaac Sim 6.0 요구사항](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html), [Isaac Lab 이식 가이드](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/docs/source/migration/migrating_to_isaaclab_3-0.rst)를 기준으로 구성했다. 최신 `main`을 설치하면 다른 Sim/API로 변경될 수 있어 커밋을 고정한다.

## 1. RunPod의 Project_Protoss2에서 디버깅

이 문서의 디버깅 코드는 RunPod의 `/workspace/Project_Protoss2`에서 실행한다. 로컬에 별도 Project_Protoss2 폴더를 만들지 않는다. 업데이트한 프로젝트 파일을 원격 경로에 반영한 뒤 아래 진입점을 사용한다. 기존 simulator stack `/workspace/robolab-595`를 이어 사용하며, `PROTOSS_DEBUG_ROOT`로 디버깅 checkout 위치를 바꿀 수 있다.

이 파일과 `Protoss/debug_robolab_595.sh`, `Protoss/setup_robolab_595.sh`, `Protoss/src/robolab_595_*.py`, `Protoss/src/robolab_isaac60_compat.py`, `Protoss/src/robolab_gr00t_client.py`, `Protoss/patches/robolab_isaac60.patch`, `Protoss/patches/robolab_isaac60_api.patch`, `Protoss/patches/isaaclab_60_dependencies.patch`, `Protoss/src/isaaclab_optional_newton.py`가 서버의 프로젝트에 있어야 한다. 기존 `/workspace/RoboLab/.venv-51`과 GR00T 학습·추론 환경은 유지한다. Isaac Sim EULA에 동의한 뒤 실행한다.

```bash
bash /workspace/Project_Protoss2/Protoss/debug_robolab_595.sh check
```

`check`는 setup 성공 후에만 smoke를 실행한다. setup이 실패하면 즉시 중단하므로, 적용되지 않은 패치로 GPU smoke를 반복하지 않는다. `setup`, `smoke`, `eval`도 별도로 지정할 수 있다. 이 진입점은 Project_Protoss2 폴더를 생성하거나 학습을 실행하지 않는다.

기본 설치 경로는 `/workspace/robolab-595`이며, 하위 `RoboLab/`, `IsaacLab/`, `.venv/`, `pyproject.toml`, `uv.lock`을 만든다. 설치 중 실패하면 같은 명령으로 다시 진행한다. 기존 checkout의 revision이 다르면 중단한다. 다른 경로를 쓰려면 모든 명령에 동일하게 `ROBOLAB_595_ROOT=/workspace/원하는경로`를 지정한다.

공식 RoboLab의 `isaac51` extra는 선택하지 않는다. Lab의 PhysX 실행에 필요한 패키지와 Sim 6.0.1을 하나의 uv 프로젝트에서 해결하여 설치 후 `uv pip check`로 충돌을 검사한다. Lab의 테스트 도구 `coverage` 의존성은 Sim kernel이 요구하는 7.4.4로 맞춘다. Kit 시작 시 별도 RL·Newton·Mimic task suite를 자동 로드하는 항목은 제거하고, RoboLab의 Python task config를 Gym registry에서 직접 읽는다. 수정된 GR00T 클라이언트도 새 checkout에 적용한다. 이후 `uv sync --extra isaac51`을 실행하면 안 된다.

## 2. 모델 서버 없이 물리·카메라 검사

이전에 `isaaclab_60_optional_newton.patch does not apply`로 중단된 문제는 `isaaclab_optional_newton.py`로 수정한다. 주변 import·공백에 의존하지 않고 `_HAS_NEWTON_ACTUATORS` 대입문을 AST로 확인해 상위 패키지부터 검사하도록 바꾼다. 원본은 `articulation.py.before_protoss_optional_newton`에 보관하며, 이미 수정된 대입문은 그대로 둔다. 예상과 다른 대입문은 덮어쓰지 않고 중단한다. smoke/eval도 GPU 초기화 전에 이 수정의 적용 여부를 확인한다.

`Original error: No module named 'isaaclab_newton'`는 선택 기능 검사에서 발생한다. Newton이 없으면 해당 기능을 끄고 기존 PhysX actuator 경로를 사용하게 한다. 설치된 Newton 패키지 내부의 다른 오류는 숨기지 않는다.

`ModuleNotFoundError: No module named 'isaacsim.core.prims'`는 RoboLab이 옛 `XFormPrim` API를 import해서 발생한다. 수정된 setup은 평가 경로에서 `BaseFrameView` 타입 검사와 `isaaclab.sim.utils.stage`를 사용하도록 추가 API 패치를 적용한다. 기존 설치도 `setup`을 다시 실행하면 업데이트된다. 단독 USD 미리보기 도구(`render_utils.py`)의 옛 World/Camera API는 이 평가 경로의 이식 범위에 포함하지 않는다.

`ModuleNotFoundError: No module named 'isaaclab_contrib'`가 발생한 기존 설치는 수정된 프로젝트 파일을 반영한 뒤 `setup`을 다시 실행한다. DROID가 촉각 센서를 사용하지 않더라도 Isaac Lab의 `InteractiveScene`이 해당 config를 import하므로 `isaaclab-contrib`가 필요하다. 새 setup은 같은 Isaac Lab checkout에서 이 패키지도 설치한다.

```bash
bash /workspace/Project_Protoss2/Protoss/debug_robolab_595.sh smoke
```

BananaOnPlateTask를 환경 1개로 생성하고 초기 팔 자세를 유지하며 16 step 실행한다. PhysX 설정과 RGB 카메라 출력까지 확인한 뒤 `SMOKE PASS`를 출력한다. 이 검사는 작업 성공률이나 학습 모델 품질을 평가하지 않는다.

실패하면 `/workspace/robolab-595/logs/smoke-*.log`에서 처음 발생한 오류와 backtrace를 확인한다. 시작 즉시 native crash가 나면 아직 렌더러/컨테이너 호환 문제가 남은 것이며, `SMOKE PASS` 없이 평가가 가능하다고 판단하지 않는다. 컨테이너에는 NVIDIA graphics/Vulkan 라이브러리도 노출되어 있어야 한다.

Kit 종료 처리 때문에 프로세스가 0으로 끝나더라도 `SMOKE PASS`가 출력되지 않으면 실행 스크립트는 실패로 종료한다.

`SimulationContext cleared` 다음에 traceback 없이 `Smoke did not complete`만 보이면 smoke 스크립트도 최신 파일로 반영한다. 실패 시 `app.close()`보다 먼저 `SMOKE FAILED at stage: ...`와 원래 traceback을 출력하도록 수정되어 있다. 로그의 `environment_creation`, `reset`, `stepping`, `camera_validation` 단계와 traceback으로 실제 실패 원인을 확인한다.

## 3. GR00T 평가

[README_test.md](README_test.md)의 모델 서버 명령으로 포트 5557에서 서버를 먼저 실행한다. 이어서:

```bash
bash /workspace/Project_Protoss2/Protoss/debug_robolab_595.sh eval \
  --headless --device cuda:0 \
  --remote-host 127.0.0.1 --remote-port 5557 \
  --task BananaOnPlateTask \
  --num-envs 1 --num-runs 1 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name v179_model_s_step500_isaac60
```

실행은 새 `.venv/bin/python`을 직접 사용한다. 시스템 Python이나 `.venv-51`을 참조하지 않는다. 로그는 `logs/eval-*.log`, 평가 결과는 새 RoboLab checkout의 `output/` 아래에 저장된다. 같은 GPU에 모델 서버도 실행한다면 VRAM 사용량을 확인한다.

`episode_results.jsonl`이 없다는 오류는 후처리 단계에서 평가 결과를 찾지 못했다는 뜻이다. 파일은 첫 episode의 결과를 기록한 뒤 생기므로, 평가가 끝까지 실행됐는지와 저장 경로를 먼저 확인한다. 595 실행 스크립트는 실제 Python/Sim 버전, RoboLab 소스 경로, 로그 경로를 출력하고, 이름을 지정한 평가에서 결과 파일이 없거나 비어 있으면 실패로 종료한다. 이 경우 metric을 계산하거나 업로드하지 않는다.

아래 명령으로 저장된 결과와 평가 로그를 찾는다. 다른 실험 결과를 현재 결과로 대신 사용하지 말고, `NAME`과 로그에 출력된 결과 폴더가 일치하는지 확인한다.

```bash
find /workspace/robolab-595/RoboLab/output /workspace/RoboLab/output \
  -maxdepth 2 -name episode_results.jsonl -print 2>/dev/null
ls -t /workspace/robolab-595/logs/eval-*.log
```

가장 최근 평가 로그의 `RoboLab runtime`, `RoboLab source`, output directory, 최초 오류와 마지막 120줄을 확인한다. 평가가 다른 경로에 정상 저장됐다면 `process_robolab_results.py --results-dir`에 그 실험 폴더를 지정해 후처리만 다시 실행할 수 있다.

Sim 버전에 따라 접촉·물리 결과가 달라질 수 있으므로 5.1 결과와 6.0 결과를 직접 비교하지 않는다. baseline과 Model-S를 모두 이 환경에서 다시 평가하고 `uv.lock`과 고정 revision을 함께 보관한다. 기존 환경으로 돌아가려면 README_test.md의 `.venv-51` 명령을 사용한다.
