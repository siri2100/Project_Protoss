# RoboLab: 드라이버 595를 유지하는 실험용 환경

RTX 3090 / Linux 드라이버 595.91.07에서 기존 Isaac Sim 5.1 대신 Isaac Sim 6.0을 사용하는 경로다. 공식 RoboLab 지원 조합이 아니며 **실제 GPU 실행은 아직 검증하지 않았다.** 아래 smoke test 통과 후 평가한다. 드라이버 버전 검사 우회나 호스트 드라이버 변경은 하지 않는다.

고정 구성:

- RoboLab `b6cfcf6ccd2113c4cd1216d35b33024a7d5691da`
- Isaac Lab 3.0 beta2.patch1 `ffff603eafc6b74264a5261cc0183d6a65390d78`
- Isaac Sim `6.0.1.0`, Python 3.12, PyTorch 2.11 / CUDA 12.8
- PhysX backend와 RoboLab의 기존 solver 설정을 명시적으로 적용

[Isaac Lab 호환 버전](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [Isaac Sim 6.0 요구사항](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html), [Isaac Lab 이식 가이드](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/docs/source/migration/migrating_to_isaaclab_3-0.rst)를 기준으로 구성했다. 최신 `main`을 설치하면 다른 Sim/API로 변경될 수 있어 커밋을 고정한다.

## 1. 별도 환경 설치

이 파일과 `Protoss/setup_robolab_595.sh`, `Protoss/src/robolab_595_*.py`, `Protoss/src/robolab_isaac60_compat.py`, `Protoss/src/robolab_gr00t_client.py`, `Protoss/patches/robolab_isaac60.patch`, `Protoss/patches/isaaclab_60_dependencies.patch`가 서버의 프로젝트에 있어야 한다. 기존 `/workspace/RoboLab/.venv-51`과 GR00T 학습·추론 환경은 유지한다. Isaac Sim EULA에 동의한 뒤 실행한다.

```bash
bash /workspace/Project_Protoss/Protoss/setup_robolab_595.sh setup
```

기본 설치 경로는 `/workspace/robolab-595`이며, 하위 `RoboLab/`, `IsaacLab/`, `.venv/`, `pyproject.toml`, `uv.lock`을 만든다. 설치 중 실패하면 같은 명령으로 다시 진행한다. 기존 checkout의 revision이 다르면 중단한다. 다른 경로를 쓰려면 모든 명령에 동일하게 `ROBOLAB_595_ROOT=/workspace/원하는경로`를 지정한다.

공식 RoboLab의 `isaac51` extra는 선택하지 않는다. Lab의 PhysX 실행에 필요한 패키지와 Sim 6.0.1을 하나의 uv 프로젝트에서 해결하여 설치 후 `uv pip check`로 충돌을 검사한다. Lab의 테스트 도구 `coverage` 의존성은 Sim kernel이 요구하는 7.4.4로 맞춘다. Kit 시작 시 별도 RL·Newton·Mimic task suite를 자동 로드하는 항목은 제거하고, RoboLab의 Python task config를 Gym registry에서 직접 읽는다. 수정된 GR00T 클라이언트도 새 checkout에 적용한다. 이후 `uv sync --extra isaac51`을 실행하면 안 된다.

## 2. 모델 서버 없이 물리·카메라 검사

```bash
bash /workspace/Project_Protoss/Protoss/setup_robolab_595.sh smoke
```

BananaOnPlateTask를 환경 1개로 생성하고 초기 팔 자세를 유지하며 16 step 실행한다. PhysX 설정과 RGB 카메라 출력까지 확인한 뒤 `SMOKE PASS`를 출력한다. 이 검사는 작업 성공률이나 학습 모델 품질을 평가하지 않는다.

실패하면 `/workspace/robolab-595/logs/smoke-*.log`에서 처음 발생한 오류와 backtrace를 확인한다. 시작 즉시 native crash가 나면 아직 렌더러/컨테이너 호환 문제가 남은 것이며, `SMOKE PASS` 없이 평가가 가능하다고 판단하지 않는다. 컨테이너에는 NVIDIA graphics/Vulkan 라이브러리도 노출되어 있어야 한다.

## 3. GR00T 평가

[README_test.md](README_test.md)의 모델 서버 명령으로 포트 5557에서 서버를 먼저 실행한다. 이어서:

```bash
bash /workspace/Project_Protoss/Protoss/setup_robolab_595.sh eval \
  --headless --device cuda:0 \
  --remote-host 127.0.0.1 --remote-port 5557 \
  --task BananaOnPlateTask \
  --num-envs 1 --num-runs 1 --open-loop-horizon 8 \
  --instruction-type default --video-mode none \
  --output-folder-name v179_model_s_step500_isaac60
```

실행은 새 `.venv/bin/python`을 직접 사용한다. 시스템 Python이나 `.venv-51`을 참조하지 않는다. 로그는 `logs/eval-*.log`, 평가 결과는 새 RoboLab checkout의 `output/` 아래에 저장된다. 같은 GPU에 모델 서버도 실행한다면 VRAM 사용량을 확인한다.

Sim 버전에 따라 접촉·물리 결과가 달라질 수 있으므로 5.1 결과와 6.0 결과를 직접 비교하지 않는다. baseline과 Model-S를 모두 이 환경에서 다시 평가하고 `uv.lock`과 고정 revision을 함께 보관한다. 기존 환경으로 돌아가려면 README_test.md의 `.venv-51` 명령을 사용한다.
