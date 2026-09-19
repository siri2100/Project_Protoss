# ACG Docker 환경

ACG의 설치 문서에 맞춰 Python 3.10, CUDA 12.8, PyTorch 2.7.1, FlashAttention
2.8.3, RoboCasa, RoboSuite, Robomimic, DexMimicGen을 설치합니다. 컨테이너는
Linux x86-64와 NVIDIA GPU를 대상으로 합니다. 호스트에 Docker Compose와
NVIDIA Container Toolkit이 필요합니다. macOS에서는 CUDA GPU를 컨테이너에
전달할 수 없으므로 NVIDIA GPU가 있는 Linux 서버에서 실행하세요.

프로젝트 루트에서 설치 및 실행 스크립트를 사용합니다.

```bash
./run_docker.sh
./run_docker.sh check
./run_docker.sh shell
```

인자 없이 실행하면 이미지를 빌드하고 `acg` 컨테이너를 생성·시작합니다.
컨테이너는 백그라운드에서 유지되며 `shell`로 접속할 수 있습니다.
이미지만 만들려면 `./run_docker.sh build`, 컨테이너를 멈추려면
`./run_docker.sh stop`을 실행하세요. `check`는 GPU 접근이 안 되면 오류로
종료합니다.

이미지 빌드 시 핵심 패키지 import를 확인합니다. 위 두 번째 명령에서 `True`가
나오면 컨테이너가 GPU를 사용할 수 있습니다. 첫 빌드는 CUDA 기반 이미지와
FlashAttention 컴파일 때문에 오래 걸리고 저장 공간도 많이 필요합니다.

Compose는 로컬 `ACG/`를 컨테이너의 `/workspace/ACG`에 연결합니다. 따라서
소스 수정과 아래 RoboCasa 에셋 다운로드가 호스트에 남습니다. 데이터셋은
로컬 `datasets/`를 컨테이너의 `/root/datasets/robot`에 연결합니다. ACG의
기본 Robomimic 설정 파일이 참조하는 `~/datasets/robot/...` 경로와 맞습니다.
학습 결과는 로컬 `outputs/`에 남습니다.

RoboCasa 환경을 사용하는 경우 에셋을 별도로 내려받습니다(약 5 GB).

```bash
./run_docker.sh assets
```

학습 데이터셋과 모델 체크포인트는 이미지에 포함되지 않습니다. ACG의
`README.md`와 `scripts/Training_Rollout_Guideline.md`에 따라 데이터셋을
`datasets/robocasa` 또는 `datasets/dexmimicgen`에 준비하고, 필요한 경우
`ACG/libs/Isaac-GR00T-N1/robomimic_configs/`의 경로를 수정하세요.

RoboCasa 소스는 NumPy 1.23.5와 MuJoCo 3.2.6을 import 시 검사합니다.
Robomimic과 RoboCasa의 오래된 PyTorch 및 NumPy 의존성은 GR00T 설치와
충돌하므로 해당 패키지는 의존성을 자동 설치하지 않고, 호환되는 버전을
Dockerfile에서 명시적으로 설치합니다.
