# GR00T N1.7 RunPod setup

프로젝트를 submodule까지 내려받습니다.

```bash
git clone --recurse-submodules https://github.com/siri2100/Project_Protoss.git
cd Project_Protoss
```

이미 clone한 저장소라면 다음 명령을 한 번 실행합니다.

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

RunPod GPU Pod에서 환경을 설치합니다.

```bash
chmod +x gr00t_n1_7_setup/*.sh
./gr00t_n1_7_setup/setup_runpod.sh
```

<https://huggingface.co/nvidia/Cosmos-Reason2-2B>의 접근 승인을 받은 다음 로그인하고
pretrained zero shot inference를 실행합니다.

```bash
cd GR00T-N1.7
source .venv/bin/activate
source .env.runpod
hf auth login
cd ..
./gr00t_n1_7_setup/run_pretrained_inference.sh
```

모델 및 uv 캐시는 `/workspace/.cache`에 저장되어 영구 Volume에서 재사용됩니다.

## Git LFS wheel 오류

`torchcodec` wheel에서 `Invalid zip file structure`가 발생하면 wheel이 실제 파일이
아니라 Git LFS 포인터로 남아 있는 것입니다. 다음 명령 후 설치를 다시 실행합니다.

```bash
git -C GR00T-N1.7 lfs pull --include="scripts/deployment/dgpu/wheels/**"
cd GR00T-N1.7
uv sync --python 3.12
```
