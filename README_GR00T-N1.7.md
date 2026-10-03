# Project Protoss
container dist : 200GB

# GR00T-N1.7 환경세팅 방법
```bash
# Step 1. Set Environment 
cd /workspace/Project_Protoss
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T Isaac-GR00T-N17
cd /workspace/Project_Protoss/Isaac-GR00T-N17
git submodule update --init --recursive

apt-get update
apt-get install -y ffmpeg curl git-lfs

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git lfs install
git lfs pull --include="scripts/deployment/dgpu/wheels/**"
git lfs pull --include="demo_data/droid_sample/**"
git lfs pull --include="demo_data/libero_demo/**"

uv sync --python 3.12
uv run python -c "import gr00t; print('GR00T installed successfully')"

# Step 2. Download Model (Huggingface)
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

# 먼저 로그인에 사용할 Hugging Face 계정으로 아래 gated model 접근을 신청하고
# 승인된 상태인지 웹페이지에서 확인한다.
# https://huggingface.co/nvidia/Cosmos-Reason2-2B

# 기존에 비어 있거나 잘못 지정된 환경변수가 저장된 token을 가리지 않도록 제거한다.
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN
unset HF_HUB_DISABLE_IMPLICIT_TOKEN

# RunPod 터미널에서 token을 화면에 노출하지 않고 입력한다.
read -rsp "Hugging Face token 입력: " HF_LOGIN_TOKEN
echo

test -n "$HF_LOGIN_TOKEN" || { echo "Token이 입력되지 않았습니다."; exit 1; }

# hf CLI 버전에 따라 --no-add-to-git-credential 옵션이 없을 수 있으므로
# Python API로 로그인한다. Git credential 저장은 필요하지 않다.
HF_LOGIN_TOKEN="$HF_LOGIN_TOKEN" uv run python - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_LOGIN_TOKEN"], add_to_git_credential=False)
PY

LOGIN_STATUS=$?

unset HF_LOGIN_TOKEN

test "$LOGIN_STATUS" -eq 0 || { echo "Hugging Face 로그인 실패"; exit "$LOGIN_STATUS"; }

# 반드시 로그인한 사용자명이 출력되어야 한다.
uv run hf auth whoami || exit 1

# COSMOS 접근 확인
rm -rf /tmp/cosmos_access_test

uv run hf download nvidia/Cosmos-Reason2-2B \
  config.json \
  --local-dir /tmp/cosmos_access_test || exit 1

test -f /tmp/cosmos_access_test/config.json || exit 1
echo "Cosmos access OK"

cd /workspace/Project_Protoss/Isaac-GR00T-N17

uv run hf download nvidia/GR00T-N1.7-LIBERO \
  --include \
    "libero_10/config.json" \
    "libero_10/embodiment_id.json" \
    "libero_10/model-*.safetensors" \
    "libero_10/model.safetensors.index.json" \
    "libero_10/processor_config.json" \
    "libero_10/statistics.json" \
  --local-dir checkpoints/GR00T-N1.7-LIBERO
```

# GR00T-N1.7 inference (RoboLab)

> **RunPod 드라이버 확인:** RoboLab의 Isaac Sim 5.0/5.1은 일부 Linux 595 드라이버
> (확인된 실패 예: 595.84, 595.91.07)에서 `librtx.scenedb.plugin.so` 크래시가 발생한다.
> Pod를 만들고 가장 먼저 `nvidia-smi`를 실행한다. 이 증상이 있으면 Pod 내부에서
> CUDA나 Python 패키지를 다시 설치하지 말고, NVIDIA가 Isaac Sim 5.1을 테스트한
> 580 계열 드라이버 호스트(기준 버전 580.65.06)로 옮긴다.

```bash
cd /workspace
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
apt-get update
# Isaac Sim 5.1 환경을 별도 디렉터리에 설치한다.
# 기존 Isaac Sim 5.0의 .venv와 섞지 않는다.
deactivate 2>/dev/null || true
uv venv --clear --python 3.11 .venv-51
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --python 3.11 --extra isaac51
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --extra isaac51 python -c \
    "import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab OK')"

# 1st terminal (GR00T-N1.7-DROID)
cd /workspace/Project_Protoss/Isaac-GR00T-N17
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

# 서버 실행 전에 로그인과 gated backbone 접근을 다시 확인한다.
uv run hf auth whoami
test -f /tmp/cosmos_access_test/config.json || \
  uv run hf download nvidia/Cosmos-Reason2-2B \
    config.json \
    --local-dir /tmp/cosmos_access_test

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-DROID \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda \
    --host 127.0.0.1 \
    --port 5555 \
    --use-sim-policy-wrapper

# 2nd terminal 
apt-get update
apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    libglvnd0 \
    libgl1 \
    libegl1 \
    libglu1-mesa \
    libxt6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxkbcommon0
ldconfig

cd /workspace/Project_Protoss/RoboLab
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
test -x .venv-51/bin/python || uv venv --python 3.11 .venv-51
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --python 3.11 --extra isaac51
export OMNI_KIT_ACCEPT_EULA=Y

# CUDA toolkit의 stub libcuda가 실제 NVIDIA driver보다 먼저 로드되면
# Warp에서 cuDeviceGetUuid를 찾지 못하고 Isaac Sim이 종료된다.
unset LD_PRELOAD
unset LD_LIBRARY_PATH

# 실행 전에 실제 driver library와 필수 symbol을 확인한다.
nvidia-smi
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --extra isaac51 python -c \
    "import ctypes; lib = ctypes.CDLL('libcuda.so.1'); getattr(lib, 'cuDeviceGetUuid'); print('libcuda OK:', lib._name)"

# 처음에는 1개 환경으로 정상 기동을 확인한 뒤 5로 늘린다.
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --extra isaac51 python policies/gr00t/run.py \
    --headless \
    --remote-host 127.0.0.1 \
    --remote-port 5555 \
    --task \
        BBQSauceInBinTask \
        BigPumpkinInBinTask \
        BlackItemsInBinTask \
        BlocksInBinTask \
        BlockStackingOrderAgnosticTask \
    --num-envs 10 \
    --num-runs 1 \
    --open-loop-horizon 8 \
    --instruction-type default \
    --video-mode sensor \
    --enable-subtask

# 01. AnimalsInBinTask \
# 02. AppleAndYogurtInBowlTask \
# 03. BagelsOnPlateTask \
# 04. BananaInBowlTask \
# 05. BananaOnPlateTask \
# 06. BananasInBinOneMoreTask \
# 07. BananasInBinThreeTotalTask \
# 08. BananasInCrateTask \
# 09. BananasOutOfBinTask \
# 10. BananaThenRubiksCubeTask \

# 11. BBQSauceInBinTask \
# 12. BigPumpkinInBinTask \
# 13. BlackItemsInBinTask \
# 14. BlocksInBinTask \
# 15. BlockStackingOrderAgnosticTask \
# 16. BlockStackingSpecifiedOrderTask \
# 17. BowlInBinTask \
# 18. BowlStackingLeftOnRightTask \
# 19. BowlStackingRightOnLeftTask \
# 20. ButterAboveRaisinTask \

# 21. CannedFoodInBinTask \
# 22. ClampInRightBinTask \
# 23. CleanUpToysTask \
# 24. ClearOrganicObjectsTask \
# 25. ClutterPlasticTask \
# 26. ClutterPumpkinTask \
# 27. CoffeePotInBinTask \
# 28. CondimentsInBinTask \
# 29. CookingClearPlateTask \
# 30. CookingPickPastaToolTask \

# CubesAndBlocksInBinTask \
# DishesInBinTask \
# ElectronicsInBinTask \
# FoodPacking1BoxesTask \
# FoodPacking1CansTask \
# FoodPacking2BoxesTask \
# FoodPacking2CansTask \
# FoodPacking3BoxesTask \
# FoodPacking3CansTask \
# FoodPackingByColorTask \
# FruitsGreenLimesOnPlateTask \
# FruitsMovingOrangeOrLimeTask \
# FruitsMovingTask \
# FruitsOnionTask \
# FruitsOnionToPlateTask \
# FruitsOnPlate3Task \
# FruitsOnPlateTask \
# FruitsOrangesOnPlateTask \
# GrabABagelTask \
# GrabAFruitTask \
# GreenSpoonsInPotTask \
# HammersInLeftBinTask \
# JugsOnShelfTask \
# KeyboardOutOfBinTask \
# LargerObjectRaisinBoxInBinTask \
# MarkerInMugTask \
# MouseOnKeyboardTask \
# MoveBananaToBagelPlateTask \
# MustardAboveRaisinTask \
# MustardInLeftBinTask \
# MustardInRightBinTask \
# NonHammerToolsInRightBinTask \
# OneBottleInSquarePailTask \
# OneBottleOnShelfTask \
# PhoneOrRemoteInBinTask \
# PickDrillTask \
# PickGlassesTask \
# PickOrangeObjectTask \
# PickUpBluePitcherTask \
# PickUpGreenObjectTask \
# PinkSpoonInPotTask \
# PlasticBottlesInSquarePailTask \
# PutBowlOnShelfTopTask \
# PutMugsOnShelfTask \
# PutTwoMugsOnShelfTask \
# RecycleCartonsOnBoxTask \
# RecycleCartonsVerticalCrateTask \
# RecycleCartonTask \
# RedDishesInBinTask \
# RedItemsInBinTask \
# ReorientAllMugsTask \
# ReorientJugTask \
# ReorientRedMugTask \
# ReorientWhiteMugsTask \
# RubiksCubeAndBananaTask \
# RubiksCubeBehindBowlTask \
# RubiksCubeInFrontOfBowlTask \
# RubiksCubeLeftOfBowlTask \
# RubiksCubeOrBananaTask \
# RubiksCubeRightOfBowlTask \
# RubiksCubesInBinTask \
# RubiksCubeTask \
# RubiksCubeThenBananaTask \
# SauceBottlesCrateTask \
# SmallerObjectButterInBinTask \
# SmallPumpkinInBinTask \
# SmartphoneInBinTask \
# SpoonInMugTask \
# SpoonsInPotTask \
# Stack3RubiksCubeTask \
# StackWhiteMugsTask \
# StackYellowOnRedTask \
# TakeMeasuringSpoonOutTask \
# TakeMugsOffOfShelfTask \
# TakeSpatulaOffShelfTask \
# ThrowAwayAppleTask \
# ThrowAwaySnacksTask \
# ToolOrganizationBothTask \
# ToolOrganizationTask \
# ToolsPickingAllHammersTask \
# ToolsPickingDrillTask \
# ToolsPickingHammerTask \
# ToyInBinTask \
# UnstackRubiksCubeTask \
# UtensilsInMugTask \
# WhiteMugInCenterOfTableTask \
# WhiteMugsInBinTask \
# WoodSpatulaToBowlTask \
# YellowAndWhiteObjectsInBinTask \
# YogurtInBowlTask \
```

# GR00T-N1.7 Inference (LIBERO)
```bash
# 1st terminal
cd /workspace/Isaac-GR00T-N17
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
uv run hf auth whoami
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper

# 2nd terminal
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
apt-get update
apt-get install -y \
    libegl1 \
    libegl-dev \
    libgl1 \
    libglvnd0 \
    libgles2 \
    libopengl0 \
    libosmesa6 \
    libglfw3
cd /workspace/Project_Protoss/Isaac-GR00T-N17
bash gr00t/eval/sim/LIBERO/setup_libero.sh

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it \
    --n-action-steps 8 \
    --n-envs 1 \
    --video-dir /workspace/Project_Protoss/libero_rollouts/LIBERO10

# 01. libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
# 02. libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
# 03. libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
# 04. libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
# 05 .libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
# 06. libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
# 07. libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
# 08. libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
# 09. libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
# 10. libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it

```

# 문제 해결

## `librtx.scenedb.plugin.so` crash (driver 595.84/595.91)

다음 조건이 모두 보이면 RoboLab 코드나 Python 환경 문제가 아니라 RunPod 호스트
드라이버와 Isaac Sim RTX renderer의 호환 문제다.

- `isaacsim==5.1.0.0`, `isaaclab==2.3.2.post1` 설치 성공
- 실행 경로가 `.venv-51` 및 `Isaac-Sim/5.1`
- `nvidia-smi`의 드라이버가 `595.84` 또는 `595.91` 계열
- backtrace가 `librtx.scenedb.plugin.so`에서 종료

현재 Pod 안에서 NVIDIA 호스트 드라이버를 변경할 수 없다. Network Volume의
`/workspace`는 유지하고 Pod만 종료한 뒤, 580 계열 드라이버 호스트에서 새 Pod를
연결한다. 새 Pod에서 `nvidia-smi`로 드라이버를 확인한 다음 기존 `.venv-51` 환경을
사용한다.

## `401 Unauthorized` 또는 `GatedRepoError: nvidia/Cosmos-Reason2-2B`

GR00T-N1.7 checkpoint 다운로드가 시작되더라도 VLM backbone인 `nvidia/Cosmos-Reason2-2B` 접근 권한이 없으면 모델 로딩은 실패한다. GR00T를 다시 설치할 필요는 없다.

1. 브라우저에서 [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)에 접속한다.
2. 모델 접근을 신청하고 승인을 확인한다.
3. 접근 승인을 받은 것과 동일한 Hugging Face 계정의 token으로 로그인한다.
4. 다음 두 명령이 모두 성공한 다음 서버를 실행한다.

```bash
cd /workspace/Project_Protoss/Isaac-GR00T-N17
export HF_HOME=/workspace/.cache/huggingface
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN

uv run hf auth whoami
uv run hf download nvidia/Cosmos-Reason2-2B \
    config.json \
    --local-dir /tmp/cosmos_access_test
```

## `LocalTokenNotFoundError`

대화형 로그인에서 token이 빈 값으로 전달된 경우다. 위 환경 설정의 `read -rsp` 방식으로 token을 다시 입력한다. RunPod 웹 터미널에서는 `Ctrl+Shift+V` 또는 마우스 오른쪽 클릭으로 붙여넣는다. 입력 중 문자가 보이지 않는 것은 정상이다.

## `syntax error near unexpected token 'Isaac Lab OK'`

`python -c` 뒤의 Python 코드를 따옴표로 묶지 않은 경우다. 다음처럼 실행한다.

```bash
UV_PROJECT_ENVIRONMENT=.venv-51 uv run --extra isaac51 python -c \
    "import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab OK')"
```

## `VIRTUAL_ENV=/workspace/Project_Protoss/RoboLab/.venv does not match ...`

RoboLab 가상환경이 활성화된 터미널에서 Isaac-GR00T 명령을 실행했을 때 나오는 경고다. `uv`는 Isaac-GR00T의 `.venv`를 사용하므로 직접적인 실패 원인은 아니지만, 환경 혼동을 피하려면 GR00T 서버 터미널에서 먼저 실행한다.

```bash
deactivate 2>/dev/null || true
cd /workspace/Project_Protoss/Isaac-GR00T-N17
```

## CUDA 저장소의 `legacy trusted.gpg` 경고

APT key 저장 방식에 관한 경고이며 이번 설치 실패 원인이 아니다. 패키지 설치가 정상 완료되었다면 무시해도 된다.
