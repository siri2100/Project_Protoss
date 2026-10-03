## 1. GR00T-N1.6 환경 설치

N1.6 모델은 N1.7 코드에서 정상적으로 로드되지 않는다. NVIDIA 공식 Isaac-GR00T 저장소의 최신 N1.6 patch release인 `n1.6.1-release` tag를 `/workspace/Isaac-GR00T-N16`에 설치한다.

```bash
cd /workspace
git clone --recurse-submodules \
    --branch n1.6.1-release \
    https://github.com/NVIDIA/Isaac-GR00T.git \
    Isaac-GR00T-N16

cd /workspace/Isaac-GR00T-N16
git submodule update --init --recursive

apt-get update
apt-get install -y ffmpeg curl git-lfs

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git lfs install

# N1.6은 Python 3.10 환경을 사용한다.
uv sync --python 3.10
uv pip install -e .

uv run python -c "import gr00t; print('GR00T N1.6 installed successfully')"
```

### Hugging Face 로그인

```bash
cd /workspace/Isaac-GR00T-N16

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

uv run hf auth login
uv run hf auth whoami
```

N1.6 DROID 모델 접근을 간단히 검사한다.

```bash
rm -rf /tmp/gr00t_n16_access_test

uv run hf download nvidia/GR00T-N1.6-DROID \
    config.json \
    --local-dir /tmp/gr00t_n16_access_test

test -f /tmp/gr00t_n16_access_test/config.json && echo "N1.6 DROID access OK"
```

## 2. GR00T-N1.6-DROID inference: RoboLab

공개 RoboLab reference 구성은 다음과 같다.

| 설정 | 값 |
|---|---|
| GR00T 코드 | `NVIDIA/Isaac-GR00T`, tag `n1.6.1-release` |
| 모델 | `nvidia/GR00T-N1.6-DROID` |
| Embodiment | `OXE_DROID` |
| Action horizon | 16 |
| RoboLab open loop horizon | 8 |
| 영상 temporal horizon | 1 |

### 2.1 RoboLab 설치

이미 `/workspace/RoboLab`을 설치했다면 이 단계는 생략한다.

```bash
cd /workspace
git clone https://github.com/NVlabs/RoboLab.git

cd /workspace/RoboLab
apt-get update
apt-get install -y ffmpeg

uv venv --python 3.11
source .venv/bin/activate
uv sync --extra isaac50

uv run --extra isaac50 python -c \
    "import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab OK')"
```

### 2.2 RoboLab 클라이언트를 N1.6 서버에 맞게 변경

최신 RoboLab 클라이언트는 N1.7용 `msgpack_numpy` 통신 형식을 사용하지만 N1.6 서버는 NumPy 배열을 `__ndarray_class__`와 `.npy` bytes로 전송하는 형식을 사용한다. N1.6-DROID의 영상 temporal horizon은 `T=1`이다.

아래 명령은 N1.7 클라이언트를 백업하고, 영상 입력을 `T=1`로 복구한 다음 serializer를 N1.6 형식으로 변경한다.

```bash
cd /workspace/RoboLab
cp -n policies/gr00t/client.py policies/gr00t/client.py.n17.bak

python3 - <<'PY'
from pathlib import Path

path = Path("/workspace/RoboLab/policies/gr00t/client.py")
text = path.read_text()

text = text.replace(
    "np.repeat(ext_image[None, None, ...], 2, axis=1).astype(np.uint8)",
    "ext_image[None, None, ...].astype(np.uint8)",
)
text = text.replace(
    "np.repeat(wrist_image[None, None, ...], 2, axis=1).astype(np.uint8)",
    "wrist_image[None, None, ...].astype(np.uint8)",
)

if "import io\n" not in text:
    text = text.replace("import functools\n", "import functools\nimport io\n", 1)

start = text.index("class _MsgSerializer:")
end = text.index("class GR00TPolicyClient:", start)

serializer = '''class _MsgSerializer:
    """Serializer compatible with the GR00T N1.6 server."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(
            data,
            default=_MsgSerializer._encode,
            use_bin_type=True,
        )

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(
            data,
            object_hook=_MsgSerializer._decode,
            raw=False,
        )

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                raise TypeError("Object-dtype arrays are unsupported")
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {
                "__ndarray_class__": True,
                "as_npy": output.getvalue(),
            }
        raise TypeError(f"Unsupported serialization type: {type(obj)!r}")

    @staticmethod
    def _decode(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        marker = obj.get(
            "__ndarray_class__",
            obj.get(b"__ndarray_class__"),
        )
        if marker:
            payload = obj.get("as_npy", obj.get(b"as_npy"))
            return np.load(io.BytesIO(payload), allow_pickle=False)
        return obj


'''

path.write_text(text[:start] + serializer + text[end:])
print("RoboLab client changed to T=1 and the N1.6 wire format")
PY
```

실제 import 경로와 배열 왕복 변환을 검사한다.

```bash
cd /workspace/RoboLab

uv run --extra isaac50 python - <<'PY'
import inspect
import numpy as np
import policies.gr00t.client as client

print("client file:", client.__file__)

source = {
    "video.exterior_image_1_left": np.zeros(
        (1, 1, 180, 320, 3), dtype=np.uint8
    )
}
decoded = client._MsgSerializer.from_bytes(
    client._MsgSerializer.to_bytes(source)
)
video = decoded["video.exterior_image_1_left"]

print("decoded type:", type(video))
print("decoded shape:", video.shape)

assert client.__file__ == "/workspace/RoboLab/policies/gr00t/client.py"
assert isinstance(video, np.ndarray)
assert video.shape == (1, 1, 180, 320, 3)
assert "__ndarray_class__" in inspect.getsource(client._MsgSerializer)
print("N1.6 serializer OK")
PY

cd /workspace/RoboLab
cp policies/gr00t/client.py.n17.bak policies/gr00t/client.py
```

### 2.3 Terminal 1: N1.6 policy server

```bash
cd /workspace/Isaac-GR00T-N16

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-DROID \
    --embodiment-tag OXE_DROID \
    --device cuda \
    --host 127.0.0.1 \
    --port 5555 \
    --use-sim-policy-wrapper

# 2.4 Terminal 2: RoboLab 실행
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
```

먼저 한 task로 smoke test를 실행한다.

```bash
cd /workspace/RoboLab
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
export OMNI_KIT_ACCEPT_EULA=Y

uv run --extra isaac50 python policies/gr00t/run.py \
    --headless \
    --remote-host 127.0.0.1 \
    --remote-port 5555 \
    --task \
        BBQSauceInBinTask \
        BigPumpkinInBinTask \
        BlackItemsInBinTask \
        BlocksInBinTask \
        BlockStackingOrderAgnosticTask \
        BlockStackingSpecifiedOrderTask \
        BowlInBinTask \
        BowlStackingLeftOnRightTask \
        BowlStackingRightOnLeftTask \
        ButterAboveRaisinTask \
    --num-envs 10 \
    --num-runs 1 \
    --open-loop-horizon 8 \
    --instruction-type default \
    --video-mode none \
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

# CannedFoodInBinTask \
# ClampInRightBinTask \
# CleanUpToysTask \
# ClearOrganicObjectsTask \
# ClutterPlasticTask \
# ClutterPumpkinTask \
# CoffeePotInBinTask \
# CondimentsInBinTask \
# CookingClearPlateTask \
# CookingPickPastaToolTask \
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


## 3. GR00T-N1.6 inference: LIBERO

NVIDIA의 N1.6 공식 모델 collection에는 공개 LIBERO checkpoint가 없다. LIBERO inference에는 다음 중 하나가 필요하다.

1. N1.6 기반으로 직접 fine-tuning한 checkpoint
2. N1.6 호환 LIBERO checkpoint를 별도로 내려받은 로컬 경로

N1.7 LIBERO checkpoint는 N1.6 코드와 섞어 사용할 수 없다.

### 3.1 LIBERO 환경 설치

```bash
cd /workspace/Isaac-GR00T-N16

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
    libglfw3 \
    libglu1-mesa

bash gr00t/eval/sim/LIBERO/setup_libero.sh

test -x gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python && \
    echo "LIBERO environment OK"
```

### 3.2 공식 N1.6 방식으로 LIBERO-10 fine-tuning

이미 N1.6 LIBERO checkpoint가 있다면 이 단계는 생략한다. 공식 스크립트는 기본적으로 8 GPU와 20,000 step을 사용하므로 RTX 3090 한 장에서 그대로 실행하기에는 무겁다.

```bash
cd /workspace/Isaac-GR00T-N16

uv run hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --repo-type dataset \
    --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot

cp examples/LIBERO/modality.json \
    examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/modality.json

# 기본 스크립트 출력 경로: /tmp/libero_10
uv run bash examples/LIBERO/finetune_libero_10.sh
```

학습이 끝난 checkpoint 예시:

```text
/tmp/libero_10/checkpoint-20000
```

RunPod의 `/tmp`는 영구 저장소가 아니므로 학습 결과는 `/workspace/checkpoints`로 복사하는 편이 안전하다.

```bash
mkdir -p /workspace/checkpoints/GR00T-N1.6-LIBERO
cp -a /tmp/libero_10/checkpoint-20000 \
    /workspace/checkpoints/GR00T-N1.6-LIBERO/
```

### 3.3 Terminal 1: N1.6 LIBERO policy server

`LIBERO_CHECKPOINT`를 실제 checkpoint 경로로 지정한다.

```bash
cd /workspace/Isaac-GR00T-N16

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export LIBERO_CHECKPOINT=/workspace/checkpoints/GR00T-N1.6-LIBERO/checkpoint-20000

test -f "$LIBERO_CHECKPOINT/config.json"

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/eval/run_gr00t_server.py \
    --model-path "$LIBERO_CHECKPOINT" \
    --embodiment-tag LIBERO_PANDA \
    --device cuda \
    --host 127.0.0.1 \
    --port 5555 \
    --use-sim-policy-wrapper
```

### 3.4 Terminal 2: LIBERO rollout

N1.6 LIBERO 스크립트는 underscore 형식의 CLI 옵션을 사용한다.

```bash
cd /workspace/Isaac-GR00T-N16

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 720 \
    --env_name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n_action_steps 8 \
    --n_envs 1 \
    --video_dir /workspace/libero_rollouts/GR00T-N1.6
```

설치된 코드에 따라 `--video_dir`가 지원되지 않을 수 있다. 다음 명령으로 지원 옵션을 확인하고, 표시되지 않으면 `--video_dir` 줄만 제거한다.

```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py --help
```

### LIBERO-10 task 목록

```text
libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
```

## 4. 자주 발생하는 오류

### `KeyError: 'Gr00tN1d6'`

N1.7 저장소에서 N1.6 checkpoint를 불러온 경우다. 서버를 `/workspace/Isaac-GR00T-N16`에서 실행한다.

### `Video key ... horizon must be 2. Got 1`

N1.6-DROID 서버가 아니라 N1.7-3B 또는 다른 processor 설정의 서버에 연결했을 가능성이 크다. Terminal 1의 모델과 embodiment가 다음인지 확인한다.

```text
Model: nvidia/GR00T-N1.6-DROID
Embodiment: OXE_DROID
```

RoboLab 클라이언트도 `np.repeat(..., 2, axis=1)` 수정이 제거된 T=1 상태여야 한다.

### `Video key ... must be a numpy array. Got <class 'dict'>`

최신 RoboLab 클라이언트가 N1.7용 `msgpack_numpy` 형식으로 배열을 보냈지만 N1.6 서버가 이를 해제하지 못한 경우다. 2.2절의 N1.6 serializer 변경과 자체 검사를 실행하고, 실행 중인 RoboLab 프로세스를 완전히 종료한 뒤 다시 시작한다. 서버는 계속 실행 중이어도 된다.

### `Model path ... does not exist`

LIBERO는 예시 경로를 그대로 사용하지 말고 실제 checkpoint 위치를 확인한다.

```bash
find /workspace/checkpoints -type f -name config.json -print
```

### Port 5555가 이미 사용 중인 경우

두 실험을 동시에 실행할 때 서버와 클라이언트에 같은 새 포트를 지정한다.

```text
실험 A: server 5555 / client 5555
실험 B: server 5556 / client 5556
```

## 참고 자료

- [NVIDIA 공식 Isaac-GR00T 저장소](https://github.com/NVIDIA/Isaac-GR00T)
- [NVIDIA 공식 N1.6.1 release source](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6.1-release)
- [GR00T N1.6 model collection](https://huggingface.co/collections/nvidia/gr00t-n16)
- [GR00T N1.7 RoboLab guide의 N1.6 reference](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/RoboLab/README.md#n16-reference)
