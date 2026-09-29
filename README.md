# Project Protoss
container dist : 200GB

# GR00T-N1.7 환경세팅 방법
```bash
# Step 1. Set Environment 
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get update && sudo apt-get install -y ffmpeg
apt-get install -y git-lfs
git lfs install
git lfs pull --include="scripts/deployment/dgpu/wheels/**"

uv sync --python 3.12
uv run python -c "import gr00t; print('GR00T installed successfully')"

# Step 2. Download Dataset(DROID)
git lfs pull --include="demo_data/droid_sample/**"
git lfs pull --include="demo_data/libero_demo/**"
uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/config.json" "libero_10/embodiment_id.json" \
    "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
    "libero_10/processor_config.json" "libero_10/statistics.json" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO

# Step 3. Download Model
unset HF_TOKEN
uv run hf auth logout
uv run hf auth login # input my huggingface access token
```

# GR00T-N1.7 inference (RoboLab)
```bash
cd /workspace
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
apt-get update
apt-get install -y ffmpeg
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run --extra isaac50 python -c import isaaclab; from isaaclab.app import AppLauncher; print('Isaac Lab OK')


# 1st terminal
cd /workspace/Isaac-GR00T
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0
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

cd /workspace/RoboLab
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
uv sync --extra isaac50
export OMNI_KIT_ACCEPT_EULA=Y
uv run --extra isaac50 python policies/gr00t/run.py \
    --headless \
    --remote-host 127.0.0.1 \
    --remote-port 5555 \
    --task BagelsOnPlateTask \
    --num-envs 1 \
    --num-runs 1 \
    --open-loop-horizon 8 \
    --instruction-type default \
    --video-mode all \
    --enable-subtask

# AnimalsInBinTask \
# AppleAndYogurtInBowlTask \
# BagelsOnPlateTask \
# BananaInBowlTask \
# BananaOnPlateTask \
# BananasInBinOneMoreTask \
# BananasInBinThreeTotalTask \
# BananasInCrateTask \
# BananasOutOfBinTask \
# BananaThenRubiksCubeTask \
# BBQSauceInBinTask \
# BigPumpkinInBinTask \
# BlackItemsInBinTask \
# BlocksInBinTask \
# BlockStackingOrderAgnosticTask \
# BlockStackingSpecifiedOrderTask \
# BowlInBinTask \
# BowlStackingLeftOnRightTask \
# BowlStackingRightOnLeftTask \
# ButterAboveRaisinTask \
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

# GR00T-N1.7 Inference (LIBERO)
```bash
# 1st terminal
cd /workspace/Isaac-GR00T
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0
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
cd /workspace/Isaac-GR00T
bash gr00t/eval/sim/LIBERO/setup_libero.sh
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n-episodes 1 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n-action-steps 8 \
    --n-envs 1 \
    --video-dir /workspace/libero_rollouts

# libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
# libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
# libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket
# libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket
# libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket
# libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket
# libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket
# libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket
# libero_sim/pick_up_the_butter_and_place_it_in_the_basket
# libero_sim/pick_up_the_milk_and_place_it_in_the_basket
# libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket
# libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket
# libero_sim/open_the_middle_drawer_of_the_cabinet
# libero_sim/put_the_bowl_on_the_stove
# libero_sim/put_the_wine_bottle_on_top_of_the_cabinet
# libero_sim/open_the_top_drawer_and_put_the_bowl_inside
# libero_sim/put_the_bowl_on_top_of_the_cabinet
# libero_sim/push_the_plate_to_the_front_of_the_stove
# libero_sim/put_the_cream_cheese_in_the_bowl
# libero_sim/turn_on_the_stove
# libero_sim/put_the_bowl_on_the_plate
# libero_sim/put_the_wine_bottle_on_the_rack
# libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
# libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
# libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
# libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
# libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
# libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
# libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
# libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
# libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
# libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
# libero_sim/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it
# libero_sim/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it
# libero_sim/KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it
# libero_sim/KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it
# libero_sim/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it
# libero_sim/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate
# libero_sim/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet
# libero_sim/KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate
# libero_sim/KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate
# libero_sim/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate
# libero_sim/KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet
# libero_sim/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle
# libero_sim/KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl
# libero_sim/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove
# libero_sim/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove
# libero_sim/KITCHEN_SCENE3_turn_on_the_stove
# libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it
# libero_sim/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer
# libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet
# libero_sim/KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack
# libero_sim/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate
# libero_sim/KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet
# libero_sim/KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet
# libero_sim/KITCHEN_SCENE6_close_the_microwave
# libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug
# libero_sim/KITCHEN_SCENE7_open_the_microwave
# libero_sim/KITCHEN_SCENE7_put_the_white_bowl_on_the_plate
# libero_sim/KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate
# libero_sim/KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove
# libero_sim/KITCHEN_SCENE8_turn_off_the_stove
# libero_sim/KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf
# libero_sim/KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet
libero_sim/KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf
libero_sim/KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet
libero_sim/KITCHEN_SCENE9_turn_on_the_stove
libero_sim/KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it
libero_sim/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket
libero_sim/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray
libero_sim/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray
libero_sim/LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray
libero_sim/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate
libero_sim/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate
libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate
libero_sim/LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate
libero_sim/LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate
libero_sim/LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate
libero_sim/LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate
libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate
libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy
libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy
libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy
libero_sim/STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy
libero_sim/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
libero_sim/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy
libero_sim/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy
libero_sim/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy
libero_sim/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy
libero_sim/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy
libero_sim/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy
libero_sim/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy
libero_sim/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy
libero_sim/STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf
libero_sim/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf
libero_sim/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf
libero_sim/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf
```

# GR00T-N1.7 inference (RoboCasa)
```bash

# 1st terminal
cd /workspace/Isaac-GR00T
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <finetuned-robocasa-checkpoint> \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper

# 2nd terminal
cd /workspace/Isaac-GR00T
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n-episodes 1 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-action-steps 8 \
    --n-envs 1 \
    --video-dir /workspace/robocasa_rollouts

# robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env
# robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env
# robocasa_panda_omron/CoffeePressButton_PandaOmron_Env
# robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env
# robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env
# robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env
# robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env
# robocasa_panda_omron/OpenDrawer_PandaOmron_Env
# robocasa_panda_omron/CloseDrawer_PandaOmron_Env
# robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env
# robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env
# robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env
# robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env
# robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env
# robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env
# robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env
# robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env
# robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env
# robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env
# robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env
# robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env
# robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env
# robocasa_panda_omron/TurnOnStove_PandaOmron_Env
# robocasa_panda_omron/TurnOffStove_PandaOmron_Env
```