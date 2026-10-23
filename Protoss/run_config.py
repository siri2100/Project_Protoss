"""Edit experiment settings here, then run: python3 Protoss/run.py <command>.

Paths are resolved from this checkout, including Project_Protoss2 on RunPod.
Tokens belong in the Hugging Face login cache, never in this file.
"""

if __package__:
    from .src.robolab_testset import TEST_TASKS, TEST_NUM_ENVS, TEST_NUM_RUNS
else:
    from src.robolab_testset import TEST_TASKS, TEST_NUM_ENVS, TEST_NUM_RUNS

GROOT_DIRECTORY = "Issac-GR00T-N17"
HF_HOME = "/workspace/.cache/huggingface"
CUDA_VISIBLE_DEVICES = "0"
ROBOLAB_STACK_ROOT = "/workspace/robolab-595"
ROBOLAB_51_ROOT = "/workspace/RoboLab"
HF_ACCOUNT = None                         # None: use the logged-in account; or an organization
DATASET_REPO = "protoss-groot-v16-shared-data"
DOWNLOAD_EPISODES = 300

# Shared LoRA/trainer options. Each model below can override these values.
TRAIN_COMMON = dict(
    lora_rank=32, lora_alpha=64, batch_size=2,
    gradient_accumulation=16, workers=2, seed=42,
)
TRAIN = {
    "FMD": dict(version="v1713", max_steps=500, save_steps=500, lr=1e-6),
    "S": dict(
        version="v1712", max_steps=2000, save_steps=500, lr=1e-6, workers=1,
        s_fmd_weight=1, s_sparc_weight=1e-3,
    ),
    "E": dict(
        version="v1712", max_steps=500, save_steps=500, lr=1e-5,
        e_fmd_weight=1, e_path_length_weight=0.01,
    ),
}

# Used by automatic evaluation after training and by the manual client.
EVALUATION = dict(
    tasks=list(TEST_TASKS), num_envs=TEST_NUM_ENVS, num_runs=TEST_NUM_RUNS,
    horizon=8, seed=0, device="cuda:0", server_timeout=600,
)

# Manual server/client and evaluate/upload commands select this saved model.
MODEL_NAME = "v1710_model_e_step500"
CHECKPOINT_STEP = 500                     # None: final exported adapter
LORA_MODE = "unmerged"                    # unmerged / merged / disabled / pretrained
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5557
ROBOLAB_STACK = "595"                     # Manual client only: 595 / 51
RESULT_NAME = None                        # None: fresh timestamped directory per evaluation
METRIC_START_NUMBER = 5
