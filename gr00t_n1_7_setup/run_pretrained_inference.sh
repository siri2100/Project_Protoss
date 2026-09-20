#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$PROJECT_DIR/GR00T-N1.7"
cd "$REPO_DIR"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Run ./gr00t_n1_7_setup/setup_runpod.sh from the project root." >&2
    exit 1
fi

[[ -f .env.runpod ]] && source .env.runpod
source .venv/bin/activate
nvidia-smi >/dev/null

if ! hf auth whoami >/dev/null 2>&1; then
    echo "Run 'hf auth login' after getting access to nvidia/Cosmos-Reason2-2B." >&2
    exit 1
fi

python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --execution-horizon 8
