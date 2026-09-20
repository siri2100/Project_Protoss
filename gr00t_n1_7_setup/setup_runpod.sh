#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$PROJECT_DIR/GR00T-N1.7"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup must run on the Linux RunPod GPU pod." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found. Start this from a RunPod GPU pod." >&2
    exit 1
fi

if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
    echo "GR00T submodule is missing. Run: git submodule update --init --recursive" >&2
    exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then SUDO=(); else SUDO=(sudo); fi

echo "[1/4] Installing Git LFS..."
"${SUDO[@]}" apt-get update -qq
"${SUDO[@]}" apt-get install -y --no-install-recommends git-lfs
git lfs install

echo "[2/4] Downloading the bundled DROID demo data..."
git -C "$REPO_DIR" lfs pull --include="demo_data/droid_sample/**"

echo "[3/4] Preparing persistent caches in $WORKSPACE_DIR..."
mkdir -p "$WORKSPACE_DIR/.cache/huggingface" "$WORKSPACE_DIR/.cache/uv" "$WORKSPACE_DIR/gr00t-output"
cat > "$REPO_DIR/.env.runpod" <<EOF
export HF_HOME="$WORKSPACE_DIR/.cache/huggingface"
export UV_CACHE_DIR="$WORKSPACE_DIR/.cache/uv"
export UV_LINK_MODE=copy
export GR00T_OUTPUT_DIR="$WORKSPACE_DIR/gr00t-output"
EOF
# shellcheck disable=SC1091
source "$REPO_DIR/.env.runpod"

echo "[4/4] Installing GR00T N1.7 and its Python 3.12 environment..."
bash "$REPO_DIR/scripts/deployment/dgpu/install_deps.sh"
source "$REPO_DIR/.venv/bin/activate"
python -c "import gr00t, torch; print(f'GR00T ready; torch={torch.__version__}; CUDA={torch.cuda.is_available()}')"

echo "Next: cd $PROJECT_DIR && ./gr00t_n1_7_setup/run_pretrained_inference.sh"
