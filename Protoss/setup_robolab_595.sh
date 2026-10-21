#!/usr/bin/env bash
# Isolated, experimental Isaac Sim 6.0 stack. Never changes /workspace/RoboLab.
set -euo pipefail
PROTOSS_TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STACK=${ROBOLAB_595_ROOT:-/workspace/robolab-595}
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-Y}
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export UV_LINK_MODE=copy
MODE=${1:-setup}
if [ "$#" -gt 0 ]; then shift; fi

case "$MODE" in
setup)
  test "$(uname -s)" = Linux || { echo 'Linux RTX server required.' >&2; exit 1; }
  command -v uv >/dev/null
  mkdir -p "$STACK"
  checkout() {
    local url=$1 target=$2 revision=$3
    if [ ! -e "$target" ]; then
      git -c credential.helper= -c http.extraHeader= -c http.version=HTTP/1.1 \
        -c protocol.version=0 clone --no-checkout "$url" "$target"
      git -C "$target" checkout --detach "$revision"
    fi
    test "$(git -C "$target" rev-parse HEAD)" = "$revision" || {
      echo "Unexpected revision: $target (expected $revision). Use a new ROBOLAB_595_ROOT." >&2
      exit 1
    }
  }
  checkout https://github.com/NVlabs/RoboLab.git "$STACK/RoboLab" b6cfcf6ccd2113c4cd1216d35b33024a7d5691da
  checkout https://github.com/isaac-sim/IsaacLab.git "$STACK/IsaacLab" ffff603eafc6b74264a5261cc0183d6a65390d78
  apply_compatibility_patch() {
    local target=$1 patch=$2
    if git -C "$target" apply --reverse --check "$patch" 2>/dev/null; then
      echo "Already applied: $patch"
    else
      git -C "$target" apply --check "$patch"
      git -C "$target" apply "$patch"
    fi
  }
  # The API patch changes context in the original compatibility patch. Once
  # this second layer is present, do not try to reapply the first layer.
  API_PATCH="$PROTOSS_TOOLS/patches/robolab_isaac60_api.patch"
  if git -C "$STACK/RoboLab" apply --reverse --check "$API_PATCH" 2>/dev/null; then
    echo "Already applied: $API_PATCH"
  else
    apply_compatibility_patch "$STACK/RoboLab" "$PROTOSS_TOOLS/patches/robolab_isaac60.patch"
    apply_compatibility_patch "$STACK/RoboLab" "$API_PATCH"
  fi
  apply_compatibility_patch "$STACK/RoboLab" "$PROTOSS_TOOLS/patches/robolab_isaac60_contact_filters.patch"
  apply_compatibility_patch "$STACK/IsaacLab" "$PROTOSS_TOOLS/patches/isaaclab_60_dependencies.patch"
  python3 "$PROTOSS_TOOLS/src/isaaclab_optional_newton.py" "$STACK/IsaacLab"
  cp "$PROTOSS_TOOLS/src/robolab_gr00t_client.py" "$STACK/RoboLab/policies/gr00t/client.py"
  cp "$PROTOSS_TOOLS/src/robolab_isaac60_compat.py" "$STACK/RoboLab/robolab/protoss_isaac60.py"
  if [ ! -e "$STACK/.venv" ]; then uv venv --python 3.12 "$STACK/.venv"; fi
  "$STACK/.venv/bin/python" "$PROTOSS_TOOLS/src/robolab_595_environment.py" "$STACK"
  uv sync --project "$STACK" --python 3.12
  uv pip check --python "$STACK/.venv/bin/python"
  "$STACK/.venv/bin/python" -c \
    'import sys; from importlib.metadata import version; print(sys.executable); print("Isaac Sim", version("isaacsim")); print("Isaac Lab", version("isaaclab"))'
  echo "Installed experimental stack. Next: bash $PROTOSS_TOOLS/setup_robolab_595.sh smoke"
  ;;
smoke|eval)
  test -x "$STACK/.venv/bin/python" || { echo 'Run setup first.' >&2; exit 1; }
  # Fail before GPU initialization if setup's optional-backend fix did not land.
  python3 "$PROTOSS_TOOLS/src/isaaclab_optional_newton.py" "$STACK/IsaacLab" --check
  if ! git -C "$STACK/RoboLab" apply --reverse --check \
    "$PROTOSS_TOOLS/patches/robolab_isaac60_contact_filters.patch" 2>/dev/null; then
    echo 'Contact-filter fix is not applied. Run setup successfully before smoke/eval.' >&2
    exit 1
  fi
  cd "$STACK/RoboLab"
  mkdir -p "$STACK/logs"
  LOG="$STACK/logs/${MODE}-$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "RoboLab log: $LOG"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$LOG"
  "$STACK/.venv/bin/python" -c '
import sys
from importlib.metadata import version
from pathlib import Path
import robolab.constants as constants
sim = version("isaacsim")
print(f"RoboLab runtime: {sys.executable}; Python {sys.version.split()[0]}; Isaac Sim {sim}", flush=True)
actual = Path(constants.PACKAGE_DIR).resolve()
print(f"RoboLab source: {actual}", flush=True)
if actual != Path.cwd().resolve():
    raise SystemExit("RoboLab imported from a different checkout. Run setup again; check PYTHONPATH.")
if sys.version_info[:2] != (3, 12) or sim != "6.0.1.0":
    raise SystemExit("Expected Python 3.12 + Isaac Sim 6.0.1.0. Run setup again.")
' 2>&1 | tee -a "$LOG"
  if [ "$MODE" = smoke ]; then
    "$STACK/.venv/bin/python" -u "$PROTOSS_TOOLS/src/robolab_595_smoke.py" \
      --headless --device cuda:0 "$@" 2>&1 | tee -a "$LOG"
    if ! grep -q '^SMOKE PASS:' "$LOG"; then
      echo "Smoke did not complete. Check log: $LOG" >&2
      exit 1
    fi
  else
    # Kit may terminate without a completed episode. Exit status alone is not
    # enough to authorize post-processing an evaluation's result file.
    RESULT_NAME=
    EXPECT_RESULT_NAME=0
    for ARG in "$@"; do
      if [ "$EXPECT_RESULT_NAME" = 1 ]; then
        RESULT_NAME=$ARG
        EXPECT_RESULT_NAME=0
      else
        case "$ARG" in
          --output-folder-name|--output_folder_name) EXPECT_RESULT_NAME=1 ;;
          --output-folder-name=*|--output_folder_name=*) RESULT_NAME=${ARG#*=} ;;
        esac
      fi
    done
    "$STACK/.venv/bin/python" -u policies/gr00t/run.py "$@" 2>&1 | tee -a "$LOG"
    if [ -n "$RESULT_NAME" ] && [ ! -s "$STACK/RoboLab/output/$RESULT_NAME/episode_results.jsonl" ]; then
      echo "Evaluation produced no episode results: $STACK/RoboLab/output/$RESULT_NAME/episode_results.jsonl" >&2
      echo "Metric calculation/upload stopped. Check evaluation log: $LOG" >&2
      tail -n 80 "$LOG" >&2
      exit 1
    fi
  fi
  ;;
*) echo "Usage: bash $0 {setup|smoke|eval} [arguments...]" >&2; exit 2 ;;
esac
