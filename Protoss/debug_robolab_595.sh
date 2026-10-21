#!/usr/bin/env bash
# RunPod-only entrypoint. Project_Protoss2 is provisioned by the user remotely.
set -euo pipefail
export PROTOSS_ROOT=${PROTOSS_DEBUG_ROOT:-/workspace/Project_Protoss2}
export ROBOLAB_595_ROOT=${ROBOLAB_595_ROOT:-/workspace/robolab-595}
RUNNER="$PROTOSS_ROOT/Protoss/setup_robolab_595.sh"
MODE=${1:-check}
if [ "$#" -gt 0 ]; then shift; fi
test -f "$RUNNER" || {
  echo "Debug checkout missing: $RUNNER. Put the updated project files in Project_Protoss2 on RunPod." >&2
  exit 1
}
echo "Debug project: $PROTOSS_ROOT"
echo "Simulator stack: $ROBOLAB_595_ROOT"
case "$MODE" in
  check)
    bash "$RUNNER" setup
    bash "$RUNNER" smoke "$@"
    ;;
  setup|smoke|eval) exec bash "$RUNNER" "$MODE" "$@" ;;
  *) echo "Usage: bash $0 {check|setup|smoke|eval} [arguments...]" >&2; exit 2 ;;
esac
