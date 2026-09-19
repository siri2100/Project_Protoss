#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

usage() {
  cat <<'EOF'
Usage: ./run_docker.sh [run|build|shell|check|assets|stop]

  run             Build the image and create/start the ACG container (default).
  build           Build the ACG image only.
  shell           Open a shell in the running container.
  check           Check imports and GPU access in the running container.
  assets          Download RoboCasa assets in the running container.
  stop            Stop the container without deleting it.
EOF
}

action="${1:-run}"

if [[ "$action" == "-h" || "$action" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d ACG ]]; then
  echo "ACG directory is missing: $project_dir/ACG" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI was not found on PATH (host: $(uname -s)/$(uname -m))." >&2
  echo "Run this ACG CUDA setup on a Linux x86-64 host with an NVIDIA GPU," >&2
  echo "Docker Compose, and NVIDIA Container Toolkit installed." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is unavailable." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or is inaccessible." >&2
  exit 1
fi

case "$action" in
  build)
    docker compose build acg
    ;;
  run)
    docker compose build acg
    mkdir -p datasets outputs
    docker compose up -d --no-build acg
    docker compose ps acg
    ;;
  shell)
    docker compose exec acg bash
    ;;
  check)
    docker compose exec -T acg python -c \
      'import torch, robocasa; print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "NVIDIA GPU is unavailable"'
    ;;
  assets)
    docker compose exec -T acg python libs/robocasa/robocasa/scripts/download_kitchen_assets.py
    ;;
  stop)
    docker compose stop acg
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
