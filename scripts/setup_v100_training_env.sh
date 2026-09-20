#!/usr/bin/env bash
set -euo pipefail

# Reproducible NNSLR training environment for NVIDIA Tesla V100 (Volta/sm_70).
# This script never installs/replaces NVIDIA drivers and does not run a GPU
# workload. Run "nnslr env --gpu-smoke" explicitly after installation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
PYTHON="${NNSLR_GPU_PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: python not found: $PYTHON" >&2
  exit 2
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# PyTorch 2.14 is the final release line publishing CUDA 12.6 binaries for
# Maxwell/Pascal/Volta. V100 needs Volta/sm_70; CUDA 13 binaries are not valid
# for this target.
"$VENV/bin/python" -m pip install   torch==2.14.0   torchvision==0.29.0   --index-url https://download.pytorch.org/whl/cu126

"$VENV/bin/python" -m pip install 'Pillow>=11,<13'
"$VENV/bin/python" -m pip install -e "$ROOT_DIR"

"$VENV/bin/python" - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("compiled architectures:", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print()
print("No GPU workload was run.")
print("Next: source .venv-gpu/bin/activate && nnslr env --gpu-smoke --gpu-device 0")
PY
