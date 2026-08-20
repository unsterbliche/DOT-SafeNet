#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DOTSAFENET_PYTHON:-python}"
SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"

# TensorFlow 2.10 needs the CUDA 11 libraries installed alongside the shared
# PyTorch runtime. Each model stage runs in its own process.
CUDA_LIBRARY_PATHS=(
  "$SITE_PACKAGES/nvidia/cudnn/lib"
  "$SITE_PACKAGES/nvidia/cublas/lib"
  "$SITE_PACKAGES/nvidia/cuda_runtime/lib"
  "/usr/local/cuda-11.8/targets/x86_64-linux/lib"
)
for directory in "${CUDA_LIBRARY_PATHS[@]}"; do
  if [[ -d "$directory" ]]; then
    export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done

export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
exec "$PYTHON" "$ROOT/scripts/profile_molecule.py" "$@"
