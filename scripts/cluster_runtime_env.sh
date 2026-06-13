#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${PROJECT:-}" ]]; then
  PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

EXTERNAL_ROOT="$PROJECT/external"
export HOME="$EXTERNAL_ROOT/runtime_home"
export MAMBA_ROOT_PREFIX="$EXTERNAL_ROOT/cache/micromamba"
export CONDA_PKGS_DIRS="$EXTERNAL_ROOT/cache/conda/pkgs"
export PIP_CACHE_DIR="$EXTERNAL_ROOT/cache/pip"
export XDG_CACHE_HOME="$EXTERNAL_ROOT/cache/xdg"
export TORCH_HOME="$EXTERNAL_ROOT/cache/torch"
export HF_HOME="$EXTERNAL_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export MPLCONFIGDIR="$EXTERNAL_ROOT/cache/matplotlib"
export CUDA_CACHE_PATH="$EXTERNAL_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$EXTERNAL_ROOT/cache/triton"
export NUMBA_CACHE_DIR="$EXTERNAL_ROOT/cache/numba"
export IPYTHONDIR="$EXTERNAL_ROOT/cache/ipython"
export JUPYTER_CONFIG_DIR="$EXTERNAL_ROOT/cache/jupyter"
export UV_CACHE_DIR="$EXTERNAL_ROOT/cache/uv"
export PYTHONUSERBASE="$EXTERNAL_ROOT/python_user"
export PYTHONPYCACHEPREFIX="$EXTERNAL_ROOT/cache/pycache"
export TMPDIR="$EXTERNAL_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1

mkdir -p \
  "$HOME" \
  "$MAMBA_ROOT_PREFIX" \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$MPLCONFIGDIR" \
  "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" \
  "$NUMBA_CACHE_DIR" \
  "$IPYTHONDIR" \
  "$JUPYTER_CONFIG_DIR" \
  "$UV_CACHE_DIR" \
  "$PYTHONUSERBASE" \
  "$PYTHONPYCACHEPREFIX" \
  "$TMPDIR"
