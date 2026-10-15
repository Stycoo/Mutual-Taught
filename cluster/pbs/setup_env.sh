#!/bin/bash
# Build the conda env (install + download only, no GPU needed):  bash cluster/pbs/setup_env.sh
# Runs nothing but conda/pip, so it is fine on login nodes that forbid computation.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set -eo pipefail
load_conda
FA_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
export MAX_JOBS=4  # if anything compiles, keep a shared login node usable

mkdir -p "$MT_WORK_DIR" "$MT_RUNS" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
if [ ! -x "$MT_ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$MT_ENV_PREFIX" --override-channels -c conda-forge python=3.12 pip
fi
activate_env
python -m pip install -e "$MT_ROOT[dev]"
python -m pip install "$FA_WHL"
python -m pip check || echo "WARNING: pip check reported problems (see above)"
echo "env ready: $MT_ENV_PREFIX (RUN_TESTS=1 with run.pbs checks it on a GPU node)"
