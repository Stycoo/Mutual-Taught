# Shared settings for PBS jobs; sourced by setup_env.sh and run.pbs.
MT_ROOT=${MT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
# export everything site.env sets (e.g. HF_TOKEN) so Python child processes see it
if [ -f "$MT_ROOT/cluster/pbs/site.env" ]; then set -a; source "$MT_ROOT/cluster/pbs/site.env"; set +a; fi
MT_WORK_DIR=${MT_WORK_DIR:-$HOME/mutual-taught}
MT_ENV_PREFIX=${MT_ENV_PREFIX:-$MT_WORK_DIR/env}
MT_RUNS=${MT_RUNS:-$MT_WORK_DIR/runs}
export HF_HOME=${HF_HOME:-$MT_WORK_DIR/hf_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$MT_WORK_DIR/pip_cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$MT_WORK_DIR/conda_pkgs}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$MT_WORK_DIR/vllm_cache}  # keep vLLM's compile cache off $HOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

load_conda() {
  if [ -n "${MT_CONDA_SH:-}" ]; then source "$MT_CONDA_SH"; else eval "$(conda shell.bash hook)"; fi
}

activate_env() {
  load_conda
  set +e; conda activate "$MT_ENV_PREFIX"; set -e  # conda's activate scripts are not set -e safe
}

# Some PBS installations set CUDA_VISIBLE_DEVICES to GPU UUIDs; map them to nvidia-smi indices.
fix_cuda_visible_devices() {
  [[ "${CUDA_VISIBLE_DEVICES:-}" == *GPU-* ]] || return 0
  local map ids=() u idx
  map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  for u in ${CUDA_VISIBLE_DEVICES//,/ }; do
    idx=$(awk -F', *' -v u="$u" '$2 == u {print $1}' <<< "$map")
    if [ -z "$idx" ]; then echo "cannot map $u to a GPU index; nvidia-smi says: $map" >&2; return 1; fi
    ids+=("$idx")
  done
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=$(printf '%s\n' "${ids[@]}" | sort -n | paste -sd,)
}
