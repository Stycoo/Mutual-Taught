#!/bin/bash
# Pre-download the models and dataset of one or more configs into $HF_HOME (network only, no GPU):
#   bash cluster/pbs/download.sh configs/smoke.yaml
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set -eo pipefail
activate_env
for config in "$@"; do mutual-taught download --config "$config"; done
