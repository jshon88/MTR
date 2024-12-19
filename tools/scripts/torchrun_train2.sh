#!/usr/bin/env bash

set -x

# Arguments
NGPUS=$1
MASTER_ADDR=$2
MASTER_PORT=$3
PY_ARGS=${@:4}

# Run torchrun with the provided master address and port
torchrun \
  --nproc_per_node=${NGPUS} \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=training_job \
  train.py --launcher pytorch ${PY_ARGS}