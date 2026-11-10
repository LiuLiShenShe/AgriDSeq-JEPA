#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/agriculture/run_phenobench.py --mode train "$@"
