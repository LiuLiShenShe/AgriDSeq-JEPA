#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python scripts/agriculture/run_phenobench.py --mode eval \
  --checkpoint runs/phenobench_dseq_vitb16_s20261110/checkpoints/phenobench_upernet_best.pth "$@"
