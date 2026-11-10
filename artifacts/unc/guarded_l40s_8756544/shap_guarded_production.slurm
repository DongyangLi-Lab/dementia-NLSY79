#!/bin/bash
#SBATCH --job-name=shap-guarded-production
#SBATCH --output=shap-guarded-production-%A_%a.log
set -euo pipefail
umask 077
module load python/3.12.4
export PYTHONPATH="$PWD" HF_HOME="$PWD/tabicl-hf" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 SHAP_MODE=production
printf '%s  %s\n' b3b8382947a048a04bb7cf7d3766f7f2b59dfea4faa147fcf0a17a9b64283d85 importance-shards/guarded-l40s-production/frozen-b3b8382947a0/worker.py | sha256sum -c -
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 10 > "importance-shards/guarded-l40s-production/gpu-monitor-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
mon=$!
trap 'kill "$mon" 2>/dev/null || true' EXIT
.venv-model/bin/python -u importance-shards/guarded-l40s-production/frozen-b3b8382947a0/worker.py
