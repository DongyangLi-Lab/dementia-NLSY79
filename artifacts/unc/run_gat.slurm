#!/bin/bash
#SBATCH --job-name=hrs-gat
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-gat-%A_%a.log
set -euo pipefail
umask 077
module load python/3.12.4
. .venv-model/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
outcomes=(final any any)
horizons=(8 6 8)
outcome=${outcomes[$SLURM_ARRAY_TASK_ID]}
horizon=${horizons[$SLURM_ARRAY_TASK_ID]}
tag="${outcome}-h${horizon}-GAT-${SLURM_JOB_ID}"
nvidia-smi > "gpu-info-${tag}.txt"
python -c 'import torch; print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available()); assert torch.cuda.is_available(); print("gpu",torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "gpu-monitor-${tag}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
/usr/bin/time -v -o "profile-benchmark-${tag}.txt" python -u -m cluster.model_dispatch run benchmark "$outcome" "$horizon" GAT "result-benchmark-${outcome}-h${horizon}-GAT.zip"
sha256sum "result-benchmark-${outcome}-h${horizon}-GAT.zip"
